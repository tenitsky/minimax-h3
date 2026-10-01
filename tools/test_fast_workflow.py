"""CPU checks for all six graphs (standard / Fast / Draft, single and batch), sampling
settings, layout and model setup."""
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("builder", ROOT / "tools/build_workflow.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
BASH = os.environ.get("TEST_BASH") or ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash"))
FAST_LORA = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"


class FastWorkflowTests(unittest.TestCase):
    variant = "fast"
    batch = False
    workflow_filename = "minimax_h3_talking_head_fast.json"
    lora_filename = FAST_LORA
    lora_url = builder.HF + "loras/"
    video_shift = 6.0
    steps = 4
    sparse_enabled = True
    canvas = 768
    name_suffix = "_fast"
    fallback_names = ["run1_fast", "h3_talking_head_fast.mp4"]

    def setUp(self):
        self.workflow = builder.build(self.variant, self.batch)
        self.nodes = {node["id"]: node for node in self.workflow["nodes"]}
        self.links = {link[0]: link for link in self.workflow["links"]}

    def node(self, node_type):
        matches = [node for node in self.nodes.values() if node["type"] == node_type]
        self.assertEqual(len(matches), 1, f"Expected one {node_type}, found {len(matches)}")
        return matches[0]

    def source(self, node, name):
        inp = next(item for item in node["inputs"] if item["name"] == name)
        self.assertIsNotNone(inp["link"], f"Disconnected {node['type']}.{name}")
        _, src, slot, dst, port, _ = self.links[inp["link"]]
        self.assertEqual((dst, node["inputs"][port]["name"]), (node["id"], name))
        return self.nodes[src], self.nodes[src]["outputs"][slot]["name"]

    def assert_source(self, node, name, expected, output):
        actual, actual_output = self.source(node, name)
        self.assertEqual((actual["id"], actual_output), (expected["id"], output))

    def test_shipped_workflow_matches_builder(self):
        path = ROOT / "workflows" / self.workflow_filename
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), self.workflow)
        ids = {w["id"] for w in builder.WORKFLOWS}
        self.assertEqual(len(ids), len(builder.WORKFLOWS), "Workflow IDs must differ")
        self.assertEqual(self.workflow["id"], next(
            w["id"] for w in builder.WORKFLOWS if w["path"] == self.workflow_filename))
        shipped = {p.name for p in (ROOT / "workflows").glob("*.json")}
        self.assertEqual(shipped, {w["path"] for w in builder.WORKFLOWS})

    def test_layout_is_spaced_grouped_and_expanded(self):
        title = 40  # LiteGraph draws the title bar above pos
        boxes = {nid: (n["pos"][0], n["pos"][1] - title, n["pos"][0] + n["size"][0],
                       n["pos"][1] + n["size"][1])
                 for nid, n in self.nodes.items()}
        for nid, node in self.nodes.items():
            self.assertNotEqual(node["flags"].get("collapsed"), True, node["type"])
        ids = sorted(boxes)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ax0, ay0, ax1, ay1 = boxes[a]
                bx0, by0, bx1, by1 = boxes[b]
                overlap = ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1
                self.assertFalse(overlap, f"{self.nodes[a]['type']} overlaps {self.nodes[b]['type']}")
        groups = self.workflow["groups"]
        self.assertEqual([g["title"][:2] for g in groups], [f"{i}." for i in range(1, 8)])
        for nid, node in self.nodes.items():
            if node["type"] == "MarkdownNote":
                continue
            x0, y0, x1, y1 = boxes[nid]
            self.assertTrue(any(g["bounding"][0] <= x0 and x1 <= g["bounding"][0] + g["bounding"][2]
                                and g["bounding"][1] <= y0 and y1 <= g["bounding"][1] + g["bounding"][3]
                                for g in groups), f"{node['type']} is outside every stage group")

    def test_all_links_are_live_reciprocal_typed_and_acyclic(self):
        self.assertEqual(len(self.nodes), len(self.workflow["nodes"]), "Duplicate node IDs")
        self.assertEqual(len(self.links), len(self.workflow["links"]), "Duplicate link IDs")
        inputs_used = set()
        children = {nid: [] for nid in self.nodes}
        parents = {nid: set() for nid in self.nodes}
        indegree = {nid: 0 for nid in self.nodes}
        for lid, src, slot, dst, port, kind in self.workflow["links"]:
            self.assertIn(src, self.nodes)
            self.assertIn(dst, self.nodes)
            out = self.nodes[src]["outputs"][slot]
            inp = self.nodes[dst]["inputs"][port]
            self.assertEqual(out["type"], kind)
            self.assertIn(inp["type"], (kind, "*"))
            self.assertEqual(inp["link"], lid)
            self.assertEqual(out["links"].count(lid), 1)
            self.assertNotIn((dst, port), inputs_used, "Multiple links drive the same input")
            inputs_used.add((dst, port))
            children[src].append(dst)
            parents[dst].add(src)
            indegree[dst] += 1

        for nid, node in self.nodes.items():
            for slot, out in enumerate(node["outputs"]):
                for lid in out["links"]:
                    self.assertIn(lid, self.links, "Output references a removed link")
                    self.assertEqual(self.links[lid][1:3], [nid, slot])
            for port, inp in enumerate(node["inputs"]):
                if inp["link"] is not None:
                    self.assertIn(inp["link"], self.links, "Input references a removed link")
                    self.assertEqual(self.links[inp["link"]][3:5], [nid, port])

        ready = deque(nid for nid, count in indegree.items() if count == 0)
        visited = set()
        while ready:
            src = ready.popleft()
            visited.add(src)
            for dst in children[src]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
        self.assertEqual(visited, set(self.nodes), "Workflow contains a dependency cycle")

        # Apart from Markdown notes, every node must contribute to the output.
        ancestors = {self.node("H3LongformWrite")["id"]}
        pending = list(ancestors)
        while pending:
            for parent in parents[pending.pop()]:
                if parent not in ancestors:
                    ancestors.add(parent)
                    pending.append(parent)
        self.assertEqual(ancestors, {nid for nid, node in self.nodes.items()
                                     if node["type"] != "MarkdownNote"})

    def assert_model_path(self):
        """LoRA -> its trained shift -> the sampling model; returns the shift node."""
        model = self.node("UNETLoader")
        lora = self.node("LoraLoaderModelOnly")
        shift = self.node("MiniMaxH3SigmaShift")
        self.assertEqual(lora["widgets_values"], [self.lora_filename, 1])
        self.assertEqual(shift["widgets_values"], [self.video_shift, 3.0])
        self.assert_source(lora, "model", model, "MODEL")
        self.assert_source(shift, "model", lora, "MODEL")
        return shift

    def test_lora_shift_and_accelerated_model_reach_scheduler_and_guider(self):
        shift = self.assert_model_path()
        backend = self.node("ModelAttentionBackend")
        sparse = self.node("BlockSparseAttention")
        guider = self.node("BasicGuider")
        scheduler = self.node("BasicScheduler")
        sampler = self.node("KSamplerSelect")
        sampling = self.node("SamplerCustomAdvanced")
        self.assertEqual(sampler["widgets_values"], ["euler"])
        self.assertEqual(scheduler["widgets_values"], ["simple", self.steps, 1])
        self.assertFalse(any(inp["name"] == "steps" for inp in scheduler["inputs"]),
                         "Fast steps must not be overridden by a linked switch")
        self.assertFalse(any(n["type"] == "ComfySwitchNode" for n in self.nodes.values()))
        self.assert_source(backend, "model", shift, "MODEL")
        self.assert_source(guider, "model", sparse, "model")
        self.assert_source(scheduler, "model", sparse, "model")
        self.assert_source(sampling, "guider", guider, "GUIDER")
        self.assert_source(sampling, "sigmas", scheduler, "SIGMAS")
        self.assert_source(sampling, "sampler", sampler, "SAMPLER")

    def test_acceleration_settings(self):
        backend = self.node("ModelAttentionBackend")
        sparse = self.node("BlockSparseAttention")
        self.assertEqual(backend["widgets_values"], ["comfy kitchen attention"])
        self.assert_source(sparse, "model", backend, "model")
        self.assertEqual(sparse["widgets_values"][0], "sol-attn")
        # Pinned portraits, text and voiceover stay exact for every query.
        self.assertEqual(sparse["widgets_values"][7], "exact_kv_and_rows")
        self.assertEqual(sparse["mode"], 0 if self.sparse_enabled else 4)

    def test_generated_prompt_follows_chunk_plan(self):
        prompt = self.node("H3LongformPrompt")
        split = self.node("H3LongformSplit")
        image_to_video = self.node("MiniMaxH3ImageToVideo")
        self.assert_source(image_to_video, "prompt", prompt, "prompt")
        self.assert_source(prompt, "length", split, "length")
        self.assert_source(prompt, "carry_frames", split, "carry_frames")
        self.assertFalse(split["outputs"][5]["links"], "alignment is written by the prompt node")
        self.assertEqual(split["widgets_values"][1:4], [8.0, 5.0, 10.0])

    def test_portrait_voiceover_carry_and_stitch_paths_are_preserved(self):
        portrait = self.node("LoadImage")
        if self.batch:
            audio, audio_out = self.node("H3LongformAudioFolder"), "audio"
        else:
            audio, audio_out = self.node("LoadAudio"), "AUDIO"
        crop = self.node("ImageScale")
        split = self.node("H3LongformSplit")
        carry = self.node("H3LongformCarry")
        image_to_video = self.node("MiniMaxH3ImageToVideo")
        guide = self.node("MiniMaxH3AddGuide")
        writer = self.node("H3LongformWrite")
        decode = self.node("VAEDecode")
        guider = self.node("BasicGuider")
        sampling = self.node("SamplerCustomAdvanced")
        video_vae, _ = self.source(image_to_video, "vae")
        audio_vae, _ = self.source(guide, "audio_vae")
        self.assertEqual(video_vae["widgets_values"], ["minimax_h3_video_vae_int8_convrot.safetensors"])
        self.assertEqual(audio_vae["widgets_values"], ["minimax_h3_audio_vae_fp32.safetensors"])
        self.assert_source(guide, "vae", video_vae, "VAE")
        self.assert_source(decode, "vae", video_vae, "VAE")
        self.assert_source(decode, "samples", sampling, "output")
        self.assert_source(sampling, "latent_image", image_to_video, "LATENT")
        self.assert_source(crop, "image", portrait, "IMAGE")
        self.assert_source(carry, "portrait", crop, "IMAGE")
        self.assert_source(carry, "carry_frames", split, "carry_frames")
        self.assert_source(image_to_video, "first_frame", carry, "first_frame")
        self.assert_source(image_to_video, "last_frame", crop, "IMAGE")
        self.assert_source(image_to_video, "length", split, "length")
        self.assert_source(guide, "image", carry, "carry_clip")
        self.assert_source(guide, "audio", split, "guide_audio")
        self.assert_source(guide, "positive", image_to_video, "positive")
        self.assert_source(guide, "latent", image_to_video, "LATENT")
        self.assert_source(guider, "conditioning", guide, "positive")
        self.assert_source(split, "audio", audio, audio_out)
        self.assert_source(writer, "images", decode, "IMAGE")
        self.assert_source(writer, "original_audio", audio, audio_out)
        self.assert_source(writer, "chunk_audio", split, "audio_chunk")
        self.assert_source(writer, "trim_start", split, "carry_frames")
        self.assert_source(writer, "keep_frames", split, "keep_frames")
        self.assert_source(writer, "total_chunks", split, "total_chunks")
        # The chunk to render comes from the files on disk, not a browser counter.
        self.assertFalse(any(i["name"] == "chunk_index" for i in split["inputs"]))
        self.assertEqual(split["widgets_values"][6:], ["off", True])
        self.assert_source(carry, "chunk_index", split, "chunk_index")
        self.assert_source(writer, "chunk_index", split, "chunk_index")
        self.assertFalse(any(n["type"] == "PrimitiveInt" and n["outputs"][0]["links"]
                             and "chunk" in n.get("title", "") for n in self.nodes.values()))

    def test_square_defaults_and_fast_names_keep_sessions_separate(self):
        crop = self.node("ImageScale")
        image_to_video = self.node("MiniMaxH3ImageToVideo")
        writer = self.node("H3LongformWrite")
        for dimension in ("width", "height"):
            control, output = self.source(image_to_video, dimension)
            self.assertEqual(control["widgets_values"], [self.canvas, "fixed"])
            self.assert_source(crop, dimension, control, output)
        self.assertEqual(crop["widgets_values"], ["lanczos", self.canvas, self.canvas, "center"])
        self.assertEqual(image_to_video["widgets_values"][1:3], [self.canvas, self.canvas])
        if self.batch:
            name = self.node("H3LongformAudioFolder")
            self.assertEqual(name["widgets_values"], ["batch_audio", self.name_suffix, True])
            # Split and Carry must use the same per-file folder as Write.
            self.assert_source(self.node("H3LongformSplit"), "session", name, "name")
            self.assert_source(self.node("H3LongformCarry"), "session", name, "name")
            fallback = ["batch" + self.name_suffix, "batch" + self.name_suffix + ".mp4"]
        else:
            name = self.node("H3LongformAudioName")
            self.assertEqual(name["widgets_values"], [".mp4", "", self.name_suffix])
            fallback = self.fallback_names
        self.assert_source(writer, "session", name, "name")
        self.assert_source(writer, "filename", name, "filename")
        self.assertEqual(writer["widgets_values"][3:5], fallback)
        # In a batch, only the folder node may clear the queue (after the last file).
        self.assertEqual(writer["widgets_values"][5], not self.batch)

    @unittest.skipUnless(BASH and Path(BASH).exists(), "Bash required for default download plan")
    def test_required_model_files_are_in_the_default_download_plan(self):
        # Execute the shipped model selection block with a recording downloader;
        # this catches a required Fast weight accidentally becoming opt-in.
        setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
        plan = setup[setup.index("# 3. Model downloads"):setup.index("# 4. Install bundled workflows")]
        with tempfile.TemporaryDirectory(prefix="h3-fast-plan-") as directory:
            script = Path(directory) / "plan.sh"
            script.write_text("set -e\nH3_TEXT_ENCODER=nvfp4\nDOWNLOAD_TURBO_LORA=1\nDOWNLOAD_REF2VA=0\n"
                              + "download_h3() { printf '%s\\n' \"$1\"; }\n"
                              + "download_turbo() { printf 'loras/%s\\n' \"$1\"; }\n" + plan,
                              encoding="utf-8", newline="\n")
            result = subprocess.run([BASH, str(script)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        downloads = set(result.stdout.splitlines())
        required = {item["directory"] + "/" + item["name"] for node in self.nodes.values()
                    for item in node["properties"].get("models", [])}
        self.assertEqual(len(required), 5)
        self.assertIn("loras/" + self.lora_filename, required)
        self.assertLessEqual(required, downloads)
        for node in self.nodes.values():
            for item in node["properties"].get("models", []):
                self.assertEqual(node["widgets_values"][0], item["name"])
                base = (self.lora_url if item["directory"] == "loras"
                        else builder.HF + item["directory"] + "/")
                self.assertEqual(item["url"], base + item["name"])


class FastDraftWorkflowTests(FastWorkflowTests):
    variant = "draft"
    workflow_filename = "minimax_h3_talking_head_draft.json"
    lora_filename = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
    video_shift = 12.0
    canvas = 544
    name_suffix = "_draft"
    fallback_names = ["run1_draft", "h3_talking_head_draft.mp4"]


class StandardWorkflowTests(FastWorkflowTests):
    variant = "standard"
    workflow_filename = "minimax_h3_talking_head.json"
    lora_filename = "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
    lora_url = builder.TURBO_HF
    steps = 8
    sparse_enabled = False
    name_suffix = ""
    fallback_names = ["run1", "h3_talking_head.mp4"]

    def test_lora_shift_and_accelerated_model_reach_scheduler_and_guider(self):
        shift = self.assert_model_path()
        model = self.node("UNETLoader")
        backend = self.node("ModelAttentionBackend")
        sparse = self.node("BlockSparseAttention")
        scheduler = self.node("BasicScheduler")
        switches = {self.source(n, "on_true")[0]["type"]: n for n in self.nodes.values()
                    if n["type"] == "ComfySwitchNode"}
        model_switch, step_switch = switches["MiniMaxH3SigmaShift"], switches["PrimitiveInt"]
        # Turbo on: 768p LoRA at its trained 6/3 shift. Off: base model, default shift.
        self.assert_source(model_switch, "on_true", shift, "MODEL")
        self.assert_source(model_switch, "on_false", model, "MODEL")
        self.assertEqual(self.source(step_switch, "on_true")[0]["widgets_values"], [8, "fixed"])
        self.assertEqual(self.source(step_switch, "on_false")[0]["widgets_values"], [20, "fixed"])
        self.assert_source(scheduler, "steps", step_switch, "output")
        self.assert_source(backend, "model", model_switch, "output")
        self.assert_source(self.node("BasicGuider"), "model", sparse, "model")
        self.assert_source(scheduler, "model", sparse, "model")
        self.assertEqual(self.node("KSamplerSelect")["widgets_values"], ["euler"])



class FastBatchWorkflowTests(FastWorkflowTests):
    batch = True
    workflow_filename = "minimax_h3_talking_head_fast_batch.json"


class DraftBatchWorkflowTests(FastDraftWorkflowTests):
    batch = True
    workflow_filename = "minimax_h3_talking_head_draft_batch.json"


class StandardBatchWorkflowTests(StandardWorkflowTests):
    batch = True
    workflow_filename = "minimax_h3_talking_head_batch.json"


if __name__ == "__main__":
    unittest.main(verbosity=2)
