"""CPU checks for the Fast workflow's graph, sampling contract, and model setup."""
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
    def setUp(self):
        self.workflow = builder.build_fast()
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
        path = ROOT / "workflows/minimax_h3_fast_workflow.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), self.workflow)
        self.assertNotEqual(self.workflow["id"], builder.build()["id"])

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

    def test_four_step_lora_and_shifted_model_reach_scheduler_and_guider(self):
        model = self.node("UNETLoader")
        lora = self.node("LoraLoaderModelOnly")
        shift = self.node("MiniMaxH3SigmaShift")
        guider = self.node("BasicGuider")
        scheduler = self.node("BasicScheduler")
        sampler = self.node("KSamplerSelect")
        sampling = self.node("SamplerCustomAdvanced")
        self.assertEqual(lora["widgets_values"], [FAST_LORA, 1])
        self.assertEqual(shift["widgets_values"], [6.0, 3.0])
        self.assertEqual(sampler["widgets_values"], ["euler"])
        self.assertEqual(scheduler["widgets_values"], ["simple", 4, 1])
        self.assertFalse(any(inp["name"] == "steps" for inp in scheduler["inputs"]),
                         "Fast steps must not be overridden by a linked switch")
        self.assertFalse(any(n["type"] == "ComfySwitchNode" for n in self.nodes.values()))
        self.assert_source(lora, "model", model, "MODEL")
        self.assert_source(shift, "model", lora, "MODEL")
        self.assert_source(guider, "model", shift, "MODEL")
        self.assert_source(scheduler, "model", shift, "MODEL")
        self.assert_source(sampling, "guider", guider, "GUIDER")
        self.assert_source(sampling, "sigmas", scheduler, "SIGMAS")
        self.assert_source(sampling, "sampler", sampler, "SAMPLER")

    def test_portrait_voiceover_carry_and_stitch_paths_are_preserved(self):
        portrait = self.node("LoadImage")
        audio = self.node("LoadAudio")
        crop = self.node("ImageScale")
        split = self.node("H3LongformSplit")
        carry = self.node("H3LongformCarry")
        image_to_video = self.node("MiniMaxH3ImageToVideo")
        guide = self.node("MiniMaxH3AddGuide")
        writer = self.node("H3LongformWrite")
        decode = self.node("VAEDecode")
        guider = self.node("BasicGuider")
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
        self.assert_source(split, "audio", audio, "AUDIO")
        self.assert_source(writer, "images", decode, "IMAGE")
        self.assert_source(writer, "original_audio", audio, "AUDIO")
        self.assert_source(writer, "chunk_audio", split, "audio_chunk")
        self.assert_source(writer, "trim_start", split, "carry_frames")
        self.assert_source(writer, "keep_frames", split, "keep_frames")
        self.assert_source(writer, "total_chunks", split, "total_chunks")
        index, _ = self.source(split, "chunk_index")
        self.assertEqual(index["widgets_values"], [0, "increment"])
        self.assert_source(carry, "chunk_index", index, "INT")
        self.assert_source(writer, "chunk_index", index, "INT")
        self.assertEqual(split["widgets_values"][-1], "off")

    def test_square_defaults_and_fast_names_keep_sessions_separate(self):
        crop = self.node("ImageScale")
        image_to_video = self.node("MiniMaxH3ImageToVideo")
        writer = self.node("H3LongformWrite")
        name = self.node("H3LongformAudioName")
        for dimension in ("width", "height"):
            control, output = self.source(image_to_video, dimension)
            self.assertEqual(control["widgets_values"], [768, "fixed"])
            self.assert_source(crop, dimension, control, output)
        self.assertEqual(crop["widgets_values"], ["lanczos", 768, 768, "center"])
        self.assertEqual(name["widgets_values"], [".mp4", "", "_fast"])
        self.assert_source(writer, "session", name, "name")
        self.assert_source(writer, "filename", name, "filename")
        self.assertEqual(writer["widgets_values"][3:5], ["run1_fast", "h3_fast_final.mp4"])

    @unittest.skipUnless(BASH and Path(BASH).exists(), "Bash required for default download plan")
    def test_required_model_files_are_in_the_default_download_plan(self):
        # Execute the shipped model selection block with a recording downloader;
        # this catches a required Fast weight accidentally becoming opt-in.
        setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
        plan = setup[setup.index("# 3. Model downloads"):setup.index("# 4. Install bundled workflows")]
        with tempfile.TemporaryDirectory(prefix="h3-fast-plan-") as directory:
            script = Path(directory) / "plan.sh"
            script.write_text("set -e\nH3_TEXT_ENCODER=nvfp4\nDOWNLOAD_TURBO_LORA=1\nDOWNLOAD_REF2VA=0\n"
                              + "download_h3() { printf '%s\\n' \"$1\"; }\n" + plan,
                              encoding="utf-8", newline="\n")
            result = subprocess.run([BASH, str(script)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        downloads = set(result.stdout.splitlines())
        required = {item["directory"] + "/" + item["name"] for node in self.nodes.values()
                    for item in node["properties"].get("models", [])}
        self.assertEqual(len(required), 5)
        self.assertIn("loras/" + FAST_LORA, required)
        self.assertLessEqual(required, downloads)
        for node in self.nodes.values():
            for item in node["properties"].get("models", []):
                self.assertEqual(node["widgets_values"][0], item["name"])
                self.assertEqual(item["url"], builder.HF + item["directory"] + "/" + item["name"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
