"""CPU regression checks: python tools/test_longform.py (torch, numpy, ffmpeg)."""
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


class ExecutionBlocker:
    def __init__(self, value):
        self.value = value


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Only the ComfyUI host is stubbed; tensors, audio and media tools are real.
folder_paths = types.ModuleType("folder_paths")
sys.modules["folder_paths"] = folder_paths
sys.modules["comfy_execution"] = types.ModuleType("comfy_execution")
graph_utils = types.ModuleType("comfy_execution.graph_utils")
graph_utils.ExecutionBlocker = ExecutionBlocker
sys.modules["comfy_execution.graph_utils"] = graph_utils
h3 = load("h3", ROOT / "custom_nodes/comfyui-h3-longform/__init__.py")
builder = load("builder", ROOT / "tools/build_workflow.py")


class LongformTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-tests-")
        self.addCleanup(self.tmp.cleanup)
        folder_paths.get_output_directory = lambda: self.tmp.name

    def prompt(self, mode, session="test"):
        return {
            "20": {"class_type": "H3LongformSplit", "inputs": {"motion_carry": mode}},
            "70": {"class_type": "H3LongformWrite", "inputs": {"session": session}},
        }

    def test_planner_coverage_and_generation_bounds(self):
        rng = random.Random(42)
        for carry in (0, 5, 22):
            for mode in ("pause", "silence", "fixed"):
                for _ in range(300):
                    minimum = rng.uniform(5, 15)
                    maximum = rng.uniform(minimum, 15)
                    total = rng.randrange(1, 24 * 600)
                    first = h3.keep_grid(minimum, maximum)
                    keeps = h3.keep_grid(minimum, maximum, carry)
                    pauses = [(p, p + 16) for p in range(130, total, 220)]
                    spans = h3.plan_chunks(total, 288, keeps, pauses, mode, first)
                    pos = 0
                    for i, (start, keep) in enumerate(spans):
                        lead = carry if i else 0
                        last = i == len(spans) - 1
                        length = h3.gen_length(keep, last, lead)
                        self.assertEqual(start, pos)
                        self.assertGreater(keep, 0)
                        self.assertLessEqual(keep + lead, length)
                        self.assertTrue(124 <= length <= 362)
                        self.assertEqual(length % 17, 5)
                        if not last:
                            self.assertEqual(length, keep + lead + 1)
                        pos += keep
                    self.assertEqual(pos, total)

    def test_workflow_carry_connections_and_schemas(self):
        wf = builder.build()
        self.assertEqual(wf, json.loads((ROOT / "workflows/minimax_h3_long_video_workflow.json").read_text(encoding="utf-8")))
        nodes = {n["id"]: n for n in wf["nodes"]}
        links = {}
        for lid, src, slot, dst, port, kind in wf["links"]:
            out, inp = nodes[src]["outputs"][slot], nodes[dst]["inputs"][port]
            self.assertEqual(out["type"], kind)
            self.assertEqual(inp["type"], kind)
            self.assertEqual(inp["link"], lid)
            self.assertIn(lid, out["links"])
            links[(dst, inp["name"])] = (src, slot)
        for dest, source in {
            (51, "first_frame"): (53, 0), (51, "last_frame"): (50, 0),
            (52, "image"): (53, 1), (52, "audio"): (20, 6),
            (52, "vae"): (43, 0), (53, "carry_frames"): (20, 7),
            (70, "trim_start"): (20, 7), (70, "chunk_audio"): (20, 0),
            (70, "original_audio"): (11, 0),
        }.items():
            self.assertEqual(links[dest], source)
        for node in nodes.values():
            cls = h3.NODE_CLASS_MAPPINGS.get(node["type"])
            if cls is None:
                continue
            schema = cls.INPUT_TYPES()
            inputs = {**schema.get("required", {}), **schema.get("optional", {})}
            self.assertEqual(tuple(o["type"] for o in node["outputs"]), cls.RETURN_TYPES)
            for inp in node["inputs"]:
                self.assertIn(inp["name"], inputs)
            widgets = [v for v in inputs.values() if isinstance(v[0], list) or v[0] in ("INT", "FLOAT", "BOOLEAN", "STRING")]
            self.assertEqual(len(widgets), len(node["widgets_values"]))
        self.assertEqual(nodes[20]["widgets_values"][-1], "off")

    def probe(self, path, *args):
        return json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", *args, "-of", "json", str(path)
        ], text=True))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
    def test_motion_audio_trim_stitch_and_resume(self):
        sr, seconds = 44100, 31
        audio = {"waveform": torch.linspace(-0.5, 0.5, sr * seconds).reshape(1, 1, -1), "sample_rate": sr}
        portrait = torch.zeros((1, 16, 16, 3))
        for mode, carry in h3.CARRY_OPTIONS.items():
            with self.subTest(mode=mode), contextlib.redirect_stdout(io.StringIO()):
                session = f"speaker's take {carry}"
                prompt = self.prompt(mode, session)
                split, opening, writer = h3.H3LongformSplit(), h3.H3LongformCarry(), h3.H3LongformWrite()
                offset, index, pieces = 0, 0, []
                while True:
                    result = split.split(audio, index, 12, 5, 15, motion_carry=mode, prompt=prompt)
                    chunk, length, keep, count, last, alignment, guide, lead = result
                    self.assertEqual(lead, carry if index else 0)
                    start_sample = round(offset * sr / 24)
                    end_sample = min(sr * seconds, round((offset + keep) * sr / 24))
                    guide_sample = round((offset - lead) * sr / 24)
                    self.assertTrue(torch.equal(chunk["waveform"][:, :1], audio["waveform"][..., start_sample:end_sample]))
                    self.assertTrue(torch.equal(guide["waveform"][:, :1], audio["waveform"][..., guide_sample:end_sample]))
                    self.assertEqual(chunk["waveform"].shape[1], 2)
                    first, clip = opening.pick(portrait, index, lead, prompt)
                    if lead:
                        self.assertIsNone(first)
                        with np.load(h3._tail_path(session, index - 1)) as saved:
                            np.testing.assert_allclose(clip.numpy(), saved["frames"][-lead:] / 255, atol=1e-7)
                        self.assertNotIn("Picture 2", alignment)
                    else:
                        self.assertIs(first, portrait)
                        self.assertIsNone(clip)
                    images = torch.linspace(0, 1, length).reshape(-1, 1, 1, 1).expand(-1, 16, 16, 3)
                    writer.write(images, audio, index, count, keep, session, f"final_{carry}.mp4", chunk, False, lead, prompt)
                    chunk_path = Path(h3._session_dir(session)) / f"chunk_{index:04d}.mp4"
                    info = self.probe(chunk_path, "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames")
                    self.assertEqual(int(info["streams"][0]["nb_read_frames"]), keep)
                    if carry and not last:
                        with np.load(h3._tail_path(session, index)) as saved:
                            expected = (images[lead:lead + keep][-carry:].numpy() * 255).astype(np.uint8)
                            np.testing.assert_array_equal(saved["frames"], expected)
                    pieces.append(chunk["waveform"])
                    offset += keep
                    if last:
                        break
                    index += 1
                self.assertEqual(offset, seconds * 24)
                self.assertTrue(torch.equal(torch.cat(pieces, dim=-1)[:, :1], audio["waveform"]))
                final = Path(self.tmp.name) / f"final_{carry}.mp4"
                info = self.probe(final, "-count_frames", "-show_entries", "stream=codec_type,nb_read_frames,duration")
                video = next(s for s in info["streams"] if s["codec_type"] == "video")
                self.assertEqual(int(video["nb_read_frames"]), seconds * 24)
                self.assertAlmostEqual(float(video["duration"]), seconds, delta=1 / 24)
                frames = self.probe(final, "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time")["frames"]
                times = [float(f["best_effort_timestamp_time"]) for f in frames]
                self.assertLess(max(abs(t - i / 24) for i, t in enumerate(times)), 0.001)
                self.assertIsInstance(split.split(audio, 0, 12, 5, 15, motion_carry=mode, prompt=prompt)[0], ExecutionBlocker)
                self.assertIsInstance(split.split(audio, count, 12, 5, 15, motion_carry=mode, prompt=prompt)[0], ExecutionBlocker)
                if carry:
                    # A final-chunk retry can still load the preceding tail.
                    self.assertIsNotNone(opening.pick(portrait, count - 1, carry, prompt)[1])
                    Path(h3._tail_path(session, 0)).unlink()
                    self.assertIsInstance(split.split(audio, 0, 12, 5, 15, motion_carry=mode, prompt=prompt)[0], dict)
                    with self.assertRaisesRegex(RuntimeError, "Queue chunks in order"):
                        opening.pick(portrait, 1, carry, prompt)
                    np.savez_compressed(h3._tail_path(session, 0), frames=np.zeros((1, 16, 16, 3), dtype=np.uint8))
                    self.assertIsInstance(split.split(audio, 0, 12, 5, 15, motion_carry=mode, prompt=prompt)[0], dict)
                    with self.assertRaisesRegex(RuntimeError, "Queue chunks in order"):
                        opening.pick(portrait, 1, carry, prompt)

    def test_resume_cache_is_invalidated(self):
        self.assertTrue(math.isnan(h3.H3LongformSplit.IS_CHANGED()))

    def test_audio_name_suffix_keeps_existing_names_and_separates_fast_resume(self):
        normal = self.prompt("off", ["12", 0])
        normal["11"] = {"class_type": "LoadAudio", "inputs": {"audio": "uploads/my voice.wav"}}
        # Existing workflows have no name_suffix field, and keep their old names.
        normal["12"] = {"class_type": "H3LongformAudioName", "inputs": {"suffix": ".mp4", "source_title": ""}}
        fast = json.loads(json.dumps(normal))
        fast["12"]["inputs"]["name_suffix"] = "_fast"
        namer = h3.H3LongformAudioName()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(namer.derive(".mp4", "", normal), ("my voice", "my voice.mp4"))
            self.assertEqual(namer.derive(".mp4", "", fast, name_suffix="_fast"),
                             ("my voice_fast", "my voice_fast.mp4"))
            self.assertEqual(h3._session_from_prompt(normal), "my voice")
            self.assertEqual(h3._session_from_prompt(fast), "my voice_fast")
            self.assertNotEqual(namer.IS_CHANGED(".mp4", prompt=normal),
                                namer.IS_CHANGED(".mp4", prompt=fast, name_suffix="_fast"))
            audio = {"waveform": torch.zeros((1, 1, 24000 * 15)), "sample_rate": 24000}
            completed = Path(h3._session_dir("my voice")) / "chunk_0000.mp4"
            completed.write_bytes(b"existing normal chunk")
            split = h3.H3LongformSplit()
            self.assertIsInstance(split.split(audio, 0, 5, 5, 5, prompt=normal)[0], ExecutionBlocker)
            self.assertIsInstance(split.split(audio, 0, 5, 5, 5, prompt=fast)[0], dict)
            fast_completed = Path(h3._session_dir("my voice_fast")) / "chunk_0000.mp4"
            fast_completed.write_bytes(b"existing fast chunk")
            self.assertIsInstance(split.split(audio, 0, 5, 5, 5, prompt=fast)[0], ExecutionBlocker)
            self.assertEqual(completed.read_bytes(), b"existing normal chunk")

    def test_audio_name_and_session_resolver_select_same_audio(self):
        prompt = self.prompt("off", ["12", 0])
        prompt["10"] = {"class_type": "LoadAudio", "inputs": {"audio": "wrong.wav"},
                        "_meta": {"title": "Other audio"}}
        prompt["11"] = {"class_type": "LoadAudio", "inputs": {"audio": "selected.wav"},
                        "_meta": {"title": "Voiceover"}}
        prompt["12"] = {"class_type": "H3LongformAudioName",
                        "inputs": {"suffix": ".mp4", "source_title": "Voiceover", "name_suffix": "_fast"}}
        with contextlib.redirect_stdout(io.StringIO()):
            name, filename = h3.H3LongformAudioName().derive(".mp4", "Voiceover", prompt, name_suffix="_fast")
        self.assertEqual((name, filename), ("selected_fast", "selected_fast.mp4"))
        self.assertEqual(h3._session_from_prompt(prompt), name)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_network_volume_render_and_new_process_resume(self):
        prompt = self.prompt("5 frames (~0.2s)")
        audio = {"waveform": torch.zeros((1, 1, 24000 * 7)), "sample_rate": 24000}
        # Stale variables from the previous template must not trigger any backup
        # or require a second volume, including when the node pack is re-imported.
        with patch.dict(os.environ, {"H3_GLOBAL_STORAGE": "1", "H3_GLOBAL_ROOT": "/missing-global-volume"}), contextlib.redirect_stdout(io.StringIO()):
            active = load("h3_network", ROOT / "custom_nodes/comfyui-h3-longform/__init__.py")
            index = 0
            while True:
                chunk, length, keep, count, last, _, _, lead = active.H3LongformSplit().split(
                    audio, index, 5, 5, 5, prompt=prompt, motion_carry="5 frames (~0.2s)")
                images = torch.zeros((length, 16, 16, 3))
                active.H3LongformWrite().write(images, audio, index, count, keep, "test", "final.mp4",
                                             chunk, False, lead, prompt, {"workflow": builder.build()})
                self.assertTrue((Path(self.tmp.name) / "h3_longform/test" / f"chunk_{index:04d}.mp4").exists())
                if last:
                    break
                index += 1
            output = Path(self.tmp.name)
            self.assertTrue((output / "final.mp4").exists())
            self.assertTrue((output / "h3_longform/test/.carry/tail_0000.npz").exists())
            self.assertEqual(json.loads((output / "h3_longform/test/workflow.json").read_text()), builder.build())
            # Re-import the nodes as a replacement Pod would, with the same volume.
            resumed = load("h3_resumed", ROOT / "custom_nodes/comfyui-h3-longform/__init__.py")
            result = resumed.H3LongformSplit().split(audio, 0, 5, 5, 5, prompt=prompt, motion_carry="5 frames (~0.2s)")
            self.assertIsInstance(result[0], ExecutionBlocker)


if __name__ == "__main__":
    unittest.main(verbosity=2)
