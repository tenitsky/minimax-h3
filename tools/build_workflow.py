#!/usr/bin/env python3
"""
Generate the standard, Fast and Fast Draft long-form workflow JSON files.

The graph is written from code rather than exported from the UI so every link, slot
and widget value is declared in one readable place and can be checked before a pod
ever loads it. Re-run after editing:

    python tools/build_workflow.py

Opening the result in ComfyUI and saving it again is fine - the UI adds sizes and
cosmetic properties, but the wiring stays the same.
"""

import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "workflows",
                   "minimax_h3_long_video_workflow.json")

HF = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
FAST_TURBO = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
FAST_OUT = os.path.join(os.path.dirname(OUT), "minimax_h3_fast_workflow.json")
FAST_DRAFT_OUT = os.path.join(os.path.dirname(OUT), "minimax_h3_fast_draft_workflow.json")

PROMPT = """integrated_multimodal_description: [Shot 1] Live-action, a presenter - the person in the reference pictures - speaks directly to the camera. The camera remains stationary on a tripod with a fixed focal length throughout the entire shot. Framing, subject size and camera distance remain identical to the reference pictures in every frame. No zoom, dolly, pan, tilt, reframing or cuts. The presenter (S1) talks in a clear, natural voice, lip movements matching the provided speech audio exactly, with subtle eyebrow and jaw movement, natural blinking and small head nods while talking. Face, hair, clothing, lighting and background match the reference pictures exactly throughout.

overall_soundscape: Quiet indoor room tone under the voice. No music, no sound effects."""

NOTE_INPUTS = """## Inputs

- **Load Portrait** - your presenter, facing the camera, face clearly visible. It anchors every chunk's end and, with motion carry off, its start.
- **Load Voiceover** - your speech track, any length. Mono is fine (it is upmixed to stereo for H3's audio VAE).

Drop files in `ComfyUI/input/`, or upload them through the nodes.

## Running a long render

1. Queue **once** with `chunk_index = 0` and read `total_chunks` in the console (or just skip this).
2. Queue with a **batch count at least that large** - surplus items are blocked and cost milliseconds.
3. The last chunk stitches everything into `output/<audio name>.mp4` and clears the queue.

Interrupted? Set `chunk_index` back to 0 and queue again: finished chunks are skipped."""

NOTE_LICENSE = """## Licence - read before running

MiniMax H3's open weights are **not licensed for use in the United States, the European Union, the United Kingdom or South Korea** (MiniMax H3 Community License, Excluded Territories). Using them there needs a separate licence from MiniMax: https://platform.minimax.io/h3-license

Also required by the licence: comply with its Acceptable Use Policy (no impersonating real people without consent, no content harming minors, no election misinformation...), and display \"MiniMax H3\" in any commercial product built on it.

Full text: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"""

NOTE_GRID = """## Why chunks are not whole seconds

H3 only generates **17k+5 frames** (124, 141, 158 ... 362 - about 5.2s to 15.1s at 24fps). Split plans every chunk on that grid:

- `length` = frames to generate (always on the grid)
- `carry_frames` = opening frames replayed from the previous chunk (0, 5 or 22)
- `keep_frames` = frames written = `length - carry_frames - 1` (non-final chunks)

The pinned portrait at the end is dropped. With carry off, the next chunk opens on that portrait. With carry on, it replays the previous tail first; Write trims this lead-in so no frames play twice.

The final chunk is generated at the next grid length up and trimmed to the end of the track.

`alignment` tells H3 where each reference picture lands, as MiniMax's prompt guide asks for FL2VA - it changes with every chunk, so it is generated rather than typed. Describe the presenter as "the person in the reference pictures", not "Picture 1/2": with motion carry, there is only one picture."""

NOTE_CARRY = """## motion_carry (on Split) - off by default, experimental

**off** - every chunk starts AND ends on the portrait. The pose resets at each join; cutting in pauses may help hide it.

**5 / 22 frames** - after chunk 0, each chunk opens on the previous chunk's last 5 (~0.2s) or 22 (~0.9s) written frames, pinned as a clip to help carry motion across the join. The **last** frame stays pinned to the portrait to limit identity drift. Visual continuity and identity preservation still need GPU testing.

The lead-in frames are generated again (with their original audio) and trimmed by Write, so nothing plays twice and sync is unchanged. They count toward H3's 362-frame ceiling, so chunks get slightly shorter.

Pick it before a render starts and use a fresh session when changing it - it changes where every chunk begins. Queue chunks in order. Compare on a short clip first: if the join shows a colour or sharpness step, go back to off."""

NOTE_SIZE = """## Resolution

H3's native canvas is a **768 px short edge**, max 768 x 1344. Set Width/Height to one of:

| Aspect | Width x Height |
|---|---|
| 1:1 | 768 x 768 |
| 3:4 | 768 x 1024 |
| 9:16 | 768 x 1344 |
| 16:9 | 1344 x 768 |

The portrait is center-cropped to this size before it reaches H3, so the first frame (stretched by the node) and the last frame (cropped by the node) are identical."""

NOTE_SPEED = """## Speed / quality

**Turbo LoRA** on (default): 8 steps. Off: 20 steps without the LoRA - roughly 2.5x slower per chunk.

A 7-minute voiceover is about 30-35 chunks. Test a single chunk first (`batch count = 1`) to check lip sync and identity before committing to the whole render.

If a chunk runs out of VRAM, lower `max_seconds` on Split, or use a smaller canvas."""


class Graph:
    def __init__(self):
        self.nodes, self.links = [], []
        self._link_id = 0

    def node(self, nid, ntype, pos, size, inputs=(), outputs=(), widgets=None,
             title=None, models=None, color=None, cnr="comfy-core"):
        props = {"Node name for S&R": ntype}
        if cnr:
            props["cnr_id"] = cnr
        if models:
            props["models"] = [{"name": n, "url": HF + d + "/" + n, "directory": d}
                               for n, d in models]
        n = {
            "id": nid, "type": ntype, "pos": list(pos), "size": list(size),
            "flags": {}, "order": len(self.nodes), "mode": 0,
            # (name, type, is_widget, optional)
            "inputs": [dict({"localized_name": nm, "name": nm, "type": tp, "link": None},
                            **({"widget": {"name": nm}} if wid else {}),
                            **({"shape": 7} if opt else {}))
                       for nm, tp, wid, opt in inputs],
            "outputs": [{"localized_name": nm, "name": nm, "type": tp, "links": []}
                        for nm, tp in outputs],
            "properties": props,
        }
        if title:
            n["title"] = title
        if widgets is not None:
            n["widgets_values"] = widgets
        if color:
            n["color"], n["bgcolor"] = color
        self.nodes.append(n)
        return nid

    def link(self, src, src_slot, dst, dst_name):
        s = next(n for n in self.nodes if n["id"] == src)
        d = next(n for n in self.nodes if n["id"] == dst)
        out = s["outputs"][src_slot]
        slot = next(i for i, inp in enumerate(d["inputs"]) if inp["name"] == dst_name)
        inp = d["inputs"][slot]
        if inp["link"] is not None:
            raise ValueError(f"{d['type']}.{dst_name} already linked")
        if inp["type"] not in (out["type"], "*") and not (
                inp["type"] == "COMBO" and out["type"] == "COMBO"):
            raise ValueError(f"type mismatch {s['type']}.{out['name']} ({out['type']})"
                             f" -> {d['type']}.{dst_name} ({inp['type']})")
        self._link_id += 1
        lid = self._link_id
        inp["link"] = lid
        out["links"].append(lid)
        self.links.append([lid, src, src_slot, dst, slot, out["type"]])


def md(g, nid, title, text, pos, size):
    g.node(nid, "MarkdownNote", pos, size, widgets=[text], title=title, cnr=None,
           color=("#222", "#000"))


def build():
    g = Graph()
    INPUT = ("#322", "#533")    # red: things you set
    W, I = True, False           # input is a widget / a plain socket

    # ---------------------------------------------------------------- notes
    md(g, 1, "Note: Inputs + how to run", NOTE_INPUTS, (-1500, -40), (460, 460))
    md(g, 2, "Note: LICENCE (read first)", NOTE_LICENSE, (-1500, 440), (460, 360))
    md(g, 3, "Why chunks are not whole seconds", NOTE_GRID, (-500, 840), (520, 480))
    md(g, 4, "Note: Resolution", NOTE_SIZE, (60, 840), (420, 380))
    md(g, 5, "Note: Speed / quality", NOTE_SPEED, (520, 840), (420, 300))
    md(g, 6, "Note: motion_carry", NOTE_CARRY, (980, 840), (460, 420))

    # ---------------------------------------------------------------- inputs
    g.node(10, "LoadImage", (-1000, -40), (400, 460),
           outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
           widgets=["person_portrait.png", "image"],
           title="Load Portrait (identity anchor)", color=INPUT)
    g.node(11, "LoadAudio", (-1000, 460), (400, 140),
           outputs=[("AUDIO", "AUDIO")], widgets=["voiceover.mp3", None, None],
           title="Load Voiceover", color=INPUT)
    g.node(12, "PrimitiveInt", (-1000, 640), (400, 90),
           outputs=[("INT", "INT")], widgets=[0, "increment"],
           title="chunk_index (set to increment)", color=INPUT)
    g.node(13, "PrimitiveInt", (-1000, 770), (190, 90),
           outputs=[("INT", "INT")], widgets=[768, "fixed"], title="Width")
    g.node(14, "PrimitiveInt", (-790, 770), (190, 90),
           outputs=[("INT", "INT")], widgets=[768, "fixed"], title="Height")

    # ---------------------------------------------------------------- longform plan
    g.node(20, "H3LongformSplit", (-500, 440), (400, 300),
           inputs=[("audio", "AUDIO", I, False), ("chunk_index", "INT", W, False)],
           outputs=[("audio_chunk", "AUDIO"), ("length", "INT"), ("keep_frames", "INT"),
                    ("total_chunks", "INT"), ("is_last", "BOOLEAN"),
                    ("alignment", "STRING"), ("guide_audio", "AUDIO"),
                    ("carry_frames", "INT")],
           widgets=[0, 12.0, 5.0, 15.0, "pause", True, "off"],
           title="1. Split Audio Chunk (H3 frame grid)", cnr="comfyui-h3-longform")
    g.node(21, "H3LongformAudioName", (-500, 740), (400, 90),
           outputs=[("name", "STRING"), ("filename", "STRING")],
           widgets=[".mp4", "", ""],
           title="Name run from audio filename (unlink to type your own)",
           cnr="comfyui-h3-longform")

    # ---------------------------------------------------------------- prompt
    g.node(30, "PrimitiveStringMultiline", (-500, -40), (520, 460),
           outputs=[("STRING", "STRING")], widgets=[PROMPT], title="Prompt",
           color=INPUT)
    g.node(31, "StringConcatenate", (60, 540), (400, 200),
           inputs=[("string_a", "STRING", W, False), ("string_b", "STRING", W, False)],
           outputs=[("STRING", "STRING")], widgets=["", "", "\n\n"],
           title="alignment + prompt")

    # ---------------------------------------------------------------- models
    g.node(40, "UNETLoader", (60, -300), (520, 90),
           outputs=[("MODEL", "MODEL")], widgets=[UNET, "default"],
           models=[(UNET, "diffusion_models")])
    g.node(41, "LoraLoaderModelOnly", (60, -180), (520, 90),
           inputs=[("model", "MODEL", I, False)], outputs=[("MODEL", "MODEL")],
           widgets=[TURBO, 1], models=[(TURBO, "loras")])
    g.node(42, "CLIPLoader", (60, -60), (520, 110),
           outputs=[("CLIP", "CLIP")], widgets=[TE, "minimax", "default"],
           models=[(TE, "text_encoders")])
    g.node(43, "VAELoader", (60, 80), (520, 60),
           outputs=[("VAE", "VAE")], widgets=[VIDEO_VAE], title="Video VAE",
           models=[(VIDEO_VAE, "vae")])
    g.node(44, "VAELoader", (60, 170), (520, 60),
           outputs=[("VAE", "VAE")], widgets=[AUDIO_VAE], title="Audio VAE",
           models=[(AUDIO_VAE, "vae")])

    g.node(45, "PrimitiveBoolean", (60, 270), (250, 60),
           outputs=[("BOOLEAN", "BOOLEAN")], widgets=[True],
           title="Turbo LoRA (8 steps)", color=INPUT)
    g.node(46, "PrimitiveInt", (60, 360), (190, 90),
           outputs=[("INT", "INT")], widgets=[20, "fixed"], title="Steps (base)")
    g.node(47, "PrimitiveInt", (270, 360), (190, 90),
           outputs=[("INT", "INT")], widgets=[8, "fixed"], title="Steps (turbo)")
    g.node(48, "ComfySwitchNode", (620, -240), (250, 110),
           inputs=[("on_false", "MODEL", I, False), ("on_true", "MODEL", I, False),
                   ("switch", "BOOLEAN", W, False)],
           outputs=[("output", "MODEL")], widgets=[True],
           title="If/Else Switch (Model)")
    g.node(49, "ComfySwitchNode", (620, 360), (250, 110),
           inputs=[("on_false", "INT", I, False), ("on_true", "INT", I, False),
                   ("switch", "BOOLEAN", W, False)],
           outputs=[("output", "INT")], widgets=[True], title="If/Else Switch (Steps)")

    # ---------------------------------------------------------------- conditioning
    g.node(50, "ImageScale", (620, -60), (300, 170),
           inputs=[("image", "IMAGE", I, False), ("width", "INT", W, False),
                   ("height", "INT", W, False)],
           outputs=[("IMAGE", "IMAGE")], widgets=["lanczos", 768, 768, "center"],
           title="Crop portrait to canvas")
    g.node(53, "H3LongformCarry", (620, 160), (300, 150),
           inputs=[("portrait", "IMAGE", I, False),
                   ("chunk_index", "INT", W, False),
                   ("carry_frames", "INT", W, False)],
           outputs=[("first_frame", "IMAGE"), ("carry_clip", "IMAGE")],
           widgets=[0, 0], title="Opening: portrait or motion carry",
           cnr="comfyui-h3-longform")
    g.node(51, "MiniMaxH3ImageToVideo", (960, -300), (400, 330),
           inputs=[("clip", "CLIP", I, False), ("vae", "VAE", I, False),
                   ("first_frame", "IMAGE", I, True), ("last_frame", "IMAGE", I, True),
                   ("prompt", "STRING", W, False), ("width", "INT", W, False),
                   ("height", "INT", W, False), ("length", "INT", W, False)],
           outputs=[("positive", "CONDITIONING"), ("LATENT", "LATENT")],
           widgets=["", 768, 768, 124])
    g.node(52, "MiniMaxH3AddGuide", (960, 80), (400, 190),
           inputs=[("positive", "CONDITIONING", I, False), ("latent", "LATENT", I, False),
                   ("vae", "VAE", I, True), ("audio_vae", "VAE", I, True),
                   ("image", "IMAGE", I, True), ("audio", "AUDIO", I, True)],
           outputs=[("positive", "CONDITIONING")], widgets=[0],
           title="Pin voiceover + optional motion carry (frame 0)")

    # ---------------------------------------------------------------- sampling
    g.node(60, "BasicGuider", (1400, -300), (240, 50),
           inputs=[("model", "MODEL", I, False),
                   ("conditioning", "CONDITIONING", I, False)],
           outputs=[("GUIDER", "GUIDER")])
    g.node(61, "RandomNoise", (1400, -210), (300, 90),
           outputs=[("NOISE", "NOISE")], widgets=[42, "fixed"])
    g.node(62, "KSamplerSelect", (1400, -90), (300, 60),
           outputs=[("SAMPLER", "SAMPLER")], widgets=["res_multistep"])
    g.node(63, "BasicScheduler", (1400, 0), (300, 110),
           inputs=[("model", "MODEL", I, False), ("steps", "INT", W, False)],
           outputs=[("SIGMAS", "SIGMAS")], widgets=["simple", 8, 1])
    g.node(64, "SamplerCustomAdvanced", (1740, -300), (260, 330),
           inputs=[("noise", "NOISE", I, False), ("guider", "GUIDER", I, False),
                   ("sampler", "SAMPLER", I, False), ("sigmas", "SIGMAS", I, False),
                   ("latent_image", "LATENT", I, False)],
           outputs=[("output", "LATENT"), ("denoised_output", "LATENT")])
    g.node(65, "VAEDecode", (1740, 80), (260, 50),
           inputs=[("samples", "LATENT", I, False), ("vae", "VAE", I, False)],
           outputs=[("IMAGE", "IMAGE")])

    # ---------------------------------------------------------------- output
    g.node(70, "H3LongformWrite", (2040, -300), (400, 310),
           inputs=[("images", "IMAGE", I, False), ("original_audio", "AUDIO", I, False),
                   ("chunk_audio", "AUDIO", I, True), ("chunk_index", "INT", W, False),
                   ("total_chunks", "INT", W, False), ("keep_frames", "INT", W, False),
                   ("session", "STRING", W, False), ("filename", "STRING", W, False),
                   ("trim_start", "INT", W, True)],
           outputs=[("status", "STRING")],
           widgets=[0, 1, 123, "run1", "h3_longform_final.mp4", True, 0],
           title="2. Write + Stitch", cnr="comfyui-h3-longform")

    # ---------------------------------------------------------------- wiring
    L = g.link
    L(11, 0, 20, "audio")
    L(12, 0, 20, "chunk_index")
    L(20, 5, 31, "string_a")          # alignment
    L(30, 0, 31, "string_b")          # your prompt

    L(40, 0, 41, "model")
    L(40, 0, 48, "on_false")
    L(41, 0, 48, "on_true")
    L(45, 0, 48, "switch")
    L(46, 0, 49, "on_false")
    L(47, 0, 49, "on_true")
    L(45, 0, 49, "switch")

    L(10, 0, 50, "image")
    L(13, 0, 50, "width")
    L(14, 0, 50, "height")

    L(42, 0, 51, "clip")
    L(43, 0, 51, "vae")
    L(50, 0, 53, "portrait")
    L(12, 0, 53, "chunk_index")
    L(20, 7, 53, "carry_frames")
    L(53, 0, 51, "first_frame")
    L(50, 0, 51, "last_frame")
    L(31, 0, 51, "prompt")
    L(13, 0, 51, "width")
    L(14, 0, 51, "height")
    L(20, 1, 51, "length")

    L(51, 0, 52, "positive")
    L(51, 1, 52, "latent")
    L(43, 0, 52, "vae")
    L(44, 0, 52, "audio_vae")
    L(53, 1, 52, "image")
    L(20, 6, 52, "audio")

    L(48, 0, 60, "model")
    L(52, 0, 60, "conditioning")
    L(48, 0, 63, "model")
    L(49, 0, 63, "steps")
    L(61, 0, 64, "noise")
    L(60, 0, 64, "guider")
    L(62, 0, 64, "sampler")
    L(63, 0, 64, "sigmas")
    L(51, 1, 64, "latent_image")
    L(64, 0, 65, "samples")
    L(43, 0, 65, "vae")

    L(65, 0, 70, "images")
    L(11, 0, 70, "original_audio")
    L(20, 0, 70, "chunk_audio")
    L(12, 0, 70, "chunk_index")
    L(20, 3, 70, "total_chunks")
    L(20, 2, 70, "keep_frames")
    L(20, 7, 70, "trim_start")
    L(21, 0, 70, "session")
    L(21, 1, 70, "filename")

    return {
        "id": "5b2f7c1e-9a41-4d3e-8f10-3a1c0f9e2b77",
        "revision": 0,
        "last_node_id": max(n["id"] for n in g.nodes),
        "last_link_id": g._link_id,
        "nodes": g.nodes,
        "links": g.links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.6, "offset": [1600, 500]}},
        "version": 0.4,
    }


def build_fast():
    """Fixed four-step FL2VA with the author's 768p LoRA settings.

    Reuse the tested audio/carry/write graph; remove the optional base-model and
    step switches. Advanced nodes are collapsed for a smaller working canvas.
    Fewer visible nodes do not themselves speed up inference.
    """
    wf = build()
    old_nodes = {n["id"]: n for n in wf["nodes"]}
    connections = [(src, slot, dst, old_nodes[dst]["inputs"][port]["name"])
                   for _, src, slot, dst, port, _ in wf["links"]]
    removed = {2, 3, 4, 5, 6, 45, 46, 47, 48, 49}
    g = Graph()
    g.nodes = [n for n in wf["nodes"] if n["id"] not in removed]
    nodes = {n["id"]: n for n in g.nodes}
    nodes[1]["title"] = "MiniMax H3 Fast - start here"
    nodes[1]["widgets_values"] = ["""# MiniMax H3 Fast

Dedicated **4-step 768p Turbo** LoRA. Same portrait + voiceover, automatic chunks,
resume and final stitching as the standard workflow. No extra model selection.

1. Upload **Portrait** and **Voiceover**. Leave chunk_index on **increment**.
2. Test one chunk (batch count 1). Reset chunk_index to 0 to resume.
3. Queue enough items: audio seconds / 12, plus a margin. The final chunk stitches
   `output/<audio name>_fast.mp4` and clears the queue.

Fast sessions have `_fast` in their names, so standard renders are kept separate.
Use a new Name suffix if you change inputs or settings. Width/Height default to
768 x 768; 1344 x 768 or 768 x 1344 use more memory and compute.

Model and sampling nodes below are collapsed: double-click to inspect them.
Four sampling steps instead of eight should reduce sampling time; total speed,
lip sync and identity still need GPU testing. Quality can differ from 8 steps.
Motion carry is optional on Split; leave off for the first test.

Licence and full instructions: https://github.com/tenitsky/minimax-h3
"""]
    nodes[21]["widgets_values"] = [".mp4", "", "_fast"]
    nodes[21]["title"] = "Output name (audio filename + name_suffix)"
    nodes[41]["widgets_values"] = [FAST_TURBO, 1]
    nodes[41]["properties"]["models"] = [
        {"name": FAST_TURBO, "url": HF + "loras/" + FAST_TURBO, "directory": "loras"}]
    nodes[41]["title"] = "Fast: 4-step 768p Turbo LoRA"
    nodes[62]["widgets_values"] = ["euler"]
    nodes[63]["widgets_values"] = ["simple", 4, 1]
    # Steps are now an ordinary widget, not a socket driven by the old switch.
    nodes[63]["inputs"] = [i for i in nodes[63]["inputs"] if i["name"] != "steps"]
    nodes[70]["widgets_values"][3:5] = ["run1_fast", "h3_fast_final.mp4"]
    g.node(75, "MiniMaxH3SigmaShift", (0, 0), (300, 110),
           inputs=[("model", "MODEL", False, False)], outputs=[("MODEL", "MODEL")],
           widgets=[6.0, 3.0], title="Fast sigma shift (video 6 / audio 3)")
    nodes[75] = g.nodes[-1]

    # Rebuild links from their named endpoints to avoid stale slot numbers after
    # removing switch-driven inputs. Both scheduler and guider use shifted model.
    for order, node in enumerate(g.nodes):
        node["order"] = order
        for inp in node["inputs"]:
            inp["link"] = None
        for out in node["outputs"]:
            out["links"] = []
    for src, slot, dst, name in connections:
        if dst in removed or src in removed - {48}:
            continue
        g.link(75 if src == 48 else src, slot, dst, name)
    g.link(41, 0, 75, "model")

    # Daily controls across the top, compact advanced groups underneath.
    layout = {
        1: ((0, 0), (360, 610)), 10: ((400, 0), (330, 360)),
        11: ((400, 390), (330, 140)), 12: ((400, 560), (330, 90)),
        30: ((770, 0), (420, 330)), 20: ((770, 365), (420, 290)),
        13: ((1230, 0), (190, 90)), 14: ((1440, 0), (190, 90)),
        21: ((1230, 130), (400, 130)), 70: ((1230, 300), (400, 310)),
    }
    for nid, (pos, size) in layout.items():
        nodes[nid]["pos"], nodes[nid]["size"] = list(pos), list(size)
    groups = [
        ("Models - loaded automatically", [40, 41, 75, 42, 43, 44], 0, "#355563"),
        ("Portrait + voice guidance", [50, 53, 31, 51, 52], 560, "#4a5568"),
        ("Four-step sampling + decode", [60, 61, 62, 63, 64, 65], 1120, "#4c566a"),
    ]
    wf["groups"] = []
    for gid, (title, ids, x, color) in enumerate(groups, 1):
        for index, nid in enumerate(ids):
            nodes[nid]["pos"] = [x + 25, 790 + index * 55]
            nodes[nid]["flags"] = {"collapsed": True}
        wf["groups"].append({"id": gid, "title": title,
                             "bounding": [x, 730, 530, 420],
                             "color": color, "font_size": 22, "flags": {}})
    wf.update(id="a4fb6153-c9bf-4ba8-9b8a-e495d1b00804", nodes=g.nodes, links=g.links,
              last_node_id=75, last_link_id=g._link_id)
    wf["extra"] = {"ds": {"scale": 0.7, "offset": [40, 50]}}
    return wf


def build_fast_draft():
    """Reduce spatial work using the 544p LoRA's supported four-step mode.

    The non-768p v1.0 LoRA supports both eight and four inference steps with
    video/audio shifts 12/3. Keep the existing chunk duration: shorter chunks
    would add more prompt encoding, boundary frames and padding per voiceover.
    """
    wf = build_fast()
    nodes = {n["id"]: n for n in wf["nodes"]}
    wf["id"] = "9b2f3be5-2a41-4416-bb4b-161d7cbef581"
    nodes[1]["title"] = "MiniMax H3 Fast Draft - lower detail, less GPU work"
    nodes[1]["widgets_values"] = ["""# MiniMax H3 Fast Draft

**544 x 544, four steps.** About half the pixels per frame of Fast 768 x 768.
Use this for speed-focused drafts; output has less detail and is not upscaled.

1. Upload **Portrait** and **Voiceover**. Leave chunk_index on **increment**.
2. Test one chunk (batch count 1). Reset chunk_index to 0 to resume.
3. Queue enough items: audio seconds / 12, plus a margin. The final chunk stitches
   `output/<audio name>_fast_draft.mp4` and clears the queue.

Reuses the installed **FL2V Turbo 8-step v1.0** LoRA in its author-supported
**four-step** mode: Euler, simple scheduler, video/audio shifts 12/3. The filename
says 8step; four-step inference is intentional for this lower-resolution draft.

Portrait anchors, the full voiceover, resume and stitching stay enabled. Motion
carry defaults to off. Chunks still run up to 15s to avoid extra boundary overhead.

Change `name_suffix` to start a new take when changing inputs or settings. Draft
sessions end in `_fast_draft`, separate from Fast and standard renders.

Less model work does not guarantee a particular speedup. Loading, encoding and
CPU offloading can still dominate. Speed, lip sync and identity need GPU testing.
Use the standard or 768p Fast workflow when you need more detail.

Licence and full instructions: https://github.com/tenitsky/minimax-h3
"""]
    for nid in (13, 14):
        nodes[nid]["widgets_values"] = [544, "fixed"]
    nodes[50]["widgets_values"][1:3] = [544, 544]
    nodes[51]["widgets_values"][1:3] = [544, 544]
    nodes[41]["widgets_values"] = [TURBO, 1]
    nodes[41]["properties"]["models"] = [
        {"name": TURBO, "url": HF + "loras/" + TURBO, "directory": "loras"}]
    nodes[41]["title"] = "544p Turbo v1.0 (four-step mode)"
    nodes[75]["widgets_values"] = [12.0, 3.0]
    nodes[75]["title"] = "Draft sigma shift (video 12 / audio 3)"
    nodes[21]["widgets_values"] = [".mp4", "", "_fast_draft"]
    nodes[70]["widgets_values"][3:5] = ["run1_fast_draft", "h3_fast_draft_final.mp4"]
    return wf


if __name__ == "__main__":
    for path, wf in ((OUT, build()), (FAST_OUT, build_fast()),
                     (FAST_DRAFT_OUT, build_fast_draft())):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(wf, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {os.path.normpath(path)}: {len(wf['nodes'])} nodes, "
              f"{len(wf['links'])} links")
