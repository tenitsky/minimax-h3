#!/usr/bin/env python3
"""
Generate workflows/minimax_h3_long_video_workflow.json.

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

PROMPT = """integrated_multimodal_description: [Shot 1] Live-action, a presenter - the person in the reference pictures - speaks directly to the camera. The camera stays locked off on a tripod. The presenter (S1) talks in a clear, natural voice, lip movements matching the provided speech audio exactly, with subtle eyebrow and jaw movement, natural blinking and small head nods while talking. Toward the end of the shot the presenter settles into the pose, framing and expression of the last reference picture. Face, hair, clothing, lighting and background match the reference pictures exactly throughout.

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
           widgets=[".mp4", ""],
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


if __name__ == "__main__":
    wf = build()
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(wf, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {os.path.normpath(OUT)}: {len(wf['nodes'])} nodes, "
          f"{len(wf['links'])} links")
