#!/usr/bin/env python3
"""
Generate the standard, Fast and Fast Draft long-form workflow JSON files.

The graph is written from code rather than exported from the UI so every link, slot
and widget value is declared in one readable place and can be checked before a pod
ever loads it. Re-run after editing:

    python tools/build_workflow.py

Opening the result in ComfyUI and saving it again is fine - the UI adds sizes and
cosmetic properties, but the wiring stays the same.

All three variants share one graph. They differ only in the settings in VARIANTS,
each taken from the Turbo LoRA authors' table (github.com/ModelTC/Minimax-H3-Turbo):
a distilled LoRA only behaves at the resolution and sigma shift it was trained at.
"""

import json
import os

WORKFLOWS = os.path.join(os.path.dirname(__file__), "..", "workflows")

HF = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
TURBO_HF = "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/"
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_544 = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
TURBO_768_8 = "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
TURBO_768_4 = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"

# Talking-Head Prompt defaults, written for the example portrait. Each starts with
# its field name because a multiline widget hides its label once it holds text;
# the node strips the tag before building the prompt.
PROMPT_FIELDS = [
    "subject: a woman with long, dark brown wavy hair falling past her shoulders, "
    "wearing a dark navy V-neck blouse with a soft satin sheen, framed from the chest "
    "up and facing the camera",
    "background: a plain, seamless light-grey studio backdrop with soft, even studio "
    "lighting and a gentle shadow-free falloff, with nothing else in the frame",
    "delivery: a warm, friendly, confident voice at a relaxed pace",
    "extra: She keeps a gentle, natural smile between sentences, and her hair and "
    "blouse stay neatly in place.",
]
# Split: chunk_index, target_seconds, min_seconds, max_seconds, cut_mode,
# skip_existing, motion_carry, auto_chunk. auto_chunk picks the first unfinished
# chunk from the files on disk, so there is no counter to reset between videos.
# Every frame between a chunk's two pinned portraits is
# invented, so drift grows with chunk length: 10s rather than H3's 15s ceiling.
SPLIT_WIDGETS = [0, 8.0, 5.0, 10.0, "pause", True, "off", True]
# Model Sparse Attention (Sol-Attn): method, tau, start_percent, end_percent,
# dense_blocks, min_tokens, extra_tokens, sink_conditioning, verbose.
# exact_kv_and_rows keeps the text, pinned portraits and voiceover rows exact for
# every query, and the generated audio rows dense.
SPARSE_WIDGETS = ["sol-attn", 1.3, 0.2, 1.0, "", 12288, 256, "exact_kv_and_rows", False]
BYPASS = 4  # LiteGraph node mode: pass the input straight through

NOTE_LICENSE = """## Licence - read before running

MiniMax H3's open weights are **not licensed for use in the United States, the European Union, the United Kingdom or South Korea** (MiniMax H3 Community License, Excluded Territories). Using them there needs a separate licence from MiniMax: https://platform.minimax.io/h3-license

Also required by the licence: comply with its Acceptable Use Policy (no impersonating real people without consent, no content harming minors, no election misinformation...), and display \"MiniMax H3\" in any commercial product built on it.

Full text: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"""

NOTE_CONSISTENCY = """## Keeping the background and the frame clean

H3 pins your portrait only at the **first and last frame** of each chunk. Every frame in between is generated, and the model fills whatever the prompt leaves open - a changing backdrop, a lower-third, a channel logo.

This workflow closes those gaps:

- **Describe the portrait** in the Talking-Head Prompt: `subject` (who, what they wear) and `background` (what is behind them, concretely). Naming the background is what holds it in place.
- The prompt uses MiniMax's own format exactly: the alignment line, then the three fields `integrated_multimodal_description`, `overall_soundscape` and `non_diegetic_music`. It states a locked-off camera, one person, and no text, logos or graphics.
- Chunks are capped at **10 s** (H3 can do 15). Less time between anchors means less invention.
- Each Turbo LoRA runs at the resolution and sigma shift it was distilled at.

**Still seeing changes?** Make `background` more specific (colours, objects, lighting), lower `max_seconds` on Split to 8, or render the final pass with Turbo off.

Use a portrait with a simple, evenly lit background, and nothing text-like in it: signs, screens and logos in the portrait tend to animate."""

NOTE_GRID = """## Why chunks are not whole seconds

H3 only generates **17k+5 frames** (124, 141, 158 ... 362 - about 5.2s to 15.1s at 24fps). Split plans every chunk on that grid:

- `length` = frames to generate (always on the grid)
- `carry_frames` = opening frames replayed from the previous chunk (0, 5 or 22)
- `keep_frames` = frames written = `length - carry_frames - 1` (non-final chunks)

The pinned portrait at the end is dropped. With carry off, the next chunk opens on that portrait. With carry on, it replays the previous tail first; Write trims this lead-in so no frames play twice.

The final chunk is generated at the next grid length up and trimmed to the end of the track.

The prompt's first line tells H3 where each portrait lands, as MiniMax's prompt guide requires. It changes with every chunk, so the Talking-Head Prompt node writes it."""

NOTE_CARRY = """## motion_carry (on Split) - off by default, experimental

**off** - every chunk starts AND ends on the portrait. The pose resets at each join; cutting in pauses may help hide it.

**5 / 22 frames** - after chunk 0, each chunk opens on the previous chunk's last 5 (~0.2s) or 22 (~0.9s) written frames, pinned as a clip to help carry motion across the join. The **last** frame stays pinned to the portrait to limit identity drift. Visual continuity and identity preservation still need GPU testing.

The lead-in frames are generated again (with their original audio) and trimmed by Write, so nothing plays twice and sync is unchanged. They count toward the chunk's length, so chunks get slightly shorter.

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

**Turbo on** (default): the 8-step **768p** Turbo LoRA with its trained video/audio shift of 6/3. **Off**: 20 steps of the base model at its default shift 12/3 - roughly 2.5x slower per chunk, the best quality for a final pass.

**Attention backend**: Comfy Kitchen INT8 attention, ComfyUI's built-in quantized attention (the same idea as SageAttention). Set it to `pytorch attention` to compare quality.

**Model Sparse Attention** (Sol-Attn) is **bypassed** here to keep this the quality reference. Select it and press Ctrl+B to enable it. It skips low-weight attention blocks on long sequences, while attention to the portraits, text and voiceover stays exact. The Fast workflows have it on.

Test a single chunk first (`batch count = 1`). If a chunk runs out of VRAM, lower `max_seconds` on Split or use a smaller canvas."""

NOTE_INPUTS = """## Inputs

- **Load Portrait** - one person facing the camera, face clearly visible, in front of the background you want to keep. It is pinned as the first and last frame of every chunk.
- **Load Voiceover** - your speech track, any length. Mono is fine.
- **Talking-Head Prompt** - describe the `subject` and the `background` you can see in the portrait, in a few concrete words. The node writes the full MiniMax-format prompt for every chunk. The fields come filled in for an example portrait; rewrite them for yours and keep the leading `subject:` / `background:` tags (optional, stripped).

Drop files in `ComfyUI/input/`, or upload them through the nodes.

## Running a long render

1. Test one chunk (batch count 1) and check `output/h3_longform/<audio name>/chunk_0000.mp4`.
2. Queue again with a **batch count of at least audio seconds / 8**, plus a margin. Each item renders the next unfinished chunk; surplus items are skipped in milliseconds.
3. The last chunk stitches everything into `output/<audio name>.mp4` and clears the queue.

**New video:** just load the new voiceover and queue - it starts at chunk 1 in its own folder. Interrupted? Queue again: finished chunks are skipped. To redo a finished video, change `name_suffix` or delete its folder."""

NOTE_FAST = """# MiniMax H3 Fast

Dedicated **4-step 768p Turbo** LoRA (video/audio shift 6/3), INT8 attention and Sol-Attn sparse attention. Same portrait + voiceover, automatic chunks, resume and stitching as the standard workflow.

1. Upload **Portrait** and **Voiceover**.
2. In **Talking-Head Prompt**, describe the `subject` and `background` you see in the portrait. This is what keeps the background from changing. The fields come filled in for an example portrait; rewrite them for yours and keep the leading `subject:` / `background:` tags (optional, stripped).
3. Test one chunk (batch count 1), then queue audio seconds / 8, plus a margin. Each item renders the next unfinished chunk; the last one stitches `output/<audio name>_fast.mp4`. A new voiceover starts at chunk 1 in its own folder - nothing to reset.

Change `name_suffix` on the output-name node to start a fresh take when changing inputs or settings.

Model and sampling nodes below are collapsed: double-click to inspect them. To compare quality, bypass **Model Sparse Attention** (Ctrl+B) or set the attention backend to `pytorch attention`.

Licence and full instructions: https://github.com/tenitsky/minimax-h3
"""

NOTE_DRAFT = """# MiniMax H3 Fast Draft

**544 x 544, four steps** - the 544p-trained Turbo LoRA in its supported four-step mode (video/audio shift 12/3), INT8 attention and Sol-Attn sparse attention. About half the pixels per frame of the 768 workflows: use it to check timing and framing, then render the final with Fast or the standard workflow.

1. Upload **Portrait** and **Voiceover**.
2. In **Talking-Head Prompt**, describe the `subject` and `background` you see in the portrait. The fields come filled in for an example portrait; rewrite them for yours and keep the leading `subject:` / `background:` tags (optional, stripped).
3. Test one chunk (batch count 1), then queue audio seconds / 8, plus a margin. Each item renders the next unfinished chunk; the last one stitches `output/<audio name>_draft.mp4`. A new voiceover starts at chunk 1 in its own folder - nothing to reset.

The LoRA filename says 8step; four-step inference is intentional and supported by its authors.

Model and sampling nodes below are collapsed: double-click to inspect them.

Licence and full instructions: https://github.com/tenitsky/minimax-h3
"""

VARIANTS = {
    "standard": dict(
        path="minimax_h3_talking_head.json", id="0d8c2f52-7f7e-4c1b-9d6c-5f2e91a4b3c0",
        canvas=768, lora=TURBO_768_8, lora_url=TURBO_HF, shift=(6.0, 3.0), steps=8,
        switch=True, sparse=False, suffix="",
        fallback=["run1", "h3_talking_head.mp4"]),
    "fast": dict(
        path="minimax_h3_talking_head_fast.json", id="6e1a9b37-2c55-4f0e-8a31-b7d4c9e0f215",
        canvas=768, lora=TURBO_768_4, lora_url=HF + "loras/", shift=(6.0, 3.0), steps=4,
        switch=False, sparse=True, suffix="_fast",
        fallback=["run1_fast", "h3_talking_head_fast.mp4"], note=NOTE_FAST,
        title="MiniMax H3 Fast - start here"),
    "draft": dict(
        path="minimax_h3_talking_head_draft.json", id="b3f7d0c4-81a9-4e62-9c5d-2a6e7f1b8d93",
        canvas=544, lora=TURBO_544, lora_url=HF + "loras/", shift=(12.0, 3.0), steps=4,
        switch=False, sparse=True, suffix="_draft",
        fallback=["run1_draft", "h3_talking_head_draft.mp4"], note=NOTE_DRAFT,
        title="MiniMax H3 Fast Draft - lower detail, less GPU work"),
}


class Graph:
    def __init__(self):
        self.nodes, self.links = [], []
        self._link_id = 0

    def node(self, nid, ntype, pos, size, inputs=(), outputs=(), widgets=None,
             title=None, models=None, color=None, cnr="comfy-core", mode=0):
        props = {"Node name for S&R": ntype}
        if cnr:
            props["cnr_id"] = cnr
        if models:
            # (filename, models/ subfolder, download URL)
            props["models"] = [{"name": n, "url": url, "directory": d}
                               for n, d, url in models]
        n = {
            "id": nid, "type": ntype, "pos": list(pos), "size": list(size),
            "flags": {}, "order": len(self.nodes), "mode": mode,
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


def build(variant="standard"):
    v = VARIANTS[variant]
    g = Graph()
    INPUT = ("#322", "#533")    # red: things you set
    W, I = True, False           # input is a widget / a plain socket
    side = v["canvas"]

    # ---------------------------------------------------------------- notes
    if v["switch"]:
        md(g, 1, "Note: Inputs + how to run", NOTE_INPUTS, (-1500, -40), (460, 520))
        md(g, 2, "Note: LICENCE (read first)", NOTE_LICENSE, (-1500, 500), (460, 360))
        md(g, 3, "Why chunks are not whole seconds", NOTE_GRID, (-500, 900), (520, 480))
        md(g, 4, "Note: Resolution", NOTE_SIZE, (60, 900), (420, 380))
        md(g, 5, "Note: Speed / quality", NOTE_SPEED, (520, 900), (440, 440))
        md(g, 6, "Note: motion_carry", NOTE_CARRY, (1000, 900), (460, 420))
    else:
        md(g, 1, v["title"], v["note"], (0, 0), (360, 640))
    md(g, 7, "Note: Keeping the background and the frame clean", NOTE_CONSISTENCY,
       (1500, 900) if v["switch"] else (1680, 0), (460, 560))

    # ---------------------------------------------------------------- inputs
    g.node(10, "LoadImage", (-1000, -40), (400, 460),
           outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
           widgets=["person_portrait.png", "image"],
           title="Load Portrait (identity anchor)", color=INPUT)
    g.node(11, "LoadAudio", (-1000, 460), (400, 140),
           outputs=[("AUDIO", "AUDIO")], widgets=["voiceover.mp3", None, None],
           title="Load Voiceover", color=INPUT)
    g.node(13, "PrimitiveInt", (-1000, 770), (190, 90),
           outputs=[("INT", "INT")], widgets=[side, "fixed"], title="Width")
    g.node(14, "PrimitiveInt", (-790, 770), (190, 90),
           outputs=[("INT", "INT")], widgets=[side, "fixed"], title="Height")

    # ---------------------------------------------------------------- longform plan
    g.node(20, "H3LongformSplit", (-500, 500), (400, 300),
           inputs=[("audio", "AUDIO", I, False)],
           outputs=[("audio_chunk", "AUDIO"), ("length", "INT"), ("keep_frames", "INT"),
                    ("total_chunks", "INT"), ("is_last", "BOOLEAN"),
                    ("alignment", "STRING"), ("guide_audio", "AUDIO"),
                    ("carry_frames", "INT"), ("chunk_index", "INT")],
           widgets=list(SPLIT_WIDGETS),
           title="1. Split Audio Chunk (H3 frame grid)", cnr="comfyui-h3-longform")
    g.node(21, "H3LongformAudioName", (-500, 810), (400, 110),
           outputs=[("name", "STRING"), ("filename", "STRING")],
           widgets=[".mp4", "", v["suffix"]],
           title="Output name (audio filename + name_suffix)",
           cnr="comfyui-h3-longform")

    # ---------------------------------------------------------------- prompt
    g.node(30, "H3LongformPrompt", (-500, -40), (520, 500),
           inputs=[("length", "INT", W, False), ("carry_frames", "INT", W, False)],
           outputs=[("prompt", "STRING")],
           widgets=[124, 0] + PROMPT_FIELDS,
           title="Talking-Head Prompt (describe subject + background)", color=INPUT,
           cnr="comfyui-h3-longform")

    # ---------------------------------------------------------------- models
    lora_title = {"standard": "Turbo LoRA: 8-step 768p",
                  "fast": "Fast: 4-step 768p Turbo LoRA",
                  "draft": "544p Turbo v1.0 (four-step mode)"}[variant]
    g.node(40, "UNETLoader", (60, -540), (520, 90),
           outputs=[("MODEL", "MODEL")], widgets=[UNET, "default"],
           models=[(UNET, "diffusion_models", HF + "diffusion_models/" + UNET)])
    g.node(41, "LoraLoaderModelOnly", (60, -420), (520, 90),
           inputs=[("model", "MODEL", I, False)], outputs=[("MODEL", "MODEL")],
           widgets=[v["lora"], 1], title=lora_title,
           models=[(v["lora"], "loras", v["lora_url"] + v["lora"])])
    shift_video, shift_audio = v["shift"]
    g.node(75, "MiniMaxH3SigmaShift", (60, -300), (520, 110),
           inputs=[("model", "MODEL", I, False)], outputs=[("MODEL", "MODEL")],
           widgets=[shift_video, shift_audio],
           title=f"Turbo sigma shift (video {shift_video:g} / audio {shift_audio:g})")
    g.node(42, "CLIPLoader", (60, -160), (520, 110),
           outputs=[("CLIP", "CLIP")], widgets=[TE, "minimax", "default"],
           models=[(TE, "text_encoders", HF + "text_encoders/" + TE)])
    g.node(43, "VAELoader", (60, -20), (520, 60),
           outputs=[("VAE", "VAE")], widgets=[VIDEO_VAE], title="Video VAE",
           models=[(VIDEO_VAE, "vae", HF + "vae/" + VIDEO_VAE)])
    g.node(44, "VAELoader", (60, 70), (520, 60),
           outputs=[("VAE", "VAE")], widgets=[AUDIO_VAE], title="Audio VAE",
           models=[(AUDIO_VAE, "vae", HF + "vae/" + AUDIO_VAE)])

    if v["switch"]:
        g.node(45, "PrimitiveBoolean", (60, 170), (250, 60),
               outputs=[("BOOLEAN", "BOOLEAN")], widgets=[True],
               title="Turbo LoRA (8 steps)", color=INPUT)
        g.node(46, "PrimitiveInt", (60, 260), (190, 90),
               outputs=[("INT", "INT")], widgets=[20, "fixed"], title="Steps (base)")
        g.node(47, "PrimitiveInt", (270, 260), (190, 90),
               outputs=[("INT", "INT")], widgets=[v["steps"], "fixed"],
               title="Steps (turbo)")
        g.node(48, "ComfySwitchNode", (620, -540), (250, 110),
               inputs=[("on_false", "MODEL", I, False), ("on_true", "MODEL", I, False),
                       ("switch", "BOOLEAN", W, False)],
               outputs=[("output", "MODEL")], widgets=[True],
               title="If/Else Switch (Model)")
        g.node(49, "ComfySwitchNode", (620, 260), (250, 110),
               inputs=[("on_false", "INT", I, False), ("on_true", "INT", I, False),
                       ("switch", "BOOLEAN", W, False)],
               outputs=[("output", "INT")], widgets=[True], title="If/Else Switch (Steps)")

    # ---------------------------------------------------------------- acceleration
    g.node(76, "ModelAttentionBackend", (1400, -660), (300, 60),
           inputs=[("model", "MODEL", I, False)], outputs=[("model", "MODEL")],
           widgets=["comfy kitchen attention"],
           title="Attention backend (INT8, like SageAttention)")
    g.node(77, "BlockSparseAttention", (1400, -560), (300, 230),
           inputs=[("model", "MODEL", I, False)], outputs=[("model", "MODEL")],
           widgets=list(SPARSE_WIDGETS),
           title="Model Sparse Attention (Sol-Attn)" + ("" if v["sparse"] else
                                                          " - Ctrl+B to enable"),
           mode=0 if v["sparse"] else BYPASS)

    # ---------------------------------------------------------------- conditioning
    g.node(50, "ImageScale", (620, -380), (300, 170),
           inputs=[("image", "IMAGE", I, False), ("width", "INT", W, False),
                   ("height", "INT", W, False)],
           outputs=[("IMAGE", "IMAGE")], widgets=["lanczos", side, side, "center"],
           title="Crop portrait to canvas")
    g.node(53, "H3LongformCarry", (620, -170), (300, 150),
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
           widgets=["", side, side, 124])
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
    # The Turbo LoRAs are distilled on Euler flow steps (authors' ComfyUI guide).
    g.node(62, "KSamplerSelect", (1400, -90), (300, 60),
           outputs=[("SAMPLER", "SAMPLER")], widgets=["euler"])
    g.node(63, "BasicScheduler", (1400, 0), (300, 110),
           inputs=[("model", "MODEL", I, False)]
           + ([("steps", "INT", W, False)] if v["switch"] else []),
           outputs=[("SIGMAS", "SIGMAS")], widgets=["simple", v["steps"], 1])
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
           widgets=[0, 1, 123] + v["fallback"] + [True, 0],
           title="2. Write + Stitch", cnr="comfyui-h3-longform")

    # ---------------------------------------------------------------- wiring
    L = g.link
    L(11, 0, 20, "audio")
    L(20, 1, 30, "length")
    L(20, 7, 30, "carry_frames")

    # Turbo branch: LoRA -> its trained shift. Base branch keeps the model's own
    # 12/3 default. Both then share the attention patches.
    L(40, 0, 41, "model")
    L(41, 0, 75, "model")
    if v["switch"]:
        L(40, 0, 48, "on_false")
        L(75, 0, 48, "on_true")
        L(45, 0, 48, "switch")
        L(46, 0, 49, "on_false")
        L(47, 0, 49, "on_true")
        L(45, 0, 49, "switch")
        L(48, 0, 76, "model")
    else:
        L(75, 0, 76, "model")
    L(76, 0, 77, "model")

    L(10, 0, 50, "image")
    L(13, 0, 50, "width")
    L(14, 0, 50, "height")

    L(42, 0, 51, "clip")
    L(43, 0, 51, "vae")
    L(50, 0, 53, "portrait")
    L(20, 8, 53, "chunk_index")
    L(20, 7, 53, "carry_frames")
    L(53, 0, 51, "first_frame")
    L(50, 0, 51, "last_frame")
    L(30, 0, 51, "prompt")
    L(13, 0, 51, "width")
    L(14, 0, 51, "height")
    L(20, 1, 51, "length")

    L(51, 0, 52, "positive")
    L(51, 1, 52, "latent")
    L(43, 0, 52, "vae")
    L(44, 0, 52, "audio_vae")
    L(53, 1, 52, "image")
    L(20, 6, 52, "audio")

    L(77, 0, 60, "model")
    L(52, 0, 60, "conditioning")
    L(77, 0, 63, "model")
    if v["switch"]:
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
    L(20, 8, 70, "chunk_index")
    L(20, 3, 70, "total_chunks")
    L(20, 2, 70, "keep_frames")
    L(20, 7, 70, "trim_start")
    L(21, 0, 70, "session")
    L(21, 1, 70, "filename")

    wf = {
        "id": v["id"],
        "revision": 0,
        "last_node_id": max(n["id"] for n in g.nodes),
        "last_link_id": g._link_id,
        "nodes": g.nodes,
        "links": g.links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.6, "offset": [1600, 700]}},
        "version": 0.4,
    }
    if not v["switch"]:
        compact(wf)
    return wf


def compact(wf):
    """Daily controls across the top, advanced nodes collapsed in groups below.

    Fewer boxes on screen do not reduce model work; this is only for readability.
    """
    nodes = {n["id"]: n for n in wf["nodes"]}
    layout = {
        1: ((0, 0), (360, 640)), 10: ((400, 0), (330, 360)),
        11: ((400, 390), (330, 140)),
        30: ((770, 0), (420, 360)), 20: ((770, 395), (420, 290)),
        13: ((1230, 0), (190, 90)), 14: ((1440, 0), (190, 90)),
        21: ((1230, 130), (400, 130)), 70: ((1230, 300), (400, 340)),
    }
    for nid, (pos, size) in layout.items():
        nodes[nid]["pos"], nodes[nid]["size"] = list(pos), list(size)
    groups = [
        ("Models + acceleration - loaded automatically",
         [40, 41, 75, 76, 77, 42, 43, 44], 0, "#355563"),
        ("Portrait + voice guidance", [50, 53, 51, 52], 560, "#4a5568"),
        ("Sampling + decode", [60, 61, 62, 63, 64, 65], 1120, "#4c566a"),
    ]
    for gid, (title, ids, x, color) in enumerate(groups, 1):
        for index, nid in enumerate(ids):
            nodes[nid]["pos"] = [x + 25, 790 + index * 55]
            nodes[nid]["flags"] = {"collapsed": True}
        wf["groups"].append({"id": gid, "title": title,
                             "bounding": [x, 730, 530, 520],
                             "color": color, "font_size": 22, "flags": {}})
    wf["extra"] = {"ds": {"scale": 0.7, "offset": [40, 50]}}


def build_fast():
    return build("fast")


def build_fast_draft():
    return build("draft")


if __name__ == "__main__":
    for name in VARIANTS:
        wf = build(name)
        path = os.path.join(WORKFLOWS, VARIANTS[name]["path"])
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(wf, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {os.path.normpath(path)}: {len(wf['nodes'])} nodes, "
              f"{len(wf['links'])} links")
