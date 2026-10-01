#!/usr/bin/env python3
"""
Generate the six talking-head workflow JSON files.

The graph is written from code rather than exported from the UI so every link, slot
and widget value is declared in one readable place and can be checked before a pod
ever loads it. Re-run after editing:

    python tools/build_workflow.py

Opening the result in ComfyUI and saving it again is fine - the UI adds sizes and
cosmetic properties, but the wiring stays the same.

Every workflow shares one graph. Standard, Fast and Draft differ only in the
settings in VARIANTS, each taken from the Turbo LoRA authors' table
(github.com/ModelTC/Minimax-H3-Turbo): a distilled LoRA only behaves at the
resolution and sigma shift it was trained at. Each comes in a single-voiceover
version and a batch version that renders every audio file in a folder.

Layout: seven numbered stages from left to right, each a coloured group with a short
note underneath, and a "Start here" note on the far left. Nothing is collapsed.
"""

import json
import os

WORKFLOW_DIR = os.path.join(os.path.dirname(__file__), "..", "workflows")

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
# Every frame between a chunk's two pinned portraits is invented, so drift grows
# with chunk length: 10s rather than H3's 15s ceiling.
SPLIT_WIDGETS = [0, 8.0, 5.0, 10.0, "pause", True, "off", True]
# Model Sparse Attention (Sol-Attn): method, tau, start_percent, end_percent,
# dense_blocks, min_tokens, extra_tokens, sink_conditioning, verbose.
# exact_kv_and_rows keeps the text, pinned portraits and voiceover rows exact for
# every query, and the generated audio rows dense.
SPARSE_WIDGETS = ["sol-attn", 1.3, 0.2, 1.0, "", 12288, 256, "exact_kv_and_rows", False]
BYPASS = 4  # LiteGraph node mode: pass the input straight through

VARIANTS = {
    "standard": dict(
        title="MiniMax H3 Talking Head",
        summary="768 x 768, 8 steps with the 768p Turbo LoRA. Switch Turbo off for 20 "
                "steps of the base model: the best quality, about 2.5x slower.",
        canvas=768, lora=TURBO_768_8, lora_url=TURBO_HF, shift=(6.0, 3.0), steps=8,
        switch=True, sparse=False, suffix="",
        lora_title="Turbo LoRA: 8-step 768p"),
    "fast": dict(
        title="MiniMax H3 Talking Head - Fast",
        summary="768 x 768, 4 steps with the dedicated 4-step 768p Turbo LoRA and "
                "sparse attention. The everyday preset.",
        canvas=768, lora=TURBO_768_4, lora_url=HF + "loras/", shift=(6.0, 3.0), steps=4,
        switch=False, sparse=True, suffix="_fast",
        lora_title="Turbo LoRA: 4-step 768p"),
    "draft": dict(
        title="MiniMax H3 Talking Head - Draft",
        summary="544 x 544, 4 steps with the 544p Turbo LoRA and sparse attention: about "
                "half the pixels of the 768 workflows. Check timing and framing here, "
                "then render the final with Fast or the standard workflow. (The LoRA "
                "file says 8step; its authors support four-step use.)",
        canvas=544, lora=TURBO_544, lora_url=HF + "loras/", shift=(12.0, 3.0), steps=4,
        switch=False, sparse=True, suffix="_draft",
        lora_title="Turbo LoRA: 544p, four-step mode"),
}

WORKFLOWS = [
    dict(variant="standard", batch=False, path="minimax_h3_talking_head.json",
         id="0d8c2f52-7f7e-4c1b-9d6c-5f2e91a4b3c0"),
    dict(variant="fast", batch=False, path="minimax_h3_talking_head_fast.json",
         id="6e1a9b37-2c55-4f0e-8a31-b7d4c9e0f215"),
    dict(variant="draft", batch=False, path="minimax_h3_talking_head_draft.json",
         id="b3f7d0c4-81a9-4e62-9c5d-2a6e7f1b8d93"),
    dict(variant="standard", batch=True, path="minimax_h3_talking_head_batch.json",
         id="4a7e2d19-6b3c-4f85-9e10-c2d8a5f7b641"),
    dict(variant="fast", batch=True, path="minimax_h3_talking_head_fast_batch.json",
         id="9c1f5e83-2d47-4a6b-8f3e-71b0d4c6a925"),
    dict(variant="draft", batch=True, path="minimax_h3_talking_head_draft_batch.json",
         id="e5b28c71-4f9a-4d3e-a6c2-8b17f0d3e549"),
]

# ------------------------------------------------------------------------ notes
NOTE_LICENSE = """## Licence - read before running

MiniMax H3's open weights are **not licensed for use in the United States, the European Union, the United Kingdom or South Korea** (MiniMax H3 Community License, Excluded Territories). Using them there needs a separate licence from MiniMax: https://platform.minimax.io/h3-license

Also required by the licence: comply with its Acceptable Use Policy (no impersonating real people without consent, no content harming minors, no election misinformation...), and display \"MiniMax H3\" in any commercial product built on it.

Full text: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"""

NOTE_INPUTS = """## 1. Inputs

**Portrait:** one person facing the camera, face clearly visible, in front of the background you want to keep. It is pinned as the first and last frame of every chunk. Any size; it is center-cropped to the canvas.

**Canvas** (Width x Height). H3's native canvas has a **768 px short edge**, max 768 x 1344:

| Aspect | Width x Height |
|---|---|
| 1:1 | 768 x 768 |
| 3:4 | 768 x 1024 |
| 9:16 | 768 x 1344 |
| 16:9 | 1344 x 768 |

Draft uses 544 x 544. Larger canvases need more memory and time."""

NOTE_PROMPT = """## 2. Describe the shot - this keeps the background steady

H3 pins your portrait only at the **first and last frame** of each chunk. Every frame in between is generated, and the model fills whatever the prompt leaves open: a changing backdrop, a lower-third, a channel logo.

- **subject:** who is on screen and what they wear, as seen in the portrait.
- **background:** what is behind them, concretely (colour, objects, lighting). Naming it is what holds it in place.
- **delivery:** how they speak.
- **extra** (optional): one small, calm action. Keep motion small.

The fields come filled in for an example portrait: rewrite them for yours. The `subject:` style tags are optional and are removed automatically.

The node writes MiniMax's exact prompt format for every chunk: a locked-off camera, one person, and no text, logos or graphics.

**Still seeing changes?** Be more specific about the background, lower `max_seconds` on Split to 8, and use a portrait with a plain, evenly lit background."""

NOTE_PLAN = """## 3. Chunk plan

H3 renders up to ~15 s at once, so the voiceover is cut into **chunks of up to 10 s**, preferably in pauses. Shorter chunks leave less time for the picture to drift between the pinned portraits.

**Each run renders the next unfinished chunk**, found from the files on disk (`auto_chunk`). There is no counter to reset: a new voiceover starts at chunk 1 in its own folder, and an interrupted render continues where it stopped.

H3 only generates **17k+5 frames** (124, 141 ... 362), so chunk lengths are planned on that grid; the extra pinned frame at each join is dropped, so nothing plays twice.

**motion_carry** (experimental, default off): with 5 or 22 frames, each chunk opens on the previous chunk's last frames instead of the portrait, so head motion continues across the join. Choose it before a render starts; change `name_suffix` when switching it."""


def note_models(v):
    if v["switch"]:
        speed = ("**Turbo on** (default): the 8-step **768p** Turbo LoRA with its trained "
                 "video/audio shift of 6/3. **Turbo off**: 20 steps of the base model at "
                 "its default shift 12/3 - the best quality, about 2.5x slower.")
        sparse = ("**Model Sparse Attention** (Sol-Attn) is **bypassed** here to keep this "
                  "the quality reference. Select it and press Ctrl+B to enable it.")
    else:
        speed = (f"**{v['lora_title']}**, {v['steps']} steps, with its trained video/audio "
                 f"shift of {v['shift'][0]:g}/{v['shift'][1]:g}.")
        sparse = ("**Model Sparse Attention** (Sol-Attn) is **on**. It skips low-weight "
                  "attention blocks on long sequences; attention to the portraits, text "
                  "and voiceover stays exact. Ctrl+B bypasses it to compare quality.")
    return f"""## 4. Models + speed - loaded automatically

Nothing to set here. Setup downloads every file on first boot.

{speed}

**Attention backend:** Comfy Kitchen INT8 attention, ComfyUI's built-in quantized attention (the same idea as SageAttention). Set it to `pytorch attention` to compare quality.

{sparse}"""


NOTE_GUIDE = """## 5. Portrait + voice guidance

- The portrait is cropped to the canvas and pinned as the **first and last frame** of the chunk (with motion carry, the first frames come from the previous chunk instead).
- The chunk's slice of your voiceover is pinned at frame 0, so H3 animates the lips to your audio instead of inventing its own speech.
- The text encoder reads the prompt together with the portraits.

Nothing to set here."""

NOTE_SAMPLING = """## 6. Sampling + decode

Fixed seed, Euler sampler and the step count the Turbo LoRA was distilled for. Nothing to set here.

Out of memory? Lower `max_seconds` on Split (stage 3), or use the Draft workflow."""


def note_output(batch):
    where = ("Each file's video is saved as `output/<file name>" if batch
             else "The final video is saved as `output/<audio name>")
    return f"""## 7. Save + stitch

Each chunk is saved to `output/h3_longform/<name>/chunk_0000.mp4`, ... as it finishes, so you can check one early.

{where}<suffix>.mp4`, stitched with your **original** audio track, not H3's reconstruction.

**Resume:** run again; finished chunks are skipped. **Redo one chunk:** delete its file and run; it is re-rendered and the video re-stitched. **Render a finished video again:** change `name_suffix`, or delete its folder under `output/h3_longform/`."""


def note_start(w, v):
    if w["batch"]:
        audio = ("2. **Audio From Folder** - put your voiceovers in `ComfyUI/input/batch_audio/` "
                 "(.wav, .mp3, .flac, .m4a, .aac, .ogg, .opus). Each file becomes its own "
                 "video, in name order. Upload them with Jupyter (port 8888) or "
                 "FileBrowser.")
        test = "4. Run with **batch count 1** to test the first chunk of the first file."
        run = ("5. Run again with a large batch count: roughly the **total seconds of all "
               "files / 8**, plus a margin. Each run renders the next unfinished chunk of "
               "the next unfinished file. When every file is done the queue is cleared, "
               "so extra runs cost nothing.")
        more = ("**Add files later:** drop them in the folder and run again; finished "
                "videos are skipped. The same portrait and description are used for every "
                "file.")
    else:
        audio = "2. **Load Voiceover** - your speech track, any length."
        test = "4. Run with **batch count 1** to test the first chunk."
        run = ("5. Run again with a batch count of at least **audio seconds / 8**, plus a "
               "margin. Each run renders the next unfinished chunk; extra runs are "
               "skipped in milliseconds.")
        more = ("**Next video:** load the new voiceover and run. It starts at chunk 1 in "
                "its own folder; there is nothing to reset.")
    return f"""# {v['title']}{' - Batch' if w['batch'] else ''}

{v['summary']}

**Red nodes are the ones you set.** Everything else runs on its own. The stages run left to right, 1 to 7, each with a note underneath.

1. **Load Portrait** - one person facing the camera, face clearly visible.
{audio}
3. **Talking-Head Prompt** - describe the person and background in the portrait.
{test} Check `output/h3_longform/<name>/chunk_0000.mp4`: lips, face, background.
{run}
6. The final video lands in `output/`, then the queue is cleared.

{more}

If the batch count box stops at 100, click Run several times: the queue adds up.

Licence and full instructions: https://github.com/tenitsky/minimax-h3"""


# ------------------------------------------------------------------------ graph
class Graph:
    def __init__(self):
        self.nodes, self.links = [], []
        self._link_id = 0

    def node(self, nid, ntype, inputs=(), outputs=(), widgets=None, title=None,
             models=None, color=None, cnr="comfy-core", mode=0, size=(440, 100)):
        props = {"Node name for S&R": ntype}
        if cnr:
            props["cnr_id"] = cnr
        if models:
            # (filename, models/ subfolder, download URL)
            props["models"] = [{"name": n, "url": url, "directory": d}
                               for n, d, url in models]
        n = {
            "id": nid, "type": ntype, "pos": [0, 0], "size": list(size),
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


def md(g, nid, title, text, size):
    g.node(nid, "MarkdownNote", widgets=[text], title=title, cnr=None,
           color=("#222", "#000"), size=size)


# Stage columns: (title, colour, node ids top to bottom, note ids below the group)
COLUMN_W, NODE_W, GAP_X, GAP_Y, TITLE_H, PAD = 520, 460, 140, 50, 120, 30
STAGE_COLORS = {"input": "#a1309b", "prompt": "#b06634", "plan": "#8a8a2c",
                "models": "#3f789e", "guide": "#4a7a4a", "sample": "#5a5a8a",
                "output": "#8a4a4a"}


def lay_out(g, stages, start_notes):
    """Place stages as spaced columns, each a titled group with notes under it."""
    nodes = {n["id"]: n for n in g.nodes}
    groups = []
    x = 0
    # Start-here column on the far left.
    y = 0
    for nid in start_notes:
        nodes[nid]["pos"] = [x, y]
        y += nodes[nid]["size"][1] + GAP_Y
    x += COLUMN_W + GAP_X
    for gid, (title, color, ids, notes) in enumerate(stages, 1):
        y = TITLE_H
        for nid in ids:
            nodes[nid]["pos"] = [x + PAD, y]
            y += nodes[nid]["size"][1] + GAP_Y
        bottom = y - GAP_Y + PAD
        groups.append({"id": gid, "title": title,
                       "bounding": [x, 0, COLUMN_W, bottom],
                       "color": STAGE_COLORS[color], "font_size": 26, "flags": {}})
        y = bottom + 60
        for nid in notes:
            nodes[nid]["pos"] = [x, y]
            nodes[nid]["size"][0] = COLUMN_W
            y += nodes[nid]["size"][1] + GAP_Y
        x += COLUMN_W + GAP_X
    return groups


def build(variant="standard", batch=False):
    v = VARIANTS[variant]
    w = next(w for w in WORKFLOWS if w["variant"] == variant and w["batch"] == batch)
    g = Graph()
    INPUT = ("#322", "#533")    # red: things you set
    W, I = True, False           # input is a widget / a plain socket
    side = v["canvas"]

    def size(h):
        return (NODE_W, h)

    # ---------------------------------------------------------------- notes
    md(g, 1, "Start here", note_start(w, v), (COLUMN_W, 880))
    md(g, 2, "Licence (read first)", NOTE_LICENSE, (COLUMN_W, 360))
    md(g, 3, "About stage 1", NOTE_INPUTS, (COLUMN_W, 470))
    md(g, 4, "About stage 2", NOTE_PROMPT, (COLUMN_W, 560))
    md(g, 5, "About stage 3", NOTE_PLAN, (COLUMN_W, 560))
    md(g, 6, "About stage 4", note_models(v), (COLUMN_W, 420))
    md(g, 7, "About stage 5", NOTE_GUIDE, (COLUMN_W, 300))
    md(g, 8, "About stage 6", NOTE_SAMPLING, (COLUMN_W, 220))
    md(g, 9, "About stage 7", note_output(batch), (COLUMN_W, 380))

    # ---------------------------------------------------------------- 1. inputs
    g.node(10, "LoadImage", outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
           widgets=["person_portrait.png", "image"],
           title="Load Portrait (identity anchor)", color=INPUT, size=size(500))
    if batch:
        g.node(11, "H3LongformAudioFolder",
               outputs=[("audio", "AUDIO"), ("name", "STRING"), ("filename", "STRING"),
                        ("file_number", "INT"), ("total_files", "INT")],
               widgets=["batch_audio", v["suffix"], True],
               title="Audio From Folder (one video per file)", color=INPUT,
               cnr="comfyui-h3-longform", size=size(190))
    else:
        g.node(11, "LoadAudio", outputs=[("AUDIO", "AUDIO")],
               widgets=["voiceover.mp3", None, None],
               title="Load Voiceover", color=INPUT, size=size(150))
        g.node(21, "H3LongformAudioName",
               outputs=[("name", "STRING"), ("filename", "STRING")],
               widgets=[".mp4", "", v["suffix"]],
               title="Output name (audio filename + name_suffix)",
               cnr="comfyui-h3-longform", size=size(130))
    g.node(13, "PrimitiveInt", outputs=[("INT", "INT")], widgets=[side, "fixed"],
           title="Width", size=size(90))
    g.node(14, "PrimitiveInt", outputs=[("INT", "INT")], widgets=[side, "fixed"],
           title="Height", size=size(90))

    # ---------------------------------------------------------------- 2. prompt
    g.node(30, "H3LongformPrompt",
           inputs=[("length", "INT", W, False), ("carry_frames", "INT", W, False)],
           outputs=[("prompt", "STRING")],
           widgets=[124, 0] + PROMPT_FIELDS,
           title="Talking-Head Prompt (describe subject + background)", color=INPUT,
           cnr="comfyui-h3-longform", size=size(720))

    # ---------------------------------------------------------------- 3. chunk plan
    g.node(20, "H3LongformSplit",
           inputs=[("audio", "AUDIO", I, False)]
           + ([("session", "STRING", I, True)] if batch else []),
           outputs=[("audio_chunk", "AUDIO"), ("length", "INT"), ("keep_frames", "INT"),
                    ("total_chunks", "INT"), ("is_last", "BOOLEAN"),
                    ("alignment", "STRING"), ("guide_audio", "AUDIO"),
                    ("carry_frames", "INT"), ("chunk_index", "INT")],
           widgets=list(SPLIT_WIDGETS),
           title="Split Audio Chunk (next unfinished chunk)", cnr="comfyui-h3-longform",
           size=size(380))

    # ---------------------------------------------------------------- 4. models
    g.node(40, "UNETLoader", outputs=[("MODEL", "MODEL")], widgets=[UNET, "default"],
           models=[(UNET, "diffusion_models", HF + "diffusion_models/" + UNET)],
           size=size(90))
    g.node(41, "LoraLoaderModelOnly", inputs=[("model", "MODEL", I, False)],
           outputs=[("MODEL", "MODEL")], widgets=[v["lora"], 1], title=v["lora_title"],
           models=[(v["lora"], "loras", v["lora_url"] + v["lora"])], size=size(90))
    shift_video, shift_audio = v["shift"]
    g.node(75, "MiniMaxH3SigmaShift", inputs=[("model", "MODEL", I, False)],
           outputs=[("MODEL", "MODEL")], widgets=[shift_video, shift_audio],
           title=f"Turbo sigma shift (video {shift_video:g} / audio {shift_audio:g})",
           size=size(110))
    if v["switch"]:
        g.node(45, "PrimitiveBoolean", outputs=[("BOOLEAN", "BOOLEAN")], widgets=[True],
               title="Turbo LoRA on (8 steps) / off (20 steps)", color=INPUT,
               size=size(60))
        g.node(46, "PrimitiveInt", outputs=[("INT", "INT")], widgets=[20, "fixed"],
               title="Steps (base)", size=size(90))
        g.node(47, "PrimitiveInt", outputs=[("INT", "INT")], widgets=[v["steps"], "fixed"],
               title="Steps (turbo)", size=size(90))
        g.node(48, "ComfySwitchNode",
               inputs=[("on_false", "MODEL", I, False), ("on_true", "MODEL", I, False),
                       ("switch", "BOOLEAN", W, False)],
               outputs=[("output", "MODEL")], widgets=[True],
               title="If/Else Switch (Model)", size=size(110))
        g.node(49, "ComfySwitchNode",
               inputs=[("on_false", "INT", I, False), ("on_true", "INT", I, False),
                       ("switch", "BOOLEAN", W, False)],
               outputs=[("output", "INT")], widgets=[True], title="If/Else Switch (Steps)",
               size=size(110))
    g.node(76, "ModelAttentionBackend", inputs=[("model", "MODEL", I, False)],
           outputs=[("model", "MODEL")], widgets=["comfy kitchen attention"],
           title="Attention backend (INT8, like SageAttention)", size=size(80))
    g.node(77, "BlockSparseAttention", inputs=[("model", "MODEL", I, False)],
           outputs=[("model", "MODEL")], widgets=list(SPARSE_WIDGETS),
           title="Model Sparse Attention (Sol-Attn)" + ("" if v["sparse"] else
                                                          " - Ctrl+B to enable"),
           mode=0 if v["sparse"] else BYPASS, size=size(270))
    g.node(42, "CLIPLoader", outputs=[("CLIP", "CLIP")], widgets=[TE, "minimax", "default"],
           models=[(TE, "text_encoders", HF + "text_encoders/" + TE)], size=size(110))
    g.node(43, "VAELoader", outputs=[("VAE", "VAE")], widgets=[VIDEO_VAE],
           title="Video VAE", models=[(VIDEO_VAE, "vae", HF + "vae/" + VIDEO_VAE)],
           size=size(70))
    g.node(44, "VAELoader", outputs=[("VAE", "VAE")], widgets=[AUDIO_VAE],
           title="Audio VAE", models=[(AUDIO_VAE, "vae", HF + "vae/" + AUDIO_VAE)],
           size=size(70))

    # ---------------------------------------------------------------- 5. guidance
    g.node(50, "ImageScale",
           inputs=[("image", "IMAGE", I, False), ("width", "INT", W, False),
                   ("height", "INT", W, False)],
           outputs=[("IMAGE", "IMAGE")], widgets=["lanczos", side, side, "center"],
           title="Crop portrait to canvas", size=size(170))
    g.node(53, "H3LongformCarry",
           inputs=[("portrait", "IMAGE", I, False), ("chunk_index", "INT", W, False),
                   ("carry_frames", "INT", W, False)]
           + ([("session", "STRING", I, True)] if batch else []),
           outputs=[("first_frame", "IMAGE"), ("carry_clip", "IMAGE")],
           widgets=[0, 0], title="Opening: portrait or motion carry",
           cnr="comfyui-h3-longform", size=size(150))
    g.node(51, "MiniMaxH3ImageToVideo",
           inputs=[("clip", "CLIP", I, False), ("vae", "VAE", I, False),
                   ("first_frame", "IMAGE", I, True), ("last_frame", "IMAGE", I, True),
                   ("prompt", "STRING", W, False), ("width", "INT", W, False),
                   ("height", "INT", W, False), ("length", "INT", W, False)],
           outputs=[("positive", "CONDITIONING"), ("LATENT", "LATENT")],
           widgets=["", side, side, 124], title="Portraits + prompt (first/last frame)",
           size=size(330))
    g.node(52, "MiniMaxH3AddGuide",
           inputs=[("positive", "CONDITIONING", I, False), ("latent", "LATENT", I, False),
                   ("vae", "VAE", I, True), ("audio_vae", "VAE", I, True),
                   ("image", "IMAGE", I, True), ("audio", "AUDIO", I, True)],
           outputs=[("positive", "CONDITIONING")], widgets=[0],
           title="Pin voiceover + optional motion carry (frame 0)", size=size(200))

    # ---------------------------------------------------------------- 6. sampling
    g.node(60, "BasicGuider",
           inputs=[("model", "MODEL", I, False), ("conditioning", "CONDITIONING", I, False)],
           outputs=[("GUIDER", "GUIDER")], size=size(70))
    g.node(61, "RandomNoise", outputs=[("NOISE", "NOISE")], widgets=[42, "fixed"],
           size=size(100))
    # The Turbo LoRAs are distilled on Euler flow steps (authors' ComfyUI guide).
    g.node(62, "KSamplerSelect", outputs=[("SAMPLER", "SAMPLER")], widgets=["euler"],
           size=size(70))
    g.node(63, "BasicScheduler",
           inputs=[("model", "MODEL", I, False)]
           + ([("steps", "INT", W, False)] if v["switch"] else []),
           outputs=[("SIGMAS", "SIGMAS")], widgets=["simple", v["steps"], 1],
           size=size(120))
    g.node(64, "SamplerCustomAdvanced",
           inputs=[("noise", "NOISE", I, False), ("guider", "GUIDER", I, False),
                   ("sampler", "SAMPLER", I, False), ("sigmas", "SIGMAS", I, False),
                   ("latent_image", "LATENT", I, False)],
           outputs=[("output", "LATENT"), ("denoised_output", "LATENT")], size=size(150))
    g.node(65, "VAEDecode",
           inputs=[("samples", "LATENT", I, False), ("vae", "VAE", I, False)],
           outputs=[("IMAGE", "IMAGE")], size=size(70))

    # ---------------------------------------------------------------- 7. output
    fallback = ["batch" + v["suffix"], "batch" + v["suffix"] + ".mp4"] if batch else \
        ["run1" + v["suffix"], "h3_talking_head" + v["suffix"] + ".mp4"]
    g.node(70, "H3LongformWrite",
           inputs=[("images", "IMAGE", I, False), ("original_audio", "AUDIO", I, False),
                   ("chunk_audio", "AUDIO", I, True), ("chunk_index", "INT", W, False),
                   ("total_chunks", "INT", W, False), ("keep_frames", "INT", W, False),
                   ("session", "STRING", W, False), ("filename", "STRING", W, False),
                   ("trim_start", "INT", W, True)],
           outputs=[("status", "STRING")],
           # In a batch, the folder node clears the queue once every file is done.
           widgets=[0, 1, 123] + fallback + [not batch, 0],
           title="Write chunk + stitch final video", cnr="comfyui-h3-longform",
           size=size(340))

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
    if batch:
        # The folder node names each file's session; Split and Carry take it
        # directly so all three agree on the folder.
        L(11, 1, 20, "session")
        L(11, 1, 53, "session")
        L(11, 1, 70, "session")
        L(11, 2, 70, "filename")
    else:
        L(21, 0, 70, "session")
        L(21, 1, 70, "filename")

    # ---------------------------------------------------------------- layout
    models = [40, 41, 75] + ([46, 47, 48, 49] if v["switch"] else []) \
        + [76, 77, 42, 43, 44]
    # The Turbo toggle is a setting, so it sits with the other inputs.
    inputs = [10, 11] + ([] if batch else [21]) + [13, 14] + ([45] if v["switch"] else [])
    stages = [
        ("1. Your inputs", "input", inputs, [3]),
        ("2. Describe the shot", "prompt", [30], [4]),
        ("3. Chunk plan (automatic)", "plan", [20], [5]),
        ("4. Models + speed (automatic)", "models", models, [6]),
        ("5. Portrait + voice guidance", "guide", [50, 53, 51, 52], [7]),
        ("6. Sampling", "sample", [60, 61, 62, 63, 64, 65], [8]),
        ("7. Save + stitch", "output", [70], [9]),
    ]
    groups = lay_out(g, stages, [1, 2])

    return {
        "id": w["id"],
        "revision": 0,
        "last_node_id": max(n["id"] for n in g.nodes),
        "last_link_id": g._link_id,
        "nodes": g.nodes,
        "links": g.links,
        "groups": groups,
        "config": {},
        "extra": {"ds": {"scale": 0.55, "offset": [80, 80]}},
        "version": 0.4,
    }


def build_fast(batch=False):
    return build("fast", batch)


def build_fast_draft(batch=False):
    return build("draft", batch)


if __name__ == "__main__":
    for w in WORKFLOWS:
        wf = build(w["variant"], w["batch"])
        path = os.path.join(WORKFLOW_DIR, w["path"])
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(wf, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {os.path.normpath(path)}: {len(wf['nodes'])} nodes, "
              f"{len(wf['links'])} links")
