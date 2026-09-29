# MiniMax H3: RunPod ComfyUI template (long-form talking avatar)

A one-shot template that installs **MiniMax H3** (Hailuo 3.0, open weights, natively
supported in ComfyUI) on RunPod. It ships a workflow that turns **one portrait and a
voiceover of any length** into a talking-presenter video, rendered in chunks and then
stitched. It's the same approach as the LTX-2.5 template, rebuilt around H3's frame
grid.

## Licence: read this first

The MiniMax H3 Community License **excludes the United States, the European Union, the
United Kingdom and South Korea**. Using, running or displaying the weights or their
outputs there isn't licensed. MiniMax does grant licences for those regions on
application: https://platform.minimax.io/h3-license

Where the pod runs matters as well as where you are, so pick the RunPod datacenter to
match. The licence also carries an Acceptable Use Policy. Among other things, it bars
impersonating people without consent, content that harms minors, and election
misinformation. Any commercial product has to display "MiniMax H3". The full text is
here: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE

This repo doesn't redistribute the weights. `setup.sh` downloads them from the
Comfy-Org repo on first boot.

## How long can H3 go?

About **15 seconds per generation**: 362 frames at 24 fps, H3's trained ceiling.
LTX-2.5 managed about 10 seconds, so a 7-minute voiceover takes roughly 30–35 chunks
instead of about 50.

H3 only accepts frame counts on a **17k+5 grid**: 124, 141, 158 … 362, or about 5.2 to
15.1 seconds. The bundled node pack plans chunks on that grid. See
`custom_nodes/comfyui-h3-longform/README.md` for why that matters.

## What boots

On first pod boot, `setup.sh`:

1. Copies the baked ComfyUI from `/opt/comfyui-baked` to `/workspace/runpod-slim/ComfyUI`
   (self-healing, skip-if-present)
2. Checks that the image's ComfyUI has `MiniMaxH3AddGuide`, and stops if not
3. Installs the bundled `comfyui-h3-longform` nodes (4 nodes, no dependencies besides
   ffmpeg)
4. Downloads the H3 model files (skip-if-exists, resumable)
5. Installs `minimax_h3_long_video_workflow` into ComfyUI's **Workflows** sidebar
6. Hands over to `/start.sh` (ComfyUI on port **8188**)

## Models pulled

All files come from [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3),
which is **ungated**, so no token is needed. The ModelScope copy of the same repo is
the automatic fallback.

| File | Size | Destination (`ComfyUI/models/`) |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 21.0 GB | `diffusion_models/` |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15.7 GB | `text_encoders/` |
| `minimax_h3_video_vae_int8_convrot.safetensors` | 2.8 GB | `vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB | `vae/` |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 2.0 GB | `loras/` |
| *optional* `minimax_h3_ref2va_pruned_int8_convrot.safetensors` + ref2v turbo LoRA | 23 GB | `diffusion_models/`, `loras/` |
| *alternative* `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.1 GB | `text_encoders/` |

The default set is about **42 GB**. With ordinary regional storage, use a **64 GB
volume at minimum; 80 GB is recommended**, and 100 GB if you also enable Ref2VA.
For Global Volumes, use the automatic storage layout below instead.

## Automatic Global Volume storage

Attach a RunPod **Global Volume** when deploying, and **explicitly set its mount
path to `/workspace-global`**. RunPod currently defaults a lone Global Volume to
`/workspace`; that default is wrong for this template because ComfyUI needs a
regular working filesystem there. Setup detects the configured mount automatically;
no manual copy commands or extra regional
network volume are needed. Use one writing pod per `minimax-h3` storage namespace.

| Data | Location and behavior |
|---|---|
| Models | `/workspace-global/minimax-h3/models/`; downloaded once, verified, then reused through local links |
| Uploaded portraits and audio | `/workspace-global/minimax-h3/input/`; ComfyUI's input directory points here |
| Saved sidebar workflows | Edited locally; copied to `/workspace-global/minimax-h3/workflows/` after successful UI saves |
| ComfyUI, Python, download cache, rendering and stitching | Pod working disk under `/workspace` |
| Completed chunks, carry tails and final videos | Copied automatically to `/workspace-global/minimax-h3/output/` and checksum-verified |

Allocate **100 GB of pod-local working disk backing `/workspace`** as a starting
point (not a measured capacity guarantee; long/high-resolution renders need more).
If `/workspace` is on the container disk, set the **container disk to 100 GB**.
If it is a separate pod-local volume disk, size that disk to 100 GB instead.
The old 5 GB container setting is only suitable when `/workspace` has its own
sufficiently large working volume. Setup requires at least **40 GiB free after
copying ComfyUI**, for a staged model download and rendering scratch. Before model
downloads, setup also tests Global Volume creation, reading, overwriting, checksums,
and access through a local model symlink.

Models download to local scratch one at a time, then copy to Global Storage and
are verified before their local staging copy is removed. Uploads do not rely on
remote atomic renames or locks. The node copies each chunk and its motion-carry
tail before reporting that chunk complete; the final video is backed up before
`FINISHED` appears or the pending queue is cleared. A failed backup stops that job
with an error, retaining local files for retry. Stop the pending queue after an
error and restart from chunk 0 once storage is available again.

**Replacement pod:** attach the same Global Volume and use the same template.
Setup automatically restores verified render files to the new working disk.
Open your saved workflow, set `chunk_index` to **0**, and queue again; completed
chunks are skipped. A copy of the workflow used for a render is also saved at
`output/h3_longform/<session>/workflow.json` when queued from the ComfyUI UI.
Queue submission itself is manual; files are restored automatically.

An abrupt termination can lose the currently rendering/uploading chunk. Completed,
verified backups survive. Partial copies are ignored on restore. Keep the same
session settings when resuming; use a new session for a new render. To permanently
remove a render, delete its global backup as well as its local output, otherwise
the next startup restores it. The template never automatically deletes backups.

For existing installations, input migration preserves local originals in
`input.before-global`; conflicting filenames stop setup rather than overwrite your
files. Existing verified local models are reused. An older version's workflow
symlink is converted back to a local directory automatically. ComfyUI saves
workflows with atomic rename, so the editor must not write directly to object
storage. Saved workflows are copied and verified before the UI save returns
success; a failed backup reports an error while retaining the local save.

Set **`H3_GLOBAL_STORAGE=1`** to require the volume (fail at boot if it is absent),
or `0` to keep the ordinary storage layout on a fresh installation. Default `auto`
uses Global Storage only when the mount is attached. Existing symlinks are not
automatically undone by setting `0`.

This implementation has Linux CI for transfer-failure, restore, symlink, workflow
save, and render regression tests, but has **not yet been tested on a live RunPod
Global Volume**. RunPod's beta
is object-storage-backed and recommends regional storage for workloads needing
full POSIX behavior or frequent/concurrent writes; rendering stays local for that
reason. Its cross-pod visibility is eventually consistent: stop writing on the old
pod before moving, and wait for backups to become visible if a replacement cannot
see them immediately. See [RunPod Global Volume limitations](https://docs.runpod.io/storage/globalvolume/overview)
and [current mount-path behavior](https://docs.runpod.io/storage/globalvolume/globalvolume-pods#mount-paths).

## RunPod template settings

| Setting | Value |
|---|---|
| Base image | `runpod/comfyui:1.4.0-rc.164-comfyuiv0.35.0-cuda12.8` (the same image as the LTX template) |
| Container disk | **100 GB** for the Global Volume setup below |
| Pod-local volume disk | **0 GB** for this setup; `/workspace` uses the container working disk |
| Template volume mount path | `/workspace-global`; also verify the attached Global Volume's own mount field at deployment |
| Global Volume | Attach at deployment; elastic capacity, no 64/80 GB size to enter |
| Ports | HTTP `8188` (ComfyUI), HTTP `8888` (JupyterLab) |
| Env vars | `H3_GLOBAL_STORAGE=1`, `FILEBROWSER_PASSWORD` (your own password); `HF_TOKEN` optional |

These values are for **one Global Volume plus the container disk**. You do not
need a regional network volume. For the traditional regional layout instead, set
`H3_GLOBAL_STORAGE=0`, mount the regional volume at `/workspace` with 64–80 GB, and
use a 5 GB container disk.

At deployment, select the template and GPU first, then **Storage → Persistent
storage → + Add volume** and choose your Global Volume. Edit its mount to
**`/workspace-global`** before deploying. Leave `/workspace` on the container disk.
Start with a GPU with 80–96 GB VRAM for initial testing; actual H3 memory usage has
not been benchmarked for this workflow.

Suggested environment values (all others may stay at defaults):

```text
H3_GLOBAL_STORAGE=1
H3_GLOBAL_MOUNT=/workspace-global
JUPYTER_NO_AUTH=1
FILEBROWSER_PASSWORD=<your-own-password>
```

`JUPYTER_NO_AUTH=1` retains this template's existing no-login Jupyter behavior.
For authenticated Jupyter, use `JUPYTER_NO_AUTH=0` and set `JUPYTER_PASSWORD`.
Do not override `COMFYUI_PATH`: the base image starts ComfyUI at its fixed path.

> **The image needs ComfyUI ≥ 0.35.0.** `MiniMaxH3AddGuide`, the node that pins your
> voiceover, doesn't exist before 0.34.0, and tags like `runpod/comfyui:cuda12.8` or
> `…-comfyuiv0.30.0-…` ship older builds. `setup.sh` stops at boot if the node is
> missing. It deliberately doesn't update ComfyUI core, because on these images that
> can reinstall torch and break the CUDA build.

**Container start command.** Paste this JSON into RunPod's container start command
field (also provided as [`runpod-start.json`](runpod-start.json)). It retries the
clone without overwriting RunPod's DNS configuration:

```json
{
  "entrypoint": ["bash", "-lc"],
  "cmd": ["set -e; for attempt in 1 2 3 4 5 6 7 8 9 10; do repo=$(mktemp -d /tmp/minimax-h3.XXXXXX); if git clone --depth 1 https://github.com/tenitsky/minimax-h3.git \"$repo\"; then exec bash \"$repo/setup.sh\"; fi; sleep 5; done; echo 'FATAL: unable to clone minimax-h3 after 10 attempts' >&2; exit 1"]
}
```

For the first launch, check the pod logs for **`Global Volume read/write/checksum
probe passed`**, then **`enabled at /workspace-global/minimax-h3`**. Model downloads
and first-time copy verification can take a while. Open ComfyUI once it is listening
on port 8188. Test two chunks before starting a long render; after each one, expect
**`chunk N backed up and verified`**, and after stitching expect **`final video
backed up and verified`** followed by **`FINISHED`**.

## After it boots

1. Open ComfyUI (port 8188), then open the **Workflows** sidebar and choose
   **`minimax_h3_long_video_workflow`**.
2. Set **Load Portrait** and **Load Voiceover**. The output is named after the audio
   file automatically.
3. **Test one chunk first:** use batch count 1 and check lip sync and identity in
   `output/h3_longform/<name>/chunk_0000.mp4`.
4. Set `chunk_index` back to 0, then queue with a batch count of **at least the number
   of chunks**, roughly audio seconds ÷ 12, plus margin. Surplus items are skipped in
   milliseconds.
5. The last chunk writes `output/<audio name>.mp4` and clears the queue.

**Resume** after an interruption: set `chunk_index` to 0 and queue again. Finished
chunks are skipped.

ComfyUI's own examples are under **Workflow → Browse Templates → Video → MiniMax H3**
(T2V, I2V, R2V, Multiframe, ControlNet). I2V works with the default downloads. R2V
needs `DOWNLOAD_REF2VA=1`.

> `setup.sh` skips existing workflows, so a workflow you edited on the pod is never
> overwritten on reboot. The flip side is that pushing an updated workflow to this
> repo won't reach an existing volume until you delete the old file from
> `ComfyUI/user/default/workflows/`.

### How the workflow works

- By default, the portrait is center-cropped to the canvas, then pinned as the
  **first and last** frame of every chunk to limit identity drift. The presenter
  returns to the portrait pose at each boundary; cuts in pauses may help hide this.
  Optional motion carry replaces the opening portrait after chunk 0 with a short
  clip from the previous chunk (see below).
- `MiniMaxH3AddGuide` pins the chunk's slice of your voiceover at frame 0. H3 then
  generates a picture that matches the sound, instead of inventing its own speech.
- The final video uses your **original** audio track, encoded as AAC, rather than
  H3's reconstruction. Chunk positions share a 24 fps timeline to avoid cumulative
  timing drift; model lip-sync quality still needs testing.
- **Turbo LoRA** is on by default (8 steps). Switch it off for 20 steps without the
  LoRA, which gives the most quality and runs about 2.5× slower per chunk.
- Describe delivery and motion in the prompt, not appearance. The prompt already
  follows MiniMax's FL2VA prompt format, and the per-chunk `alignment` line is added
  for you.

### Optional motion carry (experimental)

On **Split Audio Chunk**, set `motion_carry` before starting:

| Setting | Opening after chunk 0 |
|---|---|
| `off` (default) | Original portrait |
| `5 frames (~0.2s)` | Previous chunk's last 5 written frames |
| `22 frames (~0.9s)` | Previous chunk's last 22 written frames |

The carried clip and its original audio become a lead-in. Write trims those frames
so they never play twice. The portrait still anchors each chunk's final frame to
limit identity drift. The lead-in counts toward the generation limit, so each later
chunk contains slightly less new footage.

Queue chunks **in order, starting at 0**. Resume the same way; completed chunks with
their carry files are skipped, and missing carry files cause those chunks to render
again. Tails remain in `output/h3_longform/<session>/.carry/` for retries. Delete the
session folder only when you no longer need to resume it.

Use a **fresh session** when changing carry, audio, portrait, resolution, or chunk
settings: unlink the session input on Write and enter a new name, or remove the old
session's output files. Existing chunks are not checked against changed settings.

Test at least two chunks to compare joins. Frame accounting, audio slicing, and
stitching are tested locally with synthetic frames; H3 motion continuity, face
identity, and lip-sync quality have **not** been tested on a GPU.

For an existing pod, update the custom node pack and restart ComfyUI, then import
the new workflow JSON. Setup preserves saved workflows, so rebooting alone will
not replace the old graph.

## GPU guidance

These are estimates from the file sizes. **They haven't been measured on a pod yet.**

| GPU | Notes |
|---|---|
| 80–96 GB (A100 / H100 / RTX PRO 6000) | Comfortable at 15 s chunks. Blackwell cards run the nvfp4 text encoder natively |
| 48 GB (A6000 / L40S) | Should work. If a 15 s chunk runs out of memory, lower `max_seconds` on Split (for example to 10) |
| 24 GB | Not recommended for 768p at 15 s |

If the text encoder won't load on your GPU, set `H3_TEXT_ENCODER=int8` and choose that
file in the workflow's CLIPLoader.

## Template env vars (all optional)

| Var | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Not required (the repo is ungated). It raises Hugging Face rate limits |
| `H3_TEXT_ENCODER` | `nvfp4` | `int8` downloads the 27 GB int8 Qwen3-VL encoder instead |
| `DOWNLOAD_TURBO_LORA` | `1` | Leave this at `1`. The workflow's LoRA loader is validated even when Turbo is switched off, so without the file the workflow won't queue |
| `DOWNLOAD_REF2VA` | `0` | `1` also pulls the Ref2VA model and LoRA for ComfyUI's R2V template |
| `JUPYTER_NO_AUTH` | `1` | Disables Jupyter login (anyone with the URL gets a shell). Set `0` to keep the image's auth |
| `JUPYTER_PASSWORD` | *(empty)* | Jupyter token, used when `JUPYTER_NO_AUTH=0` |
| `FILEBROWSER_PASSWORD` | `adminadmin12` | FileBrowser on port 8080 (user `admin`). **Change the default** |
| `COMFYUI_PATH` | `/workspace/runpod-slim/ComfyUI` | Leave unset; must match the base image's fixed startup path |
| `HF_HOME` | `/workspace/.cache/huggingface` | Keeps the HF cache on the volume, not the 5 GB container disk |
| `H3_GLOBAL_STORAGE` | `auto` | Detect Global Volume; `1` requires it, `0` disables automatic configuration |
| `H3_GLOBAL_MOUNT` | `/workspace-global` | Global Volume mount point; must be an actual mounted filesystem |
| `H3_GLOBAL_ROOT` | `<mount>/minimax-h3` | Global namespace for models, inputs, workflows and verified output backups |
| `H3_LOCAL_MIN_FREE_GB` | `40` | Minimum free GiB required on the ComfyUI working disk in Global mode |

ComfyUI runs from its own virtualenv (`$COMFYUI_PATH/.venv-cu128`). If you install
anything by hand, use that interpreter:

```bash
/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python -m pip install <package>
```

## Repository layout

```
├── setup.sh                               # RunPod boot script
├── README.md
├── custom_nodes/
│   └── comfyui-h3-longform/               # Split / Carry / Write + Stitch / Name From Audio File
├── tools/
│   ├── build_workflow.py                  # generates the workflow JSON; re-run after edits
│   ├── test_longform.py                   # CPU + ffmpeg regression checks
│   ├── test_storage.py                    # backup failure and replacement-pod restore checks
│   └── test_setup.py                      # Bash download checks using local fixtures
└── workflows/
    └── minimax_h3_long_video_workflow.json
```

## References

- ComfyUI guide: https://docs.comfy.org/tutorials/video/minimax/minimax-h3
- Weights (ComfyUI format): https://huggingface.co/Comfy-Org/MiniMax-H3
- Original release, licence and prompt guides: https://huggingface.co/MiniMaxAI/MiniMax-H3
- Node source: `comfy_extras/nodes_minimax_h3.py` in ComfyUI
