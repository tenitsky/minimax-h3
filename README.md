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

1. Requires a separate POSIX volume at `/workspace`; refuses container-disk or
   Global Volume working storage
2. Copies the baked ComfyUI from `/opt/comfyui-baked` to `/workspace/runpod-slim/ComfyUI`
   (self-healing, skip-if-present)
3. Checks that the image's ComfyUI has `MiniMaxH3AddGuide`, and stops if not
4. Installs the bundled `comfyui-h3-longform` nodes (4 nodes, no dependencies besides
   ffmpeg)
5. Downloads the H3 model files (skip-if-exists, resumable)
6. Installs `minimax_h3_long_video_workflow` and `minimax_h3_fast_workflow` into
   ComfyUI's **Workflows** sidebar
7. Hands over to `/start.sh` (ComfyUI on port **8188**, JupyterLab on **8888**)

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
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | 2.0 GB | `loras/` |
| *optional* `minimax_h3_ref2va_pruned_int8_convrot.safetensors` + ref2v turbo LoRA | 23 GB | `diffusion_models/`, `loras/` |
| *alternative* `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.1 GB | `text_encoders/` |

The default model set is about **44 GB**. Use **100 GB of Network Volume storage**
as a starting point for ComfyUI, its environment, models and renders. An existing
80 GB volume can be used if it has sufficient free space. Long/high-resolution
renders or optional model downloads may need more than 100 GB.

## Persistent Network Volume storage

Attach **one regional Network Volume at `/workspace`**. Setup installs everything
there automatically; no Global Volume or second persistent volume is needed.
The 5 GB container disk holds disposable system files.

| Data | Persistent location |
|---|---|
| ComfyUI and custom nodes | `/workspace/runpod-slim/ComfyUI/` |
| ComfyUI Python environment | `/workspace/runpod-slim/ComfyUI/.venv-cu128/` |
| Models | `ComfyUI/models/` |
| Uploaded portraits and audio | `ComfyUI/input/` |
| Saved sidebar workflows | `ComfyUI/user/default/workflows/` |
| Completed chunks, carry tails, and render workflow copies | `ComfyUI/output/h3_longform/<session>/` |
| Final videos | `ComfyUI/output/` |
| Jupyter notebooks | Save under `/workspace/` |
| Jupyter settings and IPython data | `/workspace/.jupyter/`, `/workspace/.local/share/jupyter/`, `/workspace/.ipython/` |
| Download caches | `/workspace/.cache/huggingface/` and `/workspace/.hf-cli/` |

The table's `ComfyUI/` paths are relative to `/workspace/runpod-slim/`.
Setup verifies that `/workspace` is a separate mounted volume and tests file
replacement, symlinks, and actual file execution. Network mounts with fixed
permissions such as `0777` are allowed when execution works; exact `chmod` results
are not required. It refuses to silently install the workspace onto the container disk.
Mount inspection cannot verify the provider's retention policy: select a **Network
Volume**, which persists independently of the pod. A pod-local Volume disk is lost
when its pod is deleted. See [RunPod storage types](https://docs.runpod.io/pods/storage/types).

**Restart or replacement pod:** attach the same Network Volume at `/workspace` and
use this template. The installed environment, models, notebooks and render files
remain there. Existing complete models and edited workflows are reused. Open your
workflow, set `chunk_index` to **0**, and queue again to skip completed chunks.
The currently rendering chunk may need to be rendered again after an interruption.

The Network Volume is tied to its datacenter. Keep it attached when choosing a GPU
or deploying a replacement pod. Files and system-wide package installations outside
`/workspace` are disposable. Install Python packages into ComfyUI's persistent
virtualenv using the command below.

### Updating from the earlier Global Storage template

Remove `H3_GLOBAL_STORAGE`, `H3_GLOBAL_MOUNT`, `H3_GLOBAL_ROOT`, and
`H3_LOCAL_MIN_FREE_GB` from the public template and any pod overrides. The updated
startup script also clears these obsolete variables automatically. Set the template
volume mount path to **`/workspace`** and attach your Network Volume there.
Changing the public template does not change an existing pod's mount paths; deploy
a replacement pod with the same Network Volume if its current mount is wrong.

Existing data is never deleted by this switch. If an earlier successful setup
created input, workflow or model links to another volume, startup identifies those
paths and stops. Keep the original volume available, copy the linked data onto the
Network Volume, and replace those links with the copied files/directories before
restarting. Fresh installations have no migration step.

## RunPod public template settings

| Setting | Value |
|---|---|
| Template name | `MiniMax H3 - ComfyUI + JupyterLab` |
| Compute type | **NVIDIA / GPU** |
| Public template | **On** |
| Base image | `runpod/comfyui:1.4.0-rc.164-comfyuiv0.35.0-cuda12.8` |
| Registry authentication | None; image is public |
| Container disk | **5 GB** |
| Persistent storage in the public template editor | **Volume disk**, **0 GB**; no publisher-owned volume selected |
| Template volume mount path | **`/workspace`** |
| Network Volume at deployment | Each deployer attaches their own at `/workspace`; **100 GB recommended** |
| HTTP ports | **`8188,8888`** (ComfyUI, JupyterLab) |
| TCP ports | None required; expose `22` only if SSH is wanted |
| Environment variables | `JUPYTER_NO_AUTH=1`; `HF_TOKEN` optional |
| README tab | Paste [`TEMPLATE_README.md`](TEMPLATE_README.md) |

Publish the image, command, ports and defaults. Each deployer selects their own
Network Volume when deploying. The startup script does not create or attach RunPod
volumes. No storage environment variables are required.

The template editor's **Volume disk / 0 GB** is intentional: the required Network
Volume is chosen by each user at deployment and replaces the pod-local volume disk.
The template itself does not provide persistent capacity. Deploying without an
attached volume will stop at the workspace check. Keep the mount path `/workspace`;
setup creates `runpod-slim/ComfyUI` inside it automatically.
See [attaching a Network Volume](https://docs.runpod.io/storage/network-volumes).

For GPU Compatibility, start with 80 GB or more VRAM for initial testing. Actual
H3 GPU memory usage has not been benchmarked for this workflow.

`JUPYTER_NO_AUTH=1` retains the existing no-login Jupyter behavior. For authenticated
Jupyter, use `JUPYTER_NO_AUTH=0` and set `JUPYTER_PASSWORD` at deployment.
JupyterLab opens at `/workspace`; save notebooks there for persistence.
Temporary Jupyter connection files and cookie secrets use a private directory
under `/tmp`, because Jupyter requires private permissions on those files.
Setup checks this before model downloads; notebooks and settings remain on the volume.
Do not override `COMFYUI_PATH`: the image starts ComfyUI at its fixed path.

> **The image needs ComfyUI ≥ 0.35.0.** `MiniMaxH3AddGuide`, the node that pins your
> voiceover, doesn't exist before 0.34.0, and tags like `runpod/comfyui:cuda12.8` or
> `…-comfyuiv0.30.0-…` ship older builds. `setup.sh` stops at boot if the node is
> missing. It deliberately doesn't update ComfyUI core, because on these images that
> can reinstall torch and break the CUDA build.

**Container start command.** Paste this JSON into RunPod's container start command
field (also provided as [`runpod-start.json`](runpod-start.json); readable source:
[`bootstrap.sh`](bootstrap.sh)). It checks DNS for GitHub and Hugging Face first.
Working DNS is left unchanged. If two checks fail, it saves the original
`/etc/resolv.conf` under `/tmp/minimax-h3-dns.*`, then tries public resolvers
`1.1.1.1` and `8.8.8.8`. If resolution still fails, it attempts to restore the
original file and stops with a networking error. A successful fallback remains
active for setup and model downloads. Git clone failures are retried up to 10 times.

**Existing templates need this updated start command pasted into RunPod.** A GitHub
push alone cannot fix DNS failure before the repository has been downloaded:

```json
{
  "entrypoint": [
    "bash",
    "-lc"
  ],
  "cmd": [
    "set -e\ndns_ok() {\n  timeout 10 getent ahostsv4 github.com >/dev/null 2>&1 &&\n  timeout 10 getent ahostsv4 huggingface.co >/dev/null 2>&1\n}\nif ! dns_ok; then\n  echo 'DNS lookup failed; retrying in 3 seconds...'\n  sleep 3\n  if ! dns_ok; then\n    dns_backup=$(mktemp /tmp/minimax-h3-dns.XXXXXX)\n    cp /etc/resolv.conf \"$dns_backup\"\n    echo \"Trying public DNS; original resolver saved at $dns_backup\"\n    if { printf 'nameserver 1.1.1.1\\nnameserver 8.8.8.8\\noptions timeout:2 attempts:2\\n'; } > /etc/resolv.conf && dns_ok; then\n      echo 'DNS recovered; continuing setup.'\n    else\n      cat \"$dns_backup\" > /etc/resolv.conf || true\n      echo 'FATAL: DNS still unavailable or resolver is read-only. Check RunPod networking or deploy on another host with the same Network Volume.' >&2\n      exit 1\n    fi\n  fi\nfi\nfor attempt in 1 2 3 4 5 6 7 8 9 10; do\n  repo=$(mktemp -d /tmp/minimax-h3.XXXXXX)\n  if git clone --depth 1 https://github.com/tenitsky/minimax-h3.git \"$repo\"; then\n    exec bash \"$repo/setup.sh\"\n  fi\n  sleep 5\ndone\necho 'FATAL: unable to clone minimax-h3 after 10 attempts' >&2\nexit 1\n"
  ]
}
```

For the first launch, check the logs for **`Network Volume workspace ready at
/workspace`**, then model downloads and **`Setup complete!`**. Open ComfyUI on
port 8188 and JupyterLab on port 8888 once they are listening. If ComfyUI works but
Jupyter stays on initializing, inspect `/jupyter.log` from the pod's terminal for
the actual startup error.

Startup prints the template revision so you can distinguish new attempts from old
logs. Automated Linux checks cover fixed-mode workspace permissions, first boot
and reboot with local model fixtures, download failures, workflow preservation,
and a real Jupyter Server startup. They do not reproduce RunPod's actual mounted
filesystem or run H3 generation on a GPU.

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

### MiniMax H3 Fast

Open **`minimax_h3_fast_workflow`** in the Workflows sidebar for the simpler,
four-step version. Upload your portrait and voiceover, test one chunk, then queue
the remaining chunks as above. It keeps the original audio, portrait anchors,
optional motion carry, resume and automatic stitching.

Fast uses the dedicated **FL2V Turbo 4-step 768p LoRA**, strength 1, Euler sampler,
simple scheduler, denoise 1, and video/audio sigma shifts **6/3**, following the
[LoRA authors' settings](https://github.com/ModelTC/Minimax-H3-Turbo).
Setup automatically downloads this additional 2 GB file; it reuses the same
diffusion model, text encoder and VAEs. It does not need the Ref2VA models.

The graph has **27 nodes instead of 36**, with model and sampling nodes collapsed
below the main controls. Double-click a collapsed node to inspect it. The speed
change comes from four sampling steps instead of eight; fewer boxes on screen
do not reduce model work. Model loading, Qwen text/image encoding, VAE work and
stitching still take time. Total render speed has not been benchmarked against LTX.

The default canvas stays **768 x 768** to match the standard avatar workflow's
workload. The LoRA was trained at 1344 x 768; that landscape size is also available
through Width/Height, with more pixels to render. Four-step lip sync, identity and
motion continuity need GPU testing, and quality may differ from the standard
eight-step workflow. Compare one chunk before a full render.

Fast writes `output/<audio name>_fast.mp4` and stores chunks in
`output/h3_longform/<audio name>_fast/`, so it cannot reuse standard-workflow chunks
by accident. Change `name_suffix` on the output-name node to start a fresh session
when changing inputs or settings (for example `_fast_take2`).

**Existing pod:** restart it using the same start command. Setup updates the node
pack, downloads the new LoRA if missing and installs the new Fast workflow without
overwriting your saved standard workflow. Refresh ComfyUI after startup completes.
For a manual update, install both the updated node pack and the Fast JSON, and
download the LoRA; importing just the graph into an old installation is insufficient.

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

ComfyUI runs from its own virtualenv (`$COMFYUI_PATH/.venv-cu128`). If you install
anything by hand, use that interpreter:

```bash
/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python -m pip install <package>
```

## Repository layout

```
├── setup.sh                               # RunPod boot script
├── README.md
├── TEMPLATE_README.md                     # paste into the public template's README tab
├── custom_nodes/
│   └── comfyui-h3-longform/               # Split / Carry / Write + Stitch / Name From Audio File
├── tools/
│   ├── build_workflow.py                  # generates both workflow JSONs; re-run after edits
│   ├── test_fast_workflow.py              # Fast graph and automatic model download checks
│   ├── test_longform.py                   # CPU + ffmpeg regression checks
│   ├── test_storage.py                    # Network Volume mount and filesystem checks
│   ├── test_boot.py                       # full setup with image/network fixtures
│   ├── test_jupyter.py                    # auth patch and real Jupyter Server smoke test
│   └── test_setup.py                      # Bash download checks using local fixtures
└── workflows/
    ├── minimax_h3_long_video_workflow.json
    └── minimax_h3_fast_workflow.json
```

## References

- ComfyUI guide: https://docs.comfy.org/tutorials/video/minimax/minimax-h3
- Weights (ComfyUI format): https://huggingface.co/Comfy-Org/MiniMax-H3
- Original release, licence and prompt guides: https://huggingface.co/MiniMaxAI/MiniMax-H3
- Node source: `comfy_extras/nodes_minimax_h3.py` in ComfyUI
