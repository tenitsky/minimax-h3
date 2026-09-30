# MiniMax H3 - Long Talking Avatars

Turn a portrait and voiceover into a talking-avatar video stitched with your original audio.

## Storage

Attach your own **Network Volume at /workspace** when deploying. **100 GB recommended**; large renders need more space.

The public template has a **5 GB container disk** and **0 GB Volume disk**. Attach a Network Volume before deployment.

ComfyUI, its Python environment, models, uploads, workflows and render progress stay on the Network Volume. Save Jupyter notebooks under /workspace. Reattach the same volume in its datacenter when replacing a pod.

## Connect

- **Port 8188:** ComfyUI
- **Port 8888:** JupyterLab

First boot downloads **44 GB**. Wait for "Setup complete!" in logs. Models are reused on later boots.

Jupyter has no login by default. Anyone with its accessible URL can run commands. For login protection, set JUPYTER_NO_AUTH=0 and set JUPYTER_PASSWORD when deploying.

## Generate a video

1. Open ComfyUI's Workflows sidebar.
2. Select **minimax_h3_fast_draft_workflow**: 544 x 544, 4 steps.
3. Load your portrait and voiceover.
4. Test one chunk at batch count 1.
5. Reset chunk_index to 0, then queue enough items for the audio: roughly audio seconds divided by 12, plus a margin.

The last chunk stitches the final video into ComfyUI/output. To resume, keep the same settings, reset chunk_index to 0 and queue again.

Fast Draft trades detail for speed, with separate `_fast_draft` outputs. 768p Fast and standard workflows are included. Speed and quality need GPU testing. Motion carry: **5 or 22 frames**, default off. Use a new session when changing inputs or settings.

## Hardware and licence

Start testing with **80 GB or more VRAM**. GPU memory use and quality have not been validated on a live H3 pod.

The H3 licence excludes the **US, EU, UK and South Korea** unless separately licensed. Check the licence and choose an eligible datacenter.

Full instructions and licence links:
https://github.com/tenitsky/minimax-h3
