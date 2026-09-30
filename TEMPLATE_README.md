# MiniMax H3 - Long Talking Avatars

Turn one portrait and a voiceover into a talking-avatar video in ComfyUI. Audio is rendered in chunks, then stitched with your original audio.

## Storage

Attach your own **Network Volume at /workspace** when deploying. **100 GB recommended**; large renders need more space.

The public template has a **5 GB container disk** and **0 GB Volume disk**. You must attach a Network Volume before deployment. No Global Storage is needed.

ComfyUI, its Python environment, models, uploads, workflows and render progress stay on the Network Volume. Save Jupyter notebooks under /workspace. Reattach the same volume in its datacenter when replacing a pod.

## Connect

- **Port 8188:** ComfyUI
- **Port 8888:** JupyterLab

First boot downloads **42 GB** of weights. Wait for "Setup complete!" in logs. Complete models are reused on later boots.

Jupyter has no login by default. Anyone with its accessible URL can run commands. For login protection, set JUPYTER_NO_AUTH=0 and set JUPYTER_PASSWORD when deploying.

## Generate a video

1. Open ComfyUI's Workflows sidebar.
2. Select **minimax_h3_long_video_workflow**.
3. Load your portrait and voiceover.
4. Test one chunk with batch count 1.
5. Reset chunk_index to 0, then queue enough items for the audio: roughly audio seconds divided by 12, plus a margin.

The last chunk stitches the final video into ComfyUI/output. To resume, keep the same settings, reset chunk_index to 0 and queue again.

Optional motion carry uses the previous chunk's last **5 or 22 frames** to guide the next chunk. Default: off. Use a fresh session when changing inputs or settings.

## Hardware and licence

Start testing with **80 GB or more VRAM**. GPU memory use and quality have not been validated on a live H3 pod.

The H3 licence excludes the **US, EU, UK and South Korea** unless separately licensed. Check the licence and choose an eligible datacenter.

Full instructions and licence links:
https://github.com/tenitsky/minimax-h3
