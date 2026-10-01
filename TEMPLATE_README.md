# MiniMax H3 - Long Talking Avatars

Turn a portrait and voiceover into a talking-avatar video stitched with your original audio.

## Storage

Attach your own **Network Volume at /workspace** when deploying (**100 GB recommended**). The container disk is 5 GB and the Volume disk 0 GB.

ComfyUI, models, uploads, workflows and renders stay on the Network Volume. Reattach the same volume in its datacenter when replacing a pod.

## Connect

- **Port 8188:** ComfyUI
- **Port 8888:** JupyterLab

First boot downloads **46 GB**. Wait for "Setup complete!" in logs. Models are reused on later boots.

Jupyter has no login by default; anyone with its URL can run commands. For a login, set JUPYTER_NO_AUTH=0 and JUPYTER_PASSWORD.

## Generate a video

1. Open ComfyUI's Workflows sidebar.
2. Select **minimax_h3_talking_head_fast** (768, 4 steps), **minimax_h3_talking_head_draft** (544, 4 steps) or **minimax_h3_talking_head** (768, 8 steps).
3. Load your portrait and voiceover.
4. In **Talking-Head Prompt**, describe the person and background seen in the portrait. This keeps the background steady.
5. Test one chunk at batch count 1.
6. Queue roughly audio seconds divided by 8, plus a margin. Each item renders the next unfinished chunk.

The final video lands in ComfyUI/output. Next video: load its voiceover and queue. To resume, queue again.

**Batch:** the **_batch** workflows render every voiceover in ComfyUI/input/batch_audio/, one video per file, then stop.

## Hardware and licence

Requires **CUDA 13**: when deploying, set Additional filters → CUDA Versions to **13.0**. Start testing with **80 GB or more VRAM**. GPU memory use and quality have not been validated on a live H3 pod.

The H3 licence excludes the **US, EU, UK and South Korea** unless separately licensed. Check the licence and choose an eligible datacenter.

Full instructions and licence links:
https://github.com/tenitsky/minimax-h3
