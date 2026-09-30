# MiniMax H3 - ComfyUI + JupyterLab

Generate long talking-avatar videos from one portrait and a voiceover, with optional
motion carry between chunks.

## Storage

Attach **one Network Volume at `/workspace`**. Start with **100 GB**; large renders
and optional models may need more. Keep the container disk at **5 GB**.
The public template's 0 GB Volume disk default expects you to attach your own
Network Volume when deploying; it does not provide storage by itself.

ComfyUI, its Python environment, models, inputs, workflows, Jupyter files, and render
progress stay on the Network Volume. Save notebooks under `/workspace`.
No Global Volume or storage environment variables are needed.

For a replacement pod, reattach the same Network Volume in its datacenter.
Set the workflow's chunk index to 0 to resume with the same render settings.

## Connect

- **8188:** ComfyUI
- **8888:** JupyterLab

First startup downloads about 42 GB of model weights. Wait for setup to finish.
Jupyter has no login by default; anyone with its accessible URL can run commands.

In ComfyUI, open the Workflows sidebar and select
`minimax_h3_long_video_workflow`. Load your portrait and voiceover, test one chunk,
then queue enough items to render the whole track.

GPU usage and visual quality have not yet been validated on a live H3 pod.
Start initial testing with 80 GB or more VRAM.

## Model licence

The MiniMax H3 licence excludes use in the US, EU, UK, and South Korea unless
separately licensed. Check the licence and choose an eligible datacenter before use.

Full instructions and licence links:
https://github.com/tenitsky/minimax-h3
