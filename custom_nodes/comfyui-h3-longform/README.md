# H3 Longform

Five nodes that turn one portrait and a long voiceover into a talking-head video with
MiniMax H3, using ComfyUI's batch queue as the loop. They're ported from the LTX
Longform pack. No dependencies beyond ffmpeg.

ComfyUI runs each node once per queue item, so a single node can't loop over chunks.
Instead, each queue item renders one chunk, and these nodes carry the plan from one
item to the next.

| Node | Does |
|---|---|
| **H3 Longform: Split Audio Chunk** | Plans the whole track on H3's frame grid, and outputs this chunk's audio, `length` (frames to generate), `keep_frames` (frames to write), `total_chunks`, `is_last` and an `alignment` prompt line |
| **H3 Longform: Opening (portrait or motion carry)** | Returns the portrait on chunk 0 or with carry off; otherwise loads the previous chunk's tail as a guide clip |
| **H3 Longform: Write + Stitch** | Writes the chunk as an mp4. On the last chunk it concatenates them all and muxes in the original track |
| **H3 Longform: Name From Audio File** | Names the session and output after the Load Audio file |
| **H3 Longform: Talking-Head Prompt** | Writes each chunk's full prompt in MiniMax's format from short `subject`, `background` and `delivery` descriptions, with the alignment line for that chunk's length and carry |

The name node's optional `name_suffix` is appended to both names. The bundled
Fast and Draft workflows set `_fast` and `_draft`, keeping their chunks separate
from the standard workflow's. Change it when starting a new take with
different inputs or settings. Existing workflows default to an empty suffix.

## Wiring (already done in the bundled workflows)

- Split `audio_chunk` → Write `chunk_audio`
- Split `guide_audio` → `MiniMaxH3AddGuide.audio` (frame_idx 0), including the carry lead-in
- Split `carry_frames` → Carry `carry_frames` and Write `trim_start`
- Split `length` → `MiniMaxH3ImageToVideo.length`
- Split `length` and `carry_frames` → Talking-Head Prompt; its `prompt` → `MiniMaxH3ImageToVideo.prompt`
  (Split's `alignment` output is kept for older graphs that prepend it to a typed prompt)
- The cropped portrait goes to Carry `portrait` and H3 `last_frame`
- Carry `first_frame` → H3 `first_frame`; Carry `carry_clip` → Add Guide `image`
- Connect the video VAE to Add Guide `vae`, and the audio VAE to `audio_vae`
- The incrementing chunk index goes to Split, Carry and Write
- `VAEDecode` → Write `images`. Also connect `keep_frames`, `total_chunks`, and the **full** Load Audio to `original_audio`

## How H3 differs from LTX

- **Frame grid.** H3 only generates 17k+5 frames, and its trained range is 124–362
  (about 5.2–15.1 s). Chunks are planned in frames on that grid rather than in whole
  seconds, because ComfyUI silently rounds the length *up* to the next valid value.
- **The dropped frame (carry off).** Every chunk but the last is generated at 17k+5 frames and
  written at 17k+4. The dropped frame is the pinned portrait, and the next chunk opens
  on that same portrait. This avoids a doubled boundary frame, and the frames
  kept track the audio slice to within sample rounding.
- **Motion carry (experimental).** Set `motion_carry` on Split to `5 frames (~0.2s)`
  or `22 frames (~0.9s)`. After chunk 0, the previous chunk's last written frames
  replace the opening portrait. They are generated again with their original audio,
  then trimmed. A non-final chunk keeps `length - carry_frames - 1` frames; the final
  chunk is padded to the frame grid and trimmed to the remaining track length.
  The portrait still anchors the end to limit identity drift, but visual quality
  and continuity need GPU testing. Default is `off`.
- **Cuts land in pauses.** Pauses are detected as intervals. `pause` mode takes the
  furthest grid position that falls *inside* a pause, and snaps to the nearest one
  when no grid point fits.
- **Stereo.** H3's audio VAE encodes exactly two channels, so mono voiceovers are
  upmixed before the guide. The final mux encodes your original track as AAC.
- **No `-shortest`.** The AAC stream comes out a few ms short of the WAV, and
  `-shortest` then drops a video frame. Losing one frame per chunk makes the picture
  drift progressively ahead of the voice, so neither mux uses it.

## Running

1. Turn Split's `auto_chunk` on (the bundled workflows do) and wire its
   `chunk_index` **output** to Carry and Write. Each queue item then renders the
   session's first unfinished chunk, found on disk. A browser-side counter keeps
   climbing through surplus items and never learns that a render finished, so a
   new voiceover would otherwise start at chunk 2001.
2. Queue with a batch count **at least** `total_chunks`. Surplus items are blocked
   and cost milliseconds.
3. The last chunk stitches everything and, with `stop_when_done` on, clears the
   pending queue. Note that this clears **all** pending items, not just this render's.

**Resume:** queue again. A chunk counts as finished when its mp4 exists and, with
carry, so does its tail; missing tails cause those chunks to render again. The final
chunk is redone until a stitch succeeds, which writes a `.finished` marker. After
that the session is skipped; writing any chunk again removes the marker, so the
next item re-stitches. Tails are retained after stitching for retries.

With `auto_chunk` off (graphs made before it existed), set `chunk_index` to
increment, and back to 0 before each render or resume.

**Changing settings:** use a fresh session when changing carry mode, chunk settings,
audio, portrait, or resolution. Unlink Write's session input to type a new name.
Resume checks file presence, not whether the settings match the earlier render.

**Several pods, one render (carry off only, `auto_chunk` off):** chunks are independent, since every
chunk starts and ends on the portrait. Pods sharing a network volume can take a range of indices
in the same session. A chunk counts as done when its mp4 exists, and each is written
under a temp name and then renamed. Run the last chunk once all the others are on
disk; if any are missing it refuses and names them.

With carry enabled, chunks depend on their predecessors: queue them sequentially.
A missing or too-short tail stops the render with instructions to resume from 0.

Intermediate files live in `output/h3_longform/<session>/`.

## Persistent Network Volume

The template stores ComfyUI and all render files on the Network Volume at
`/workspace`. Chunk videos and carry tails remain in the session folder for resume.
When queued through ComfyUI's UI, the graph is saved alongside them as
`workflow.json`. Final videos stay in ComfyUI's output directory.

Attach the same Network Volume to a replacement pod and queue from chunk 0 with
the same settings. Completed chunks are reused directly from that volume.
No remote backup service or second storage volume is required.

Run local CPU checks with `python tools/test_longform.py` from the template root
(requires torch, numpy, ffmpeg and ffprobe). They use synthetic frames to verify
planning, audio overlap, carry loading, trimming, resume, workflow wiring, and mux
timing without downloading model weights.
Run `python tools/test_storage.py` for Network Volume mount and filesystem checks.
