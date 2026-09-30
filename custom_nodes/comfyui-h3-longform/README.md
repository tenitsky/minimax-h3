# H3 Longform

Four nodes that turn one portrait and a long voiceover into a talking-head video with
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

## Wiring (already done in `minimax_h3_long_video_workflow`)

- Split `audio_chunk` → Write `chunk_audio`
- Split `guide_audio` → `MiniMaxH3AddGuide.audio` (frame_idx 0), including the carry lead-in
- Split `carry_frames` → Carry `carry_frames` and Write `trim_start`
- Split `length` → `MiniMaxH3ImageToVideo.length`
- Split `alignment` + your prompt → `MiniMaxH3ImageToVideo.prompt`
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

1. Leave `chunk_index` on **increment**, starting at 0.
2. Queue with a batch count **at least** `total_chunks`. Surplus items are blocked
   and cost milliseconds.
3. The last chunk stitches everything and, with `stop_when_done` on, clears the
   pending queue. Note that this clears **all** pending items, not just this render's.

**Resume:** set `chunk_index` to 0 and queue again. With `skip_existing` on, finished
chunks are skipped if their carry files also exist when carry is enabled. Missing
tails cause those chunks to render again. The final chunk always re-renders, because
it's what triggers the stitch. Tails are retained after stitching for retries.

**Changing settings:** use a fresh session when changing carry mode, chunk settings,
audio, portrait, or resolution. Unlink Write's session input to type a new name.
Resume checks file presence, not whether the settings match the earlier render.

**Several pods, one render (carry off only):** chunks are independent, since every
chunk starts and ends on the portrait. Pods sharing a network volume can take a range of indices
in the same session. A chunk counts as done when its mp4 exists, and each is written
under a temp name and then renamed. Run the last chunk once all the others are on
disk; if any are missing it refuses and names them.

With carry enabled, chunks depend on their predecessors: queue them sequentially.
A missing or too-short tail stops the render with instructions to resume from 0.

Intermediate files live in `output/h3_longform/<session>/`.

## Global Volume backups

When setup detects `/workspace-global`, encoding and stitching still run on the
persistent regional Network Volume at `/workspace`. After each completed chunk,
Write copies its carry tail and video to
`/workspace-global/minimax-h3/output/` and verifies their checksums. A failed copy
raises an error; resume from 0 retries backups for locally completed chunks too.
Write only announces `FINISHED` after the final video's backup has verified.

Setup on a replacement pod restores completed backups automatically. Partial or
corrupt copies are ignored. Inputs live directly on the Global Volume. Sidebar
workflows are edited locally (ComfyUI requires atomic rename) and backed up after
successful UI saves; the current UI workflow is also captured with each rendered chunk. Queue
from 0 to resume with the same settings. Use only one writing pod per global
namespace; cross-pod concurrent writes are not supported in Global mode.

Run local CPU checks with `python tools/test_longform.py` from the template root
(requires torch, numpy, ffmpeg and ffprobe). They use synthetic frames to verify
planning, audio overlap, carry loading, trimming, resume, workflow wiring, and mux
timing without downloading model weights.
Run `python tools/test_storage.py` for standard-library-only persistence tests.
