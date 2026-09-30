"""
H3 Longform - turn one portrait + a long voiceover into a minutes-long talking-head
video with MiniMax H3, using ComfyUI's own batch queue as the loop.

Same design as the LTX Longform pack this was ported from: ComfyUI's graph is a DAG,
so each queue item renders one chunk and these nodes carry the plan between them.

    Split  -> this chunk's audio, generation length and keep length
    Carry  -> the chunk's opening: the portrait, or the previous chunk's tail
    Write  -> saves the chunk, and on the final one stitches everything together

What is different for H3:

  * Frame grid. H3 only generates 17k+5 frames (124, 141, ... 362 in its trained
    range, i.e. ~5.2s to ~15.1s at 24fps). Whole-second chunks - what the LTX pack
    planned in - almost never land on that grid, and the node silently rounds the
    length *up*. So chunks are planned in frames, on the grid.
  * Last frame pinned to the portrait. Every chunk ends on the original portrait, so
    whatever drift a chunk picks up is gone by its end and cannot accumulate.
  * Opening, two ways (motion_carry on Split):
      off - the portrait is also the first frame. Chunk N ends on the frame chunk N+1
            starts on; each non-final chunk is generated at 17k+5 frames and written
            at 17k+4, dropping the pinned portrait the next chunk opens with.
      5 / 22 frames - chunk N+1 opens on chunk N's last 5 or 22 written frames,
            pinned as a clip, so head motion carries across the join instead of
            restarting from the portrait pose. Those lead-in frames are generated
            again and trimmed, so nothing plays twice. Identity still resets at every
            chunk's end, because the last frame stays pinned to the portrait.
  * Audio guide. The chunk's slice of the voiceover (plus the lead-in, with carry) is
    fed to MiniMaxH3AddGuide at frame 0. H3's audio VAE is stereo-only, so mono
    tracks are upmixed here.

The final mux always uses the untouched original track, never the per-chunk VAE
reconstructions, so sync is exact by construction.

No dependencies beyond what ComfyUI already has, plus ffmpeg.
"""

import json
import os
import shutil
import subprocess

import numpy as np
import torch

import folder_paths

try:
    # Returning this from a node blocks everything downstream, so an
    # out-of-range chunk finishes in milliseconds instead of sampling.
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:  # pragma: no cover - older ComfyUI
    try:
        from comfy_execution.graph import ExecutionBlocker
    except ImportError:
        ExecutionBlocker = None

FPS = 24                       # H3 is trained at 24fps only
GRID = 17                      # frames per latent block
GRID_OFFSET = 5                # valid lengths are GRID*k + GRID_OFFSET
GEN_MIN, GEN_MAX = 124, 362    # trained range, per ComfyUI's node tooltips
SESSION_ROOT = "h3_longform"
LOG = "[H3 Longform]"
# Guide clips must themselves be 17k+5 frames long, so these are the two useful sizes.
CARRY_OPTIONS = {"off": 0, "5 frames (~0.2s)": 5, "22 frames (~0.9s)": 22}


# --------------------------------------------------------------------------- helpers
def align_up(n):
    """Smallest valid H3 length >= n. Mirrors ComfyUI's align_frame_count."""
    n = max(GRID_OFFSET, int(n))
    while n % GRID != GRID_OFFSET:
        n += 1
    return n


def keep_grid(min_seconds, max_seconds, carry=0):
    """Frame counts a non-final chunk may keep.

    A chunk is generated at a valid length n and keeps n - 1 - carry: the pinned
    portrait at the end is dropped, and so is the lead-in with motion carry. Bounds
    are matched to the nearest grid length (half a block, ~0.35s, either way) so e.g.
    max_seconds=15 still reaches the 362-frame ceiling.
    """
    half = GRID // 2
    lens = [n for n in range(GEN_MIN, GEN_MAX + 1, GRID)
            if min_seconds * FPS - half <= n <= max_seconds * FPS + half]
    if not lens:
        # Bounds outside the trained range: use the closest end of it.
        lens = [GEN_MIN] if max_seconds * FPS < GEN_MIN else [GEN_MAX]
    return [n - 1 - carry for n in lens]


def _session_dir(session):
    d = os.path.join(folder_paths.get_output_directory(), SESSION_ROOT, session)
    os.makedirs(d, exist_ok=True)
    return d


def _mono(waveform):
    """[B,C,T] -> 1-D numpy, averaged across batch and channels."""
    w = waveform
    if w.dim() == 3:
        w = w[0]
    if w.dim() == 2:
        w = w.mean(dim=0)
    return w.detach().cpu().float().numpy()


def _stereo(waveform):
    """H3's audio VAE encodes exactly two channels; anything else breaks packing."""
    c = waveform.shape[-2]
    if c == 2:
        return waveform
    if c == 1:
        return waveform.repeat_interleave(2, dim=-2)
    return waveform[..., :2, :]


def find_pauses(samples, sr, thresh_db=-34.0, min_len=0.25, win=0.02):
    """Quiet stretches as (start, end) in seconds, computed from the waveform.

    In-memory rather than ffmpeg silencedetect so the plan is deterministic and
    identical on every queue item - each chunk recomputes it independently.
    Intervals, not midpoints: a cut has to snap to the frame grid, and it should
    still land inside the pause after snapping.
    """
    n = max(1, int(sr * win))
    usable = (len(samples) // n) * n
    if usable < n:
        return []
    frames = samples[:usable].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    peak = float(rms.max())
    if peak <= 0:
        return []
    quiet = 20 * np.log10(rms / peak + 1e-12) < thresh_db

    pauses, run = [], 0
    for i, q in enumerate(quiet):
        if q:
            run += 1
            continue
        if run * win >= min_len:
            pauses.append(((i - run) * win, i * win))
        run = 0
    if run * win >= min_len:
        pauses.append(((len(quiet) - run) * win, len(quiet) * win))
    return pauses


def plan_chunks(total, target, keeps, pauses, mode="pause", keeps_first=None):
    """Split [0, total) frames into (start_frame, keep_frames) spans.

    Every span but the last keeps a length from its grid - `keeps_first` for chunk 0,
    `keeps` for the rest (they differ with motion carry, since chunk 0 has no lead-in).
    The last span keeps whatever is left and is generated at the next valid length
    up, then trimmed.

    `pauses` are (start, end) frame intervals. Boundary choice:
      * pause   - the furthest grid position inside a pause; speech sets the lengths
      * silence - the grid position inside a pause closest to `target`
      * fixed   - the grid length closest to `target`, pauses ignored
    With no grid position inside any pause, the nearest one to a pause midpoint in
    reach is used, and failing that the clock (max length for pause, target else).
    """
    keeps_first = keeps_first or keeps
    spans, pos = [], 0
    while True:
        grid_k = keeps_first if not spans else keeps
        kmin, kmax = min(grid_k), max(grid_k)
        if total - pos <= kmax:
            break
        grid = [pos + k for k in grid_k]
        if mode == "fixed":
            cut = min(grid, key=lambda g: abs(g - (pos + target)))
        else:
            inside = [g for g in grid if any(a <= g <= b for a, b in pauses)]
            if inside:
                cut = max(inside) if mode == "pause" else \
                    min(inside, key=lambda g: abs(g - (pos + target)))
            else:
                reach = GRID // 2
                mids = [(a + b) / 2 for a, b in pauses
                        if pos + kmin - reach <= (a + b) / 2 <= pos + kmax + reach]
                if mids:
                    m = max(mids) if mode == "pause" else \
                        min(mids, key=lambda p: abs(p - (pos + target)))
                else:
                    m = pos + (kmax if mode == "pause" else target)
                cut = min(grid, key=lambda g: abs(g - m))
        spans.append((pos, cut - pos))
        pos = cut

    if total - pos >= 1:
        spans.append((pos, total - pos))

    # A stubby tail generates poorly and reads as an offcut: fold it into the
    # previous chunk, or rebalance the last two so both land in range.
    if len(spans) >= 2 and spans[-1][1] < min(keeps):
        prev_grid = keeps_first if len(spans) == 2 else keeps
        (p_start, p_keep), (_, l_keep) = spans[-2], spans[-1]
        combined = p_keep + l_keep
        if combined <= max(prev_grid):
            spans[-2:] = [(p_start, combined)]
        else:
            fits = [k for k in prev_grid if min(keeps) <= combined - k <= max(keeps)]
            if fits:
                k = min(fits, key=lambda k: abs(k - combined / 2))
                spans[-2:] = [(p_start, k), (p_start + k, combined - k)]
    return spans


def gen_length(keep, is_last, carry=0):
    """Frames to generate for a chunk that keeps `keep` frames after a `carry` lead-in."""
    if is_last:
        return max(GEN_MIN, align_up(carry + keep))
    return carry + keep + 1


def alignment_line(length, carry):
    """The first prompt line, verbatim from MiniMax's prompt guide.

    The text encoder labels the pictures it is shown "<Picture 1>", "<Picture 2>" in
    connection order. Without carry both portraits are shown, so this is the FL2VA
    instruction. With carry the first frame is a guide clip the encoder never sees,
    only the last portrait, which is the guide's L2VA instruction.
    """
    end_mark = f"{(length - 1) / FPS:.2f}"
    if carry:
        return ("How the reference pictures align with the target video — <Picture 1> "
                f"(from [Shot 1]) aligns with the {end_mark}-second mark of the target video.")
    return ("How the reference pictures align with the target video — Picture 1 (from "
            "Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from "
            f"Shot 1) aligns with the {end_mark}-second mark of the target video.")


DEFAULT_SUBJECT = "the person shown in the reference picture"
DEFAULT_BACKGROUND = "the background shown in the reference picture"
DEFAULT_DELIVERY = "a clear, calm, natural voice"


def _field(text, name):
    """A description with whitespace collapsed and an optional 'name:' tag removed.

    Multiline widgets hide their label once they hold text, so the bundled
    workflows start each value with its field name ("subject: a woman ...").
    """
    text = " ".join((text or "").split())
    if text.lower().startswith(name + ":"):
        text = text[len(name) + 1:].strip()
    return text


def talking_head_prompt(length, carry, subject="", background="", delivery="", extra=""):
    """A complete single-shot H3 prompt for a locked-off talking head.

    Follows the guide's structure: instruction line, then exactly three fields. The
    body names what stays fixed in concrete terms - the model drifts toward whatever
    the text leaves open, and a bare "presenter" pulls in broadcast graphics.
    """
    subject = _field(subject, "subject") or DEFAULT_SUBJECT
    background = _field(background, "background") or DEFAULT_BACKGROUND
    delivery = _field(delivery, "delivery") or DEFAULT_DELIVERY
    if carry:
        start = (f"[Shot 1] Live-action, a static shot continuing seamlessly from the provided "
                 f"opening frames: {subject} is already mid-sentence, with the same framing, "
                 "lighting and background as those frames.")
        end = ("Toward the end of the shot the person settles into the pose, expression and "
               "composition established by <Picture 1>.")
    else:
        start = (f"[Shot 1] Live-action, a static shot of {subject}, beginning in the exact "
                 "pose, framing, lighting and background established by Picture 1.")
        end = ("Toward the end of the shot the person settles back into the pose, expression "
               "and composition established by Picture 2.")
    body = [
        start,
        "The camera holds a static shot on a locked-off tripod for the entire shot, with no "
        "change in framing, focus or exposure.",
        f"The person (S1) speaks directly to the camera in {delivery}, lips moving in exact "
        "sync with the speech, with natural blinking, small eyebrow movements and gentle head "
        "nods, while the shoulders and body stay in place.",
        f"Behind the person, {background} stays completely unchanged throughout: the same "
        "objects in the same places, the same colours and the same lighting.",
        "The frame contains only this one person in front of this background from the first "
        "frame to the last; no other person, object, text, caption, logo, watermark or "
        "on-screen graphic appears at any point.",
    ]
    extra = _field(extra, "extra")
    if extra:
        body.append(extra)
    body.append(end)
    return (alignment_line(length, carry) + "\n\n"
            + "integrated_multimodal_description: " + " ".join(body) + "\n\n"
            + "overall_soundscape: A quiet, steady room tone sits under the voice, with no "
              "other sounds.\n\n"
            + "non_diegetic_music: N/A")


def _prompt_nodes(prompt, class_type):
    if not isinstance(prompt, dict):
        return
    for node in prompt.values():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            yield node


def _carry_setting(prompt):
    """Lead-in length chosen on the Split node, read from the prompt.

    Write needs it to decide whether to keep a tail for the next chunk, including on
    chunk 0, which has no lead-in of its own. If it cannot be read, assume the
    largest so a needed tail is never missing.
    """
    if not isinstance(prompt, dict):
        return max(CARRY_OPTIONS.values())
    for node in _prompt_nodes(prompt, "H3LongformSplit"):
        v = (node.get("inputs") or {}).get("motion_carry", "off")
        if isinstance(v, str):
            return CARRY_OPTIONS.get(v, 0)
        return max(CARRY_OPTIONS.values())
    return 0


def _audio_basename(prompt, namer_id=None, default="run1", source_title=None):
    """The LoadAudio selection, without directory or extension."""
    if not isinstance(prompt, dict):
        return default
    want = source_title.strip() if isinstance(source_title, str) else ""
    if source_title is None and namer_id is not None:
        n = prompt.get(str(namer_id)) or {}
        want = ((n.get("inputs") or {}).get("source_title") or "").strip()
    for node in _prompt_nodes(prompt, "LoadAudio"):
        if want and ((node.get("_meta") or {}).get("title") or "") != want:
            continue
        v = (node.get("inputs") or {}).get("audio")
        if isinstance(v, str) and v.strip():
            base = os.path.basename(v.strip().replace("\\", "/"))
            stem = os.path.splitext(base)[0].strip()
            if stem:
                return stem
    return default


def _session_from_prompt(prompt, default="run1"):
    """Read the session name off the Write node, so it is set in one place.

    Split and Carry need it but run before Write, so they cannot take the value as a
    link without creating a cycle.
    """
    for node in _prompt_nodes(prompt, "H3LongformWrite"):
        v = (node.get("inputs") or {}).get("session")
        if isinstance(v, str) and v.strip():
            return v.strip()
        # A widget converted to an input arrives as [node_id, slot]. Follow it when
        # it comes from the namer, otherwise Split would check the wrong folder.
        if isinstance(v, list) and len(v) == 2:
            src = prompt.get(str(v[0])) or {}
            if src.get("class_type") == "H3LongformAudioName":
                name_suffix = (src.get("inputs") or {}).get("name_suffix", "")
                return _audio_basename(prompt, v[0], default) + name_suffix
    return default


def _tail_path(session, chunk_index):
    """Last written frames of a chunk, kept for the next chunk's lead-in."""
    return os.path.join(_session_dir(session), ".carry", f"tail_{chunk_index:04d}.npz")


def _read_tail(session, chunk_index, carry_frames):
    try:
        with np.load(_tail_path(session, chunk_index), allow_pickle=False) as saved:
            tail = saved["frames"]
        if (tail.ndim == 4 and tail.shape[0] >= carry_frames
                and tail.shape[-1] == 3 and tail.dtype == np.uint8):
            return tail[-carry_frames:]
    except (OSError, ValueError, KeyError, EOFError):
        pass
    return None


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg not found. On the RunPod template it is installed by setup.sh; "
            "otherwise: apt-get install -y ffmpeg")
    return exe


def _write_wav(audio, path, ff):
    wav = audio["waveform"]
    if wav.dim() == 3:
        wav = wav[0]
    arr = wav.detach().cpu().float().numpy().T  # [samples, channels]
    pcm = (arr.clip(-1, 1) * 32767).astype(np.int16)
    p = subprocess.Popen(
        [ff, "-y", "-v", "error", "-f", "s16le", "-ar",
         str(int(audio["sample_rate"])), "-ac", str(pcm.shape[1]),
         "-i", "-", path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = p.communicate(pcm.tobytes())
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing audio:\n{err.decode()[-1500:]}")


# ----------------------------------------------------------------------------- nodes
class H3LongformSplit:
    """Slice one chunk out of a long track, on H3's frame grid."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 100000,
                                        "tooltip": "Set this to increment, then queue "
                                                   "with a batch count of at least "
                                                   "total_chunks."}),
                "target_seconds": ("FLOAT", {"default": 12.0, "min": 5.0, "max": 15.0,
                                             "step": 0.5,
                                             "tooltip": "Only used by cut_mode silence "
                                                        "and fixed."}),
                "min_seconds": ("FLOAT", {"default": 5.0, "min": 5.0, "max": 15.0,
                                          "step": 0.5}),
                "max_seconds": ("FLOAT", {"default": 15.0, "min": 5.0, "max": 15.0,
                                          "step": 0.5,
                                          "tooltip": "H3's trained ceiling is 362 "
                                                     "frames (~15.1s), lead-in "
                                                     "included. Lower it if a chunk "
                                                     "runs out of VRAM."}),
                "cut_mode": (["pause", "silence", "fixed"], {
                    "default": "pause",
                    "tooltip": "pause: the speech decides - take the furthest pause "
                               "within max_seconds. silence: aim for target_seconds, "
                               "snapping to a pause in range. fixed: a strict grid "
                               "near target_seconds."}),
                "skip_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip chunks whose .mp4 is already on disk. Resume an "
                               "interrupted render by queueing from 0 again. The "
                               "final chunk always re-renders so the stitch fires."}),
                "motion_carry": (list(CARRY_OPTIONS), {
                    "default": "off",
                    "tooltip": "off: every chunk starts on the portrait (no drift, "
                               "but the pose resets at each join). 5/22 frames: each "
                               "chunk opens on the previous chunk's last frames so "
                               "motion continues across the join; the last frame "
                               "stays pinned to the portrait, so identity still "
                               "resets every chunk. Change it only between renders - "
                               "it changes where every chunk starts."}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("AUDIO", "INT", "INT", "INT", "BOOLEAN", "STRING", "AUDIO", "INT")
    RETURN_NAMES = ("audio_chunk", "length", "keep_frames", "total_chunks", "is_last",
                    "alignment", "guide_audio", "carry_frames")
    FUNCTION = "split"
    CATEGORY = "H3 Longform"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Resume decisions depend on files written by earlier queue items, and on
        # the session stored in the hidden prompt, not only on visible inputs.
        return float("nan")

    def split(self, audio, chunk_index, target_seconds, min_seconds, max_seconds,
              cut_mode="pause", skip_existing=True, motion_carry="off", prompt=None):
        if min_seconds > max_seconds:
            min_seconds, max_seconds = max_seconds, min_seconds
        carry_setting = CARRY_OPTIONS.get(motion_carry, 0)
        keeps_first = keep_grid(min_seconds, max_seconds)
        keeps = keep_grid(min_seconds, max_seconds, carry_setting)
        target = round(max(min_seconds, min(max_seconds, target_seconds)) * FPS)

        wav, sr = audio["waveform"], int(audio["sample_rate"])
        total = int(round(wav.shape[-1] * FPS / sr))
        pauses = []
        if cut_mode != "fixed":
            pauses = [(a * FPS, b * FPS) for a, b in find_pauses(_mono(wav), sr)]
        if chunk_index == 0:
            # Pause detection is thresholded against the track's peak, so music or
            # room tone under the voice can leave it finding nothing.
            print(f"{LOG} cut_mode={cut_mode}, motion_carry={carry_setting}, "
                  f"pauses found: {len(pauses)}")
        spans = plan_chunks(total, target, keeps, pauses, cut_mode, keeps_first)
        if not spans:
            raise RuntimeError("Audio too short to split into chunks.")

        blocked = tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES) \
            if ExecutionBlocker is not None else None
        if chunk_index >= len(spans):
            # Past the end: blocking makes a generous batch count safe.
            if blocked:
                if chunk_index == len(spans):
                    print(f"{LOG} all {len(spans)} chunks done - skipping surplus "
                          f"queue items.")
                return blocked
            print(f"{LOG} past the last chunk; stop the queue manually.")

        if skip_existing and blocked and chunk_index < len(spans) - 1:
            session = _session_from_prompt(prompt)
            done = os.path.join(_session_dir(session),
                                f"chunk_{chunk_index:04d}.mp4")
            tail_ready = not carry_setting or _read_tail(session, chunk_index, carry_setting) is not None
            if os.path.exists(done) and tail_ready:
                print(f"{LOG} chunk {chunk_index + 1}/{len(spans)} already rendered "
                      f"- skipping.")
                return blocked

        idx = min(chunk_index, len(spans) - 1)
        start, keep = spans[idx]
        is_last = idx >= len(spans) - 1
        carry = carry_setting if idx > 0 else 0
        length = gen_length(keep, is_last, carry)

        # Sample offsets come from frame offsets, so the slices tile the track
        # exactly and line up with the kept frames.
        def audio_slice(f0, f1):
            a = int(round(f0 * sr / FPS))
            b = min(wav.shape[-1], int(round(f1 * sr / FPS)))
            return {"waveform": _stereo(wav[..., a:b]), "sample_rate": sr}

        # The guide covers the lead-in too, so the replayed tail frames hear the
        # same audio they were first generated with.
        guide = audio_slice(start - carry, start + keep)
        chunk_audio = audio_slice(start, start + keep) if carry else guide

        # Kept for graphs built before H3LongformPrompt, which wire it ahead of a
        # hand-written prompt. With a lead-in there is only the last picture.
        alignment = alignment_line(length, carry)

        print(f"{LOG} chunk {idx + 1}/{len(spans)}  {start / FPS:.2f}s "
              f"+{keep / FPS:.2f}s  (generate {length} = lead-in {carry} + keep "
              f"{keep} + tail; track {total / FPS:.1f}s)")
        return (chunk_audio, length, keep, len(spans), is_last, alignment,
                guide, carry)


class H3LongformCarry:
    """The chunk's opening: the portrait, or the previous chunk's last frames.

    Outputs exactly one of the two - the other is None, which leaves that input of
    the H3 nodes unconnected in effect. A lead-in and a portrait at frame 0 would
    contradict each other.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "portrait": ("IMAGE",),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "carry_frames": ("INT", {"default": 0, "min": 0, "max": 22,
                                         "tooltip": "Wire from Split. 0 = start on "
                                                    "the portrait."}),
            },
            # Session name is taken from the Write node so it is set in one place.
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("first_frame", "carry_clip")
    FUNCTION = "pick"
    CATEGORY = "H3 Longform"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The tail file depends on the session and on files written by earlier queue
        # items. ComfyUI calls IS_CHANGED with an empty hidden prompt and without
        # linked values, so neither can be read here: always run. Loading a few
        # frames is cheap, and a cached clip from another render would be wrong.
        return float("nan")

    def pick(self, portrait, chunk_index, carry_frames, prompt=None):
        if not carry_frames:
            return (portrait, None)

        tail = _read_tail(_session_from_prompt(prompt), chunk_index - 1, carry_frames)
        if tail is not None:
            clip = torch.from_numpy(tail.astype(np.float32) / 255.0)
            print(f"{LOG} chunk {chunk_index}: opening on the last {carry_frames} "
                  f"frames of chunk {chunk_index - 1}")
            return (None, clip)

        raise RuntimeError(
            f"Motion carry needs {carry_frames} tail frames from chunk {chunk_index - 1}. "
            "Queue chunks in order starting at 0 to rebuild missing tails. Use a fresh "
            "session if you changed motion_carry.")


class H3LongformWrite:
    """Write this chunk, and on the final one stitch the whole render together.

    Takes decoded frames rather than a VIDEO so the lead-in and the pinned last frame
    can be dropped before anything is encoded.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "original_audio": ("AUDIO", {"tooltip": "The FULL track, not the "
                                                        "chunk - used for the final mux."}),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "total_chunks": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "keep_frames": ("INT", {"default": 123, "min": 1, "max": 100000}),
                "session": ("STRING", {"default": "run1",
                                       "tooltip": "Folder under output/h3_longform/. "
                                                  "Change it to start a fresh render."}),
                "filename": ("STRING", {"default": "h3_longform_final.mp4"}),
            },
            "optional": {
                # Muxed into the per-chunk file only, so one chunk can be checked for
                # sync on its own. The finished render always uses original_audio.
                "chunk_audio": ("AUDIO", {"tooltip": "Connect Split's audio_chunk "
                                                     "to make chunks playable."}),
                "stop_when_done": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "After the final chunk, clear the pending queue. This "
                               "clears ALL pending items, including unrelated jobs."}),
                "trim_start": ("INT", {"default": 0, "min": 0, "max": 100000,
                                       "tooltip": "Wire from Split's carry_frames: "
                                                  "the lead-in frames to drop."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "write"
    OUTPUT_NODE = True
    CATEGORY = "H3 Longform"

    def write(self, images, original_audio, chunk_index, total_chunks, keep_frames,
              session, filename, chunk_audio=None, stop_when_done=True, trim_start=0,
              prompt=None, extra_pnginfo=None):
        ff = _ffmpeg()
        sdir = _session_dir(session)

        # Drop the lead-in (motion carry), then the pinned final frame (non-final
        # chunks) or the overshoot past the track's end (final chunk). Either way the
        # chunk ends up exactly as long as its audio slice.
        if images.shape[0] < trim_start + keep_frames:
            raise RuntimeError(
                f"chunk {chunk_index}: got {images.shape[0]} frames but need "
                f"{trim_start} + {keep_frames}. Is Split's `length` wired to the H3 "
                f"node?")
        frames = images[trim_start:trim_start + keep_frames]

        arr = (frames.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        n, h, w, _ = arr.shape

        # Keep this chunk's tail for the next chunk's lead-in, before anything else
        # can fail. Atomic rename, like the chunk itself.
        carry = _carry_setting(prompt)
        if carry and chunk_index < total_chunks - 1:
            tail = _tail_path(session, chunk_index)
            os.makedirs(os.path.dirname(tail), exist_ok=True)
            tmp = tail[:-4] + ".writing.npz"
            np.savez_compressed(tmp, frames=arr[-carry:])
            os.replace(tmp, tail)

        chunk_mp4 = os.path.join(sdir, f"chunk_{chunk_index:04d}.mp4")
        # Temp name + atomic rename: the finished path either does not exist or is a
        # complete chunk, which is what lets its existence act as the manifest.
        chunk_tmp = os.path.join(sdir, f".writing_{chunk_index:04d}.mp4")
        cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-"]
        # Frames go in on stdin, which carries one stream, so audio lands on disk
        # briefly and is deleted once muxed.
        #
        # No -shortest, here or in the final mux: the AAC stream comes out a few ms
        # short of the WAV, and -shortest then drops a video frame. One lost frame per
        # chunk shifts everything after it, so over a long render the picture drifts
        # ahead of the voice. The frames are already cut to the audio's exact length.
        chunk_wav = None
        if chunk_audio is not None:
            chunk_wav = os.path.join(sdir, f".chunk_{chunk_index:04d}.wav")
            _write_wav(chunk_audio, chunk_wav, ff)
            cmd += ["-i", chunk_wav, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-an"]
        cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "medium",
                "-pix_fmt", "yuv420p", chunk_tmp]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, err = p.communicate(arr.tobytes())
            if p.returncode != 0:
                raise RuntimeError(f"ffmpeg failed writing chunk:\n{err.decode()[-1500:]}")
        finally:
            if chunk_wav and os.path.exists(chunk_wav):
                os.remove(chunk_wav)
        os.replace(chunk_tmp, chunk_mp4)
        workflow = (extra_pnginfo or {}).get("workflow")
        if isinstance(workflow, dict):
            workflow_path = os.path.join(sdir, "workflow.json")
            # Different chunk indices may render on separate pods sharing this volume.
            workflow_tmp = os.path.join(sdir, f".workflow_{chunk_index:04d}.json.tmp")
            with open(workflow_tmp, "w", encoding="utf-8") as stream:
                json.dump(workflow, stream, ensure_ascii=False, indent=2)
            os.replace(workflow_tmp, workflow_path)

        msg = f"chunk {chunk_index + 1}/{total_chunks} written ({n} frames)"
        print(f"{LOG} {msg}")
        if chunk_index < total_chunks - 1:
            return (msg,)

        # Final chunk: stitch. File presence is the record, so several pods can
        # write into one session folder on a network volume without a race.
        paths = [os.path.join(sdir, f"chunk_{i:04d}.mp4") for i in range(total_chunks)]
        missing = [i for i, p in enumerate(paths) if not os.path.exists(p)]
        if missing:
            # A gap would shift everything after it out of sync, so refuse.
            raise RuntimeError(
                f"Cannot stitch: chunks {missing} are not on disk yet. Queue the "
                f"missing indices in session '{session}' (or wait for the pod "
                f"rendering them), then run the last chunk again.")

        listfile = os.path.join(sdir, "concat.txt")
        with open(listfile, "w", encoding="utf-8") as f:
            for p in paths:
                # Relative, generated names avoid quoting user-supplied session
                # paths (apostrophes and Windows separators break concat syntax).
                f.write(f"file '{os.path.basename(p)}'\n")

        silent = os.path.join(sdir, "video_only.mp4")
        wav_path = os.path.join(sdir, ".original.wav")
        out = os.path.join(folder_paths.get_output_directory(), filename)
        try:
            subprocess.run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                            "-i", listfile, "-c:v", "copy", "-an", silent], check=True)
            # The untouched original track: each chunk's audio is a VAE
            # reconstruction, so the source is strictly better.
            _write_wav(original_audio, wav_path, ff)
            subprocess.run([ff, "-y", "-v", "error", "-i", silent, "-i", wav_path,
                            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                            "-c:a", "aac", "-b:a", "192k", out],
                           check=True)
        finally:
            for tmp in (wav_path, silent, listfile):
                if os.path.exists(tmp):
                    os.remove(tmp)
        # Keep tails for a resumed render or a retry of the final chunk. Removing
        # them here would make the next run skip chunks whose lead-ins are missing.

        msg = f"FINISHED: {out}"
        print(f"{LOG} {msg}")

        if stop_when_done:
            try:
                from server import PromptServer
                PromptServer.instance.prompt_queue.wipe_queue()
                print(f"{LOG} pending queue cleared.")
            except Exception as e:
                print(f"{LOG} could not clear the queue ({e}); surplus items will "
                      f"be skipped instead.")
        return (msg,)


class H3LongformPrompt:
    """Build each chunk's full prompt from a few plain descriptions.

    The alignment line changes with every chunk's length and with motion carry, and
    the body has to name the pictures the same way, so the whole prompt is generated
    here rather than typed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "length": ("INT", {"default": 124, "min": 5, "max": 3600,
                                   "tooltip": "Wire from Split's length."}),
                "carry_frames": ("INT", {"default": 0, "min": 0, "max": 22,
                                         "tooltip": "Wire from Split's carry_frames."}),
                "subject": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Who is on screen, as seen in the portrait: e.g. 'a woman in "
                               "her thirties with shoulder-length brown hair, wearing a navy "
                               "blazer over a white shirt'. Blank uses a generic description."}),
                "background": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "What is behind them, concretely: e.g. 'a plain light-grey "
                               "wall with soft even lighting'. Naming it is what keeps it "
                               "from changing. Blank uses a generic description."}),
                "delivery": ("STRING", {
                    "default": DEFAULT_DELIVERY,
                    "tooltip": "How they speak, completing 'speaks to the camera in ...'."}),
            },
            "optional": {
                "extra": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional extra sentence for the shot, e.g. 'She smiles "
                               "briefly between sentences.' Keep motion small."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "H3 Longform"

    def build(self, length, carry_frames, subject, background, delivery, extra=""):
        return (talking_head_prompt(length, carry_frames, subject, background, delivery,
                                    extra),)


class H3LongformAudioName:
    """Turn the loaded audio's filename into strings for session and output name."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "suffix": ("STRING", {"default": ".mp4",
                                      "tooltip": "Appended to the filename output "
                                                 "only, normally .mp4."}),
            },
            "optional": {
                "source_title": ("STRING", {
                    "default": "",
                    "tooltip": "Title of the Load Audio node to read, if the graph "
                               "has more than one. Blank uses the first found."}),
                "name_suffix": ("STRING", {
                    "default": "",
                    "tooltip": "Appended to both the session and output name. "
                               "Use _fast to keep fast renders separate."}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("name", "filename")
    FUNCTION = "derive"
    CATEGORY = "H3 Longform"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The name comes from the Load Audio node, which is not an input of this
        # one. ComfyUI calls IS_CHANGED with an empty hidden prompt, so the audio
        # filename cannot be read here: a key built from it never changed, the
        # cached name outlived the audio swap, and a new voiceover's chunks were
        # written into the previous render's folder. Always run; it is a string.
        return float("nan")

    def derive(self, suffix, source_title="", prompt=None, name_suffix=""):
        name = _audio_basename(prompt, source_title=source_title) + name_suffix
        print(f"{LOG} run name from audio file: {name}")
        return (name, name + suffix)


NODE_CLASS_MAPPINGS = {
    "H3LongformSplit": H3LongformSplit,
    "H3LongformCarry": H3LongformCarry,
    "H3LongformWrite": H3LongformWrite,
    "H3LongformAudioName": H3LongformAudioName,
    "H3LongformPrompt": H3LongformPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LongformSplit": "H3 Longform: Split Audio Chunk",
    "H3LongformCarry": "H3 Longform: Opening (portrait or motion carry)",
    "H3LongformWrite": "H3 Longform: Write + Stitch",
    "H3LongformAudioName": "H3 Longform: Name From Audio File",
    "H3LongformPrompt": "H3 Longform: Talking-Head Prompt",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
