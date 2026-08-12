# Spec: Two-Mode CLI for Surface Back-Projection Pipeline

Two separate scripts sharing one intermediate result format. Mode 1 computes and
bakes results; Mode 2 only visualizes previously baked results and never
recomputes.

## 1. Shared artifact: the "baked results" file

Mode 1's entire job is to produce this file; Mode 2's entire job is to read it.

```python
@dataclass
class BakedResults:
    video_id: str
    frame_order: list[str]                       # canonical frame_name sequence
    detections: dict[str, list[Detection]]         # per frame, post-YOLO-spec
    patches: dict[str, np.ndarray]                   # det_id -> (N, 3) world points; None entries omitted
    landmark_of: dict[str, int]                        # det_id -> landmark_id (post clustering)
    landmark_count: int                                  # final fruit count
    skipped_frames: list[str]                              # from reader spec §3.3 log
    params: dict                                             # eps, min_overlap_frac, conf threshold, etc.
```

Serialize as a single `.npz` (arrays) + sidecar `.json` (everything else,
including `det_id -> landmark_id` and per-frame detection metadata), or a single
pickle if simplicity is preferred over cross-language readability — pick one and
document it; this repo's README should state which.

File location convention: `WORKSPACE/results/<video_id>.baked.{npz,json}` (or
`.pkl`). Mode 2 looks here by convention, not by prompting the user for a path
every time — pass `video_id` (or the file path) as the CLI arg and derive the rest.

## 2. Mode 1 — `count.py` (pure CLI output)

```
python count.py --workspace WORKSPACE --video-id <id> [--eps 0.02] [--min-overlap-frac 0.15] [--conf 0.4]
```

**Behavior:**
1. Run reader spec (§3) → `frames` dict.
2. Run YOLO spec (§3) → `detections` dict.
3. Build 2D tracks (temporal association across adjacent frames — separate
   module, not covered by this spec, referenced here as a precondition).
4. For each track, back-project each member detection's mask
   (`backproject_mask_to_surface`) and pool into a per-track landmark patch.
5. Run `LandmarkMerger` over all landmark patches → final `landmark_count`.
6. Write `BakedResults` to the location in §1.
7. **stdout: print only the integer fruit count, nothing else.** No progress
   bars, no logging, on stdout — those go to stderr if needed. This is what makes
   `pure CLI output` mean something concrete: `count.py ... > result.txt` should
   contain exactly one line, one integer. Anything diagnostic (frame counts,
   skipped-frame warnings, timing) goes to stderr so it doesn't corrupt
   machine-parseable stdout.
8. Non-zero exit code on failure (missing COLMAP output, no registered frames,
   etc.) with the error on stderr — no baked file written on failure, so Mode 2's
   "run mode 1 first" check (§3) can rely on file-existence as a correctness
   signal, not just a convenience one.

**Explicitly not this mode's job:** any plotting, any interactivity, any
per-frame stepping. If asked for those, that's Mode 2.

## 3. Mode 2 — `inspect.py` (interactive 3D debugger)

```
python inspect.py --workspace WORKSPACE --video-id <id>
```

**Precondition check (must run first, before anything else):**
```python
baked_path = results_path(workspace, video_id)
if not baked_path.exists():
    print(f"No baked results for '{video_id}'. Run:\n"
          f"  python count.py --workspace {workspace} --video-id {video_id}\n"
          f"before using inspect.py.", file=sys.stderr)
sys.exit(1)
```
This script **never computes anything** — no YOLO inference, no back-projection,
no COLMAP reads. It only deserializes `BakedResults` and renders. If that
separation gets blurred (e.g. "just recompute if missing"), the two-mode
contract from the request breaks down into one script with a flag, which is not
what was asked for — keep Mode 2 strictly read-only against the baked artifact.

**Interaction loop:**
- State: `current_frame_idx`, initialized to `0`.
- Key bindings (matplotlib `key_press_event`):
  - Right arrow → `current_frame_idx = min(current_frame_idx + 1, len(frame_order) - 1)`, re-render.
  - Left arrow → `current_frame_idx = max(current_frame_idx - 1, 0)`, re-render.
  - Escape → close figure, exit process cleanly (exit code 0).
- Frames with no detections/landmarks still navigable (empty scatter, not skipped
  from `frame_order` — consistent stepping regardless of content, so the user
  can see *that* a frame had nothing, not just skip past confusing gaps).

**Render, per frame-step (matplotlib 3D scatter, `Axes3D`):**
- Plot every landmark's pooled 3D points that have at least one contributing
  detection in `current_frame_idx` — highlighted/full-opacity.
- Plot all *other* landmarks (contributed by other frames) at reduced
  opacity/smaller marker size, for spatial context — this is what makes it a
  debugging tool rather than just a per-frame viewer; you want to see where the
  current frame's detections sit relative to the whole reconstructed set.
- **Consistent color per landmark, stable across every frame render and across
  the whole session:** color assignment must be `landmark_id -> color` computed
  once (e.g. a hash of `landmark_id` into a fixed colormap, or a precomputed
  palette indexed by sorted landmark id), not re-derived per frame — otherwise
  the same fruit changes color as you step, defeating the purpose of the view.
- Label each visible landmark's point cluster with its `landmark_id` (matplotlib
  `text` at the patch centroid) and its class (majority vote across contributing
  detections' `class_name`).
- Title/annotation per frame: `frame_name`, index `i/N`, count of detections in
  this frame, running `landmark_count` (static, from baked results — this view
  does not recompute it).
- Also plot the camera position for `current_frame_idx` (from the frame's `t`,
  transformed to world space: `-R.T @ t`) as a distinct marker, so orbit
  progression is visually legible against the fruit landmarks — this is the
  "camera path" context the deliverables list asks for as a visual artifact, and
  Mode 2 is the natural place to produce a static version of it too (e.g. an
  extra flag `--export-camera-path-png` for the non-interactive artifact,
  separate from the interactive session).

## 4. What's explicitly out of scope for both modes

- Mode 1 does not visualize anything, ever, regardless of flags — that's a
  deliberate constraint from the request, not an oversight. A separate
  `--debug-plot` flag on Mode 1 would violate "pure CLI output."
- Mode 2 does not accept `--recompute` or similar — if the baked results are
  stale (e.g. params changed), the fix is deleting/re-running Mode 1, not adding
  a compute path to the inspector. Keeping Mode 2 dumb-and-read-only is what
  keeps it fast to open and safe to run repeatedly while debugging.

## 5. Validation

- Run Mode 1 on a short test video, confirm stdout is exactly one integer and
  stderr carries everything else.
- Run Mode 2 against that output: step through the full frame range in both
  directions, confirm colors never change for a landmark that persists across
  frames, confirm Escape exits cleanly (no traceback, exit code 0).
- Delete the baked file, run Mode 2 again, confirm the exact error message and
  non-zero exit — this is the "must render an error message requiring to run the
  first mode" requirement, and it should be tested as a real case, not assumed.
