# AI-Enabled Fitness App

A workout form analyser. Upload a video of push-ups, squats or a plank, and the Streamlit app tracks the body with **MediaPipe Pose**, counts repetitions from joint angles, gives live form cues on each frame, and ends with a written session report.

Everything runs locally on the CPU. The video is read with OpenCV from a temporary file that is deleted after processing, so no footage leaves the machine.

<!-- demo: docs/demo.gif -->

## What it does

| Exercise | Joint angle tracked | How a rep or hold is counted | Live form cues |
|---|---|---|---|
| Push-up | Elbow (shoulder, elbow, wrist), averaged over left and right arm | One rep each time the average elbow angle goes from above 150 degrees (up) to below 90 degrees (down) | "Extend Arms Fully" and "Go Lower", based on how far the angle is from the top and bottom thresholds |
| Squat | Knee (hip, knee, ankle), averaged over left and right leg | One rep each time the average knee angle goes from above 165 degrees (standing) to below 90 degrees (squatting) | "Stand Straighter" and "Go Deeper", on the same logic |
| Plank | Body line (shoulder, hip, ankle), left side | Hold timer runs while the angle stays within 160 ± 15 degrees and resets when form breaks | "Hold Straight", "Lift Hips", "Lower Hips" |

On every frame the app draws:

- the pose skeleton, with the left and right limbs in different colours,
- the live joint angles next to the elbows, knees or hip,
- a status box with the rep count, the current cue and, for the plank, the hold timer,
- a progress bar showing how far the current rep is between the up and down thresholds (push-up and squat).

When the video ends, it writes a **session report**:

- push-up and squat: total reps, pace in reps per minute, average bottom (depth) and top angles, and rule-based recommendations when depth or extension fell short of the thresholds,
- a left/right symmetry check that flags an average difference above 10 degrees between sides in the down phase,
- plank: the longest continuous hold in correct form, and recommendations based on how often the hips were out of line,
- a count of each correction that was triggered, and a visibility note when the person was often not detected.

## How it works

1. OpenCV reads the uploaded video frame by frame.
2. MediaPipe Pose (detection and tracking confidence 0.5) returns 33 body landmarks per frame.
3. For each exercise, three landmarks define a joint. The angle at the middle landmark is computed from the two vectors with `arctan2`, in pixel coordinates.
4. A small state machine (`up` / `down`, or `hold` / `adjust` for the plank) turns the angle stream into reps, cues and hold time, using the thresholds in the `thresholds` dict at the top of `src/app.py`.
5. Per-frame metrics (angles, state, cue, rep count, hold time) are collected and summarised into the report.

## Files

| File | What it does |
|---|---|
| `src/app.py` | The Streamlit app: video upload, pose tracking, rep counting, form cues, overlays and the session report |
| `bicep_curl.py` | Early webcam prototype: tracks the left elbow angle live and shows a form cue for bicep curls |
| `main.py` | Minimal webcam demo that draws the MediaPipe pose skeleton |

## How to run

Requires Python 3.10 or newer. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

```bash
uv sync
uv run streamlit run src/app.py
```

Then open the local URL Streamlit prints, choose an exercise in the sidebar and upload a video (mp4, mov, avi or mkv). For best results, film from the side with the whole body in frame and good lighting.

The two webcam scripts need a webcam and a display; press `q` to quit:

```bash
uv run python bicep_curl.py
uv run python main.py
```

## Limitations

- Fixed angle thresholds, one set per exercise. They are not adapted to the person, the camera angle or the variant of the exercise.
- Angles are 2D, from the image plane, so the result depends on camera position. A side view works best.
- The plank check uses the left side only and an unsigned angle, so it cannot tell sagging hips from raised hips, and a body line above 175 degrees falls outside the accepted band.
- The plank hold timer uses wall-clock time during processing, not the video's own timestamps, so hold times depend on processing speed.
- The report's recommendations are simple rules over the per-frame cues, not a trained model.
- No automated tests yet.

## License

MIT, see [LICENSE](LICENSE).
