# AI-Enabled Fitness App

A webcam rep counter for strength training. It follows a body through a workout with **MediaPipe Pose**, reads the joint angle of the working limb from the tracked landmarks, and uses that angle to decide when a repetition has been completed.

Everything runs locally. The webcam feed is read with **OpenCV** and the pose model runs on the CPU, so no video ever leaves the machine.

## Files

| File | What it does |
|---|---|
| `main.py` | Opens the webcam, runs pose estimation on every frame and draws the tracked landmarks |
| `bicep_curl.py` | The geometry: an angle-between-three-points helper used to judge where the limb is in the movement |
| `src/app.py` | Application entry point |

## Running it

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

```bash
uv sync
uv run python main.py
```

It needs a working webcam and a display; it is a desktop tool, not a web app.

## Notes

Counting repetitions from a joint angle is simple and works well for a single exercise in a stable camera position. It is sensitive to camera framing and to lighting, because that is what the pose model is sensitive to. Anything more robust would need the camera calibrated per session or a small classifier over the angle time series instead of fixed thresholds.
