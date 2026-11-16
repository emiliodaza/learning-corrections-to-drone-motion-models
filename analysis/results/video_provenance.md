# Simulation replay and figure provenance

## Source and encoding

The [archived MP4](../../presentation/media/drone_demo.mp4) is the original `isaac-11043212/quadrotor_showcase.mp4` export. Its SHA-256 is `8d51d6c92da233746e7092c2631571f7ad89f78356ed33efed25227cf5f68a6f`. `ffprobe` reports one H.264 video stream at 1920 × 1080 and 30 frames per second. The file contains 396 frames over 13.200 s and is 1,036,880 bytes. It has no audio stream.

The replay shows three synchronized orthographic views of the thrust, direct neural, and residual neural controllers for the 6 m/s initial-speed trial with seed 42. The overlays show the commanded position, actual ground trail, current three-dimensional position error, and error curves. A separate table summarizes the three training seeds at 6 m/s.

The on-screen label reads “Python replay of Isaac Sim poses.” [`replay_scene.py`](../../quadrotor_showcase/replay_scene.py) renders the airframes from archived PhysX positions and orientations. [`present_showcase.py`](../../quadrotor_showcase/present_showcase.py) composes the views and plots. The distributed video uses the Python renderer with no camera masks.

## Simulation time and playback

The 720 saved pose frames map to simulation time as `(frame + 1) / 30`, from 0.033333 to 24.0 s. The replay shows active motion at 1× and selected quiet spans at 4×. For a span to qualify as quiet, all three drones must stay below 0.10 m tracking error and 0.08 m/s estimated speed for at least 1.5 s. The playback plan excludes the intervals from 0.3 s before gust onset through 0.8 s after each gust ends.

| Video time (s) | Displayed simulation time (s) | Playback rate |
|---|---|---|
| 0.000–5.000 | 0.033–5.000 | 1× |
| 5.000–6.133 | 5.033–9.433 | 4× |
| 6.133–10.600 | 9.567–14.000 | 1× |
| 10.600–13.067 | 14.033–23.767 | 4× |
| 13.067–13.200 | 23.900–24.000 | 1× |

The simulation ranges give the timestamps of the first and last displayed frames in each block.

## Frame extraction

The archived PNGs are full decoded video frames. Frame indices start at zero.

| File in `report/figures/` | Frame | Video time (s) | Simulation time (s) | Phase |
|---|---:|---:|---:|---|
| `snapshot_braking_gust.png` | 53 | 1.766667 | 1.80 | +2 m/s crosswind during braking |
| `snapshot_braking_recovery.png` | 74 | 2.466667 | 2.50 | Braking after the first gust |
| `snapshot_hover_gust.png` | 251 | 8.366667 | 11.80 | −3 m/s crosswind during hover |
| `snapshot_hover_recovery.png` | 287 | 9.566667 | 13.00 | Recovery after the hover gust |

Run from the repository root:

```sh
ffmpeg -hide_banner -loglevel error -i presentation/media/drone_demo.mp4 -vf "select=eq(n\,53)" -vsync 0 -frames:v 1 report/figures/snapshot_braking_gust.png
ffmpeg -hide_banner -loglevel error -i presentation/media/drone_demo.mp4 -vf "select=eq(n\,74)" -vsync 0 -frames:v 1 report/figures/snapshot_braking_recovery.png
ffmpeg -hide_banner -loglevel error -i presentation/media/drone_demo.mp4 -vf "select=eq(n\,251)" -vsync 0 -frames:v 1 report/figures/snapshot_hover_gust.png
ffmpeg -hide_banner -loglevel error -i presentation/media/drone_demo.mp4 -vf "select=eq(n\,287)" -vsync 0 -frames:v 1 report/figures/snapshot_hover_recovery.png
```

Figure 3 of the report uses rectangular crops of the braking and hover gust frames. Both use the FFmpeg filter `crop=1740:498:90:112` (width, height, x, y). The resulting images are `snapshot_braking_views.png` and `snapshot_hover_views.png`.

The position errors displayed in those frames are instantaneous values for seed 42. At 1.80 s, the errors are 56.0 cm for thrust, 12.6 cm for direct, and 15.7 cm for residual. At 11.80 s, they are 34.0 cm, 34.3 cm, and 34.1 cm, respectively. See [`simulation_aggregate.csv`](simulation_aggregate.csv) for tracking RMSE across the full trials.
