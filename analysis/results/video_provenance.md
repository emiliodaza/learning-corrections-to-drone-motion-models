# Supplied showcase video audit

## Source and encoding

Source in this repository: `presentation/media/drone_demo.mp4`. Original export: `isaac-11043212/quadrotor_showcase.mp4`.
SHA-256: `8d51d6c92da233746e7092c2631571f7ad89f78356ed33efed25227cf5f68a6f`.
`ffprobe` reports one H.264 video stream, 1920×1080, 30 fps, 396 frames, 13.200 s, 1,036,880 bytes. No audio stream is present. Container encoder is Lavf62.12.102.

## Actual visual contents and provenance

Inspected all four extracted frames at original resolution. Each displays three synchronized orthographic drone views for thrust, direct NN and residual NN; commanded floor rings, ground track dots/trails, and a 0.5 m grid; current 3D tracking errors; full 24 s error curves and residual-minus-direct curve with a playhead; and a fixed summary table across seeds 42–44 at 6 m/s. The video shows trial initial speed 6 m/s, model seed42, alongside the full-seed summary.

The visible header explicitly reads **“Python replay of Isaac Sim poses”**. The airframes are drawn in Python from saved simulator poses, as implemented in `replay_scene.py::draw_view` when no native_mask is supplied, and `present_showcase.py::make_video` chooses that exact header when no masks directory is used. Therefore this supplied MP4 is a Python visualization of saved Isaac Sim/PhysX motion, **not actual Isaac camera footage and not real drone footage**. The existing `render/frames/rgb_*.png` are a separate unusable grayscale capture and are not the source of these snapshots. Use the supplied MP4 frames for the requested visual evidence.

## Simulation time and playback

`ReplayScene` defines simulator times as (frame+1)/30 for 720 recorded pose frames, from 0.033333 to24.0 s. Its playback plan has exactly396 output frames, matching this MP4. Active motion plays at1×; selected quiet spans play at4×. “Quiet” requires all three drones below0.10m tracking error and0.08m/s estimated speed, held in a run of at least1.5 s. It excludes windows from0.3 s before gust onset through0.8 s after gust ending.

|Video time interval (s)|Displayed simulated interval (s)|Playback|
|---|---|---|
|0.000–5.000|0.033–5.000|1×|
|5.000–6.133|5.033–9.433|4×|
|6.133–10.600|9.567–14.000|1×|
|10.600–13.067|14.033–23.767|4×|
|13.067–13.200|23.900–24.000|1×|

These endpoints describe the timestamps of frames in each block, not a resampled continuous dynamics record. The video clock visibly states simulation time and playback rate. It is acceptable to say “13.2 s replay of a24 s simulation, with quiet intervals accelerated4×.”

## Extracted snapshots

Full frame, exact H.264 decode into PNG, with no drawing, recoloring, or compositing:

|PNG in `report/figures/`|Zero-based video frame|Video timestamp (s)|Displayed simulation time (s)|Playback|Phase|
|---|---|---|---|---|---|
|`snapshot_braking_gust.png`|53|1.766667|1.80|1×|+2m/s crosswind during braking|
|`snapshot_braking_recovery.png`|74|2.466667|2.50|1×|Braking, first gust ended|
|`snapshot_hover_gust.png`|251|8.366667|11.80|1×|−3m/s crosswind during hover|
|`snapshot_hover_recovery.png`|287|9.566667|13.00|1×|Recovery after hover gust|

Recommended pair: **`snapshot_braking_gust.png` and `snapshot_hover_gust.png`**, showing both active disturbances. At1.80s the visible current errors are56.0cm thrust,12.6cm direct,15.7cm residual. At11.80s the visible current errors are34.0cm thrust,34.3cm direct,34.1cm residual. These are instantaneous errors for seed42, not full-trial RMSEs. Snapshot table RMSE values round to one decimal cm and cannot communicate the submillimeter post-gust differences used in the report.

For report layout, a normal crop to the top three-view band may improve drone visibility. Keep controller names and show simulation times in native captions. Captions should identify Python replay of saved PhysX trajectories, speed6m/s and seed42. A full screenshot preserves source and clock labels but all internal text becomes small when scaled to a narrow figure.

Extraction commands used:

```sh
ffmpeg -hide_banner -loglevel error -i presentation/media/drone_demo.mp4 -vf "select=eq(n\,53)" -vsync 0 -frames:v 1 report/figures/snapshot_braking_gust.png
ffmpeg -hide_banner -loglevel error -i presentation/media/drone_demo.mp4 -vf "select=eq(n\,74)" -vsync 0 -frames:v 1 report/figures/snapshot_braking_recovery.png
ffmpeg -hide_banner -loglevel error -i presentation/media/drone_demo.mp4 -vf "select=eq(n\,251)" -vsync 0 -frames:v 1 report/figures/snapshot_hover_gust.png
ffmpeg -hide_banner -loglevel error -i presentation/media/drone_demo.mp4 -vf "select=eq(n\,287)" -vsync 0 -frames:v 1 report/figures/snapshot_hover_recovery.png
```

## Report crops and presentation packaging

Report Figure 3 uses ordinary crops from the full braking/hover frames: `crop=1740:498:90:112` in FFmpeg (width:height:x:y). Files are `snapshot_braking_views.png` and `snapshot_hover_views.png`. Labels and instantaneous errors remain intact; simulation times are provided in LaTeX subcaptions. No drone geometry or simulated values are changed.

The original video is available at `presentation/media/drone_demo.mp4`. The Beamer deck attaches the same MP4 and links to that file from the poster on slide 4. The report contains the two gust snapshots with their simulation times.
