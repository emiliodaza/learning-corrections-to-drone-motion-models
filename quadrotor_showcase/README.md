# Quadrotor braking and gust recovery

This benchmark compares a thrust controller with two controllers that use learned lateral acceleration compensation. It uses a common rigid-body plant in Isaac Sim 6.2.0 and pretrained direct and residual networks. The published experiment covers two initial speeds, three training seeds, and three controllers for 18 trajectories.

## Verify and replay the archived experiment

Run these commands from the repository root with Python 3 and NumPy installed. For the interactive viewer, also install Matplotlib. The MP4 export additionally requires Pillow and FFmpeg on `PATH`.

```sh
python analysis/verify_simulation.py --repo .
python quadrotor_showcase/present_showcase.py isaac-11043212
python quadrotor_showcase/present_showcase.py isaac-11043212 --plot
```

The verifier checks trial keys, sample counts, timestamps, and flags. It recomputes the metrics from 51,858 stored samples and compares them with `flight_metrics.csv` to a tolerance of `1e-12`. These commands use archived data and run locally without Isaac Sim or a GPU. The [numerical results](../analysis/results/simulation_aggregate.csv) and [verification record](../analysis/results/simulation_verification.json) accompany the data.

The [released MP4](../presentation/media/drone_demo.mp4) uses Python-drawn quadrotors positioned and oriented by saved PhysX states from the 6 m/s trial with seed 42. Active motion plays at 1×. Selected quiet intervals play at 4×, so the 24 s experiment fits in 13.2 s of playback. The [video provenance record](../analysis/results/video_provenance.md) gives source checksums and the simulation timestamps of the report frames.

Generate another annotated replay from the archived results with the following command. It writes `isaac-11043212/quadrotor_showcase.mp4`.

```sh
python quadrotor_showcase/present_showcase.py isaac-11043212 --video
```

## Experimental settings

| Setting | Value |
| --- | --- |
| Physics sampling | 120 Hz for 24 s, including the samples at 0 and 24 s |
| Initial forward speed | 3 or 6 m/s |
| Position reference | Half-cosine braking over 3 s, followed by a fixed waypoint at 2.3 m altitude |
| Crosswind along world y | +2 m/s from 1 to 2 s, then −3 m/s from 10 to 12 s |
| Shared plant | 0.772 kg mass, diagonal inertia of 0.0025, 0.0025, and 0.0043 kg m², and 0.04 s rotor lag |
| Model variants | Thrust without learned compensation, direct neural prediction, residual neural prediction |
| Neural weights | Training seeds 42, 43, and 44 from `neurobem_dynamics.npz` |

All controllers share the reference, position and attitude feedback, motor mixing, and scripted drag. The checkpoint supplies the fitted thrust coefficient, training feature scaler, and network weights. The learned controllers rotate the predicted body x and y accelerations into the world horizontal plane, limit their norm to 3 m/s², and taper compensation toward zero at low horizontal speed. Both use ground-relative velocity as input. The physical drag uses air-relative velocity.

Tracking RMSE is the square root of the mean squared 3D position distance from the reference. The post-gust window begins at 12 s. Recovery requires position error of at most 0.25 m and speed of at most 0.25 m/s through the end of the trial, with at least one second remaining and no altitude below 0.15 m during the trial. Standard deviations describe variation across trained neural checkpoints. The thrust trials have identical scores across the three seeds.

The comparison concerns lateral compensation in this specific plant. It uses a hand-designed drag law, two fixed wind events, and the same simulator parameters across seeds. The vertical network output is outside the feedback experiment. See the [report](../report/report.pdf) for the separate held-out acceleration prediction results and interpretation.

## Run Isaac Sim on Slurm

Use a Linux compute environment with a compatible NVIDIA GPU and driver, Slurm, and Apptainer. The batch script expects an Isaac Sim 6.2.0 container named `isaac-sim-6.2.0.sif` and `neurobem_dynamics.npz` in the directory from which you submit. Obtain the container through your site's NVIDIA and Apptainer workflow. The script's default GPU partitions belong to Dartmouth Discovery. Override the partition at submission on another site, or adapt the resource directives to the local scheduler.

```sh
sbatch quadrotor_showcase/run_discovery_showcase.sbatch
```

The script mounts the repository at `/benchmark`, uses `/isaac-sim/python.sh` in the container, and writes cache files under `.isaac-cache/`. It saves the 120 Hz trajectories and metrics under `results/isaac-JOB_ID/flight/`. That directory also holds the 30 Hz `replay.npz` for the 6 m/s trial with seed 42. The mask renderer then saves transparent airframe silhouettes to `results/isaac-JOB_ID/render/masks/`.

Pass an existing job ID to render its saved poses again. For example, the command below expects `results/isaac-123456/flight/replay.npz` and writes the new masks under `results/isaac-123456/showcase-NEW_JOB_ID/masks/`.

```sh
sbatch quadrotor_showcase/run_discovery_showcase.sbatch 123456
```

Use these corresponding commands in an existing Isaac Sim environment when running without Slurm. Run them from the repository root and adjust the launcher path to the installation.

```sh
/isaac-sim/python.sh quadrotor_showcase/run_isaac.py --long-run \
  --checkpoint neurobem_dynamics.npz --output results/local/flight
/isaac-sim/python.sh quadrotor_showcase/render_showcase.py \
  results/local/flight/replay.npz results/local/render
python quadrotor_showcase/present_showcase.py results/local --video \
  --masks results/local/render/masks
```

The `--masks` option uses silhouettes derived from Isaac Sim camera masks and combines them with the Python overlays. Omitting it uses the Python replay shown in the released demo. Rendering operates on stored poses. Full simulation execution uses the `--long-run` command above. The [reproduction guide](../analysis/README.md) covers checkpoint verification, training, and LaTeX builds.
