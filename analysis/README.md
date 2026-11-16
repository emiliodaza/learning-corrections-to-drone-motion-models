# Reproducing the experiments

The project has two experiments. The first evaluates acceleration predictors on held-out NeuroBEM flights. The second compares position tracking in a separate Isaac Sim braking and gust benchmark. The steps below distinguish fresh training and simulation from evaluation of the archived model weights and trajectories.

Run the shell commands from the repository root.

## NeuroBEM data and checkpoint evaluation

Download the 657 MB [processed flight archive](https://download.ifi.uzh.ch/rpg/NeuroBEM/processed_data.zip) and the [official test flight list](https://download.ifi.uzh.ch/rpg/NeuroBEM/testset.txt) from the University of Zurich. The authors publish the [dataset documentation](https://rpg.ifi.uzh.ch/neuro_bem/Readme.html) separately. Keep the ZIP intact because the verification script reads its CSV segments directly.

Use a Python environment with NumPy and pandas. The archived verification used Python 3.12.14, NumPy 2.3.5, and pandas 2.2.3.

```sh
python -m pip install numpy==2.3.5 pandas==2.2.3
mkdir -p data
curl -fL https://download.ifi.uzh.ch/rpg/NeuroBEM/processed_data.zip -o data/processed_data.zip
curl -fL https://download.ifi.uzh.ch/rpg/NeuroBEM/testset.txt -o data/testset.txt
python analysis/reproduce_checkpoint.py \
  --archive data/processed_data.zip \
  --test-list data/testset.txt \
  --checkpoint neurobem_dynamics.npz \
  --output analysis/results
```

| Input | SHA-256 |
|---|---|
| `processed_data.zip` | `a3e2e83d156b48a95c346c4fd9b3ded0b62781331dc613f1cb9ab285be2b36a8` |
| `testset.txt` | `d59cd589105cf4cd3fcfe0a34a0b4824d6ae2560db13d34b9a7acd169deeb367` |
| `neurobem_dynamics.npz` | `3a4d837f8f955724a0210b965f42c69065b556a3bb3d6581bf33cbcda6a8b826` |

The verification script groups segments by original flight ID, reserves every flight in the authors' test list, and reconstructs the seeded train and validation split. It checks the archive before retaining every eighth observation within each segment. It refits the thrust coefficient, ordinary least squares model, and three ridge candidates on the training partition. NumPy evaluates the six archived neural networks using the saved weights and training-fitted scaler. The resulting metrics agree with the executed notebook to the displayed precision.

| Output in `analysis/results/` | Contents |
|---|---|
| `verification_provenance.json` | Environment versions, input hashes, split counts, scaler checks, and selected ridge penalty |
| `flight_splits.csv`, `data_audit.csv` | Segment assignments and the archive quality checks |
| `ridge_validation.csv`, `validation_checkpoint_metrics.csv` | Validation scores for the ridge grid and saved neural checkpoints |
| `test_metrics.csv`, `test_metrics_by_run.csv` | Pooled and per-axis acceleration errors, including individual neural seeds |
| `metrics_by_speed.csv`, `metrics_by_flight.csv` | Acceleration RMSE by speed group and original flight |
| `paired_high_speed_flights.csv`, `paired_diagnostics.json` | Exploratory paired comparison on flights with observations at or above 5 m/s |
| `verified_test_predictions.npz` | Locally generated, row-level prediction arrays used to regenerate the diagnostic figures |

For the prediction experiment, the standard deviations describe variability across the three training seeds. The paired high-speed analysis uses one error difference per eligible flight.

## Fresh neural training

Open the [executed notebook](../Learning%20Corrections%20to%20Drone%20Models.ipynb) in Google Colab and run its cells in order with NumPy, pandas, scikit-learn, PyTorch, SciPy, Matplotlib, and ipywidgets available. The notebook downloads the same archive and test flight list, fits the baselines and neural networks, and saves the selected weights to `results/neurobem_dynamics.npz` in the notebook runtime. A local Jupyter run needs equivalents for the `!wget` commands and the `google.colab.files` export cell. The video display cell is optional.

The notebook specifies the grouped split, sampling stride, input features, ridge grid, neural architecture, Adam settings, paired seeds, and validation checkpoint selection. The original training hardware and full environment versions are unrecorded. Compare fresh runs with the archived scores while allowing for numerical differences across hardware and software versions.

## Archived Isaac Sim results

The [archived trial files](../isaac-11043212/flight/) contain 18 trajectories for two initial speeds, three training seeds, and three controllers. Recompute the trial metrics and inspect the summary with NumPy installed:

```sh
python analysis/verify_simulation.py --repo . --output analysis/results
python quadrotor_showcase/present_showcase.py isaac-11043212
```

`verify_simulation.py` uses the project's `score_flight` function to score all 51,858 saved trajectory rows. It checks trial keys, sample counts, timestamps, and recovery flags, and compares the numeric metrics with the archived CSV using a tolerance of `1e-12`. The recorded verification found zero numerical discrepancy. The script writes `simulation_aggregate.csv`, `simulation_paired.csv`, `simulation_windows.csv`, and `simulation_verification.json`. This workflow checks the stored simulation output without launching Isaac Sim. The variation across seeds reflects the network checkpoints evaluated on the same simulated plant.

Use `--plot` for the interactive Matplotlib view or `--video` to export an annotated replay. Video export uses Matplotlib, Pillow, and FFmpeg. The [archived replay](../presentation/media/drone_demo.mp4) renders saved simulator poses in Python. The [video provenance](results/video_provenance.md) records its checksum, playback timeline, and figure extraction commands.

## Fresh Isaac Sim benchmark

The [Slurm script](../quadrotor_showcase/run_discovery_showcase.sbatch) records the original cluster setup. It needs an NVIDIA GPU, Apptainer with GPU support, and an Isaac Sim 6.2.0 container named `isaac-sim-6.2.0.sif` at the repository root. Adapt its Slurm partitions and any module setup for the cluster in use.

```sh
sbatch quadrotor_showcase/run_discovery_showcase.sbatch
```

The script runs `run_isaac.py --long-run` for the 24-second braking and gust task and saves flight metrics, trajectories, and replay poses under `results/isaac-<job-id>/flight/`. It then renders camera masks. Pass an existing physics job ID as the script's first argument to capture views from that job's saved poses. Fresh physics execution requires the simulator environment. The archived numerical verification above can run independently of that environment.

## Figures and LaTeX

Install Matplotlib and run checkpoint evaluation before regenerating the prediction plots:

```sh
python analysis/render_figures.py
python analysis/plot_simulation.py --repo .
```

`render_figures.py` reads the locally generated `verified_test_predictions.npz` and writes the two three-axis flight diagnostics, the combined vertical acceleration and speed figure, and the paired-flight chart to `report/figures/`. `plot_simulation.py` reads the archived trajectories and writes the tracking figure. The committed validation learning curves come from the executed notebook. Run the notebook to regenerate its per-epoch values and plot.

Use TeX Live with Beamer, AMS math, graphics, booktabs, caption, microtype, geometry, float, hyperref, attachfile2, standard Times fonts, and BibTeX:

```sh
bash build.sh
```

This generates `report/report.pdf` and `presentation/slides.pdf` and writes auxiliary files below `build/`. For Overleaf, upload the contents of either `report/` or `presentation/` and select `report.tex` or `slides.tex` as the main document. The presentation attaches the replay MP4 and links to it from the poster on slide 4. The [standalone video](../presentation/media/drone_demo.mp4) works independently of PDF viewer support for attachments.
