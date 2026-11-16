# Learning Corrections to Drone Motion Models

Emilio Sebastian Daza Vigo · CS 74 final project · Track 3

This project compares a thrust model, two linear regressors, and two small neural networks for predicting body-frame acceleration from real NeuroBEM flight logs. The residual network learns the error of the thrust model. At recorded speeds of 5 m/s or higher, it lowers held-out acceleration RMSE from 0.6963 to 0.6772 m/s² compared with direct neural prediction. Direct prediction performs slightly better below 5 m/s.

A separate Isaac Sim/PhysX experiment compares controllers during braking and crosswind gusts. At an initial speed of 6 m/s, mean position tracking RMSE is 19.89 cm for thrust, 10.85 cm for the direct network, and 11.10 cm for the residual network. The demo renders saved simulator poses in Python.

## Report and presentation

- [Report PDF](report/report.pdf) and [LaTeX source](report/report.tex). Seven pages including references and two drone snapshots.
- [Presentation PDF](presentation/slides.pdf) and [Beamer source](presentation/slides.tex). Five slides with the MP4 attached on slide 4.
- [Drone video](presentation/media/drone_demo.mp4) and [timed speaker notes](presentation/presentation_notes.md).

## Code and dataset

- **Code.** The executed notebook contains the preprocessing, model training, and held-out evaluation. The Isaac Sim scripts contain the braking and gust benchmark. [Training and evaluation notebook](Learning%20Corrections%20to%20Drone%20Models.ipynb) · [Isaac Sim source](quadrotor_showcase/)
- **Dataset.** Download the University of Zurich's processed NeuroBEM flight logs (657 MB) and official test-flight list. The split manifest records which flights this project uses for training, validation, and testing. [Processed flight archive](https://download.ifi.uzh.ch/rpg/NeuroBEM/processed_data.zip) · [Official test-flight list](https://download.ifi.uzh.ch/rpg/NeuroBEM/testset.txt) · [Dataset documentation](https://rpg.ifi.uzh.ch/neuro_bem/Readme.html) · [Project split manifest](analysis/results/flight_splits.csv)
- **Simulation data.** The saved trajectories and metrics are included in this repository. [Isaac Sim flight logs](isaac-11043212/flight/)

## Repository layout

| Location | Contents |
| --- | --- |
| [`Learning Corrections to Drone Models.ipynb`](Learning%20Corrections%20to%20Drone%20Models.ipynb) | Executed notebook for the flight split, training, and evaluation |
| `neurobem_dynamics.npz` | Trained neural weights, fitted thrust coefficient, and input scaler |
| [`quadrotor_showcase/`](quadrotor_showcase/) | Isaac Sim benchmark, camera capture, and replay scripts |
| `isaac-11043212/flight/` | Saved simulation trajectories, metrics, and replay poses |
| [`analysis/`](analysis/) | Reproduction scripts, aggregate results, and provenance |
| [`report/`](report/) and [`presentation/`](presentation/) | Final deliverables and editable LaTeX |

## Reproduce

Open the notebook in Colab to rerun training on the public NeuroBEM release. For checkpoint verification, simulation replay, and LaTeX builds, follow the [reproduction guide](analysis/README.md). The guide includes the dataset links and checksums. The 657 MB source dataset is downloaded separately.

With TeX Live installed, run `bash build.sh` from the repository root to rebuild both PDFs.
