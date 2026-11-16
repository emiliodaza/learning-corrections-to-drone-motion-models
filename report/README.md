# Research report

The report defines the acceleration-prediction task, documents the grouped flight split and model training, and compares held-out errors across speed groups. It also describes the Isaac Sim benchmark, explains its limits, and presents snapshots from both gusts. [Report PDF](report.pdf) · [LaTeX source](report.tex) · [References](references.bib)

The primary experimental evidence comes from held-out NeuroBEM flight logs. The Isaac Sim study is a separate closed-loop experiment with a shared simulated plant. The report explains which conclusions each experiment supports. [Reproduction guide](../analysis/README.md) · [Result tables](../analysis/results/)

Run `bash build.sh` from the repository root to compile the report and presentation. The build uses TeX Live and the figures included with the source.
