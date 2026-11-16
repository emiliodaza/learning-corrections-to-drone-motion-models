# Research presentation

The five-slide presentation summarizes the comparison of direct and residual neural predictors. It covers the flight-level split, acceleration error by speed, and a separate braking and gust-recovery benchmark in Isaac Sim. [Presentation PDF](slides.pdf) · [Beamer source](slides.tex)

Slide 4 shows synchronized replay of three controllers on the same 6 m/s initial-speed trial. Isaac Sim/PhysX computes the motion. The Python renderer draws the airframes and plots the errors from the saved trajectories. The 13.2-second clip covers 24 seconds of simulation by accelerating quiet intervals. The clip shows seed 42, while the tracking table summarizes three model seeds. [Demo video](media/drone_demo.mp4) · [Video provenance](../analysis/results/video_provenance.md)

The slide PDF embeds the MP4 as an attachment and links to the adjacent media file from the poster. The standalone MP4 works with standard video players when a PDF viewer cannot open attachments.

To compile the Beamer source, run `bash build.sh` from the repository root with TeX Live installed. [Build prerequisites](../analysis/README.md)
