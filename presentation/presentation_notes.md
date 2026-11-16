# Four-minute presentation script with drone demo

Use the five-slide presentation without adding a cover or reference slide. The schedule reserves 24 seconds on slide 4 for the demonstration and final-frame inspection, and finishes at 3:55. Rehearse aloud and personalize the explanations.

The packaged `media/drone_demo.mp4` lasts **13.2 seconds** and shows **24 seconds of simulation**. Quiet intervals play at 4×, while braking and gust transients play at 1×. Keep the on-screen simulation clock visible. The video shows seed 42 at 6 m/s. Its tracking table summarizes all three training seeds.

## Slide 1 — 0:00–0:32

“My question is whether learning a correction to a simple drone model improves acceleration prediction on unseen flights. The physical baseline predicts thrust from squared rotor speeds. The residual network learns its remaining error, and we add that correction back. I compare this with linear regression and direct neural prediction, using the same ten measured inputs. I then examine transfer to simulated control.”

## Slide 2 — 0:32–1:15

“The key evaluation choice is splitting by original flight. All segments from one flight stay together, because nearby samples are correlated. Expanding the authors’ test list to whole flights leaves 65 training, 17 validation, and 13 test flights. I retain every eighth row and fit scaling on training data only. Both networks have one 32-unit hidden layer and identical training settings. Validation selects checkpoints and the ridge penalty. Three paired training seeds measure optimization variability. The final viewer uses ipywidgets, and Isaac Sim extends the proposal.”

## Slide 3 — 1:15–2:00

“Nonlinear learning gives the main improvement. Overall acceleration RMSE falls from about 1.74 for thrust and 0.88 for ordinary least squares to about 0.50 for either network. Residual prediction has the lowest overall mean, but the margin is small. Direct prediction performs best below five meters per second. At or above that threshold, residual RMSE is 0.677 versus 0.696, a 2.7 percent reduction. Residual prediction wins in ten of twelve fast test flights, providing exploratory support. The reported seed standard deviation measures training variability, rather than uncertainty over future flights.”

## Slide 4 — 2:00–3:20, including the demo

### Introduction — 2:00–2:18

“Here is the six-meter-per-second braking and gust trial. Isaac Sim’s PhysX engine generated the motion. The visualization replays saved poses with Python-rendered drones and tracking-error plots. The floor ring marks the command.”

### Demo and inspection — 2:18–2:42

**Click Play once and let the entire 13.2-second clip finish.** Stay silent during playback so the audience can follow the drones, gust annotations, and synchronized plots. Use the remaining roughly 11 seconds to point to the final comparison and the tracking-error table. Do not replay the clip during the timed talk. Return to slide 4 after the clip ends.

### Interpretation — 2:42–3:20

“Both learned controllers improve full-trial tracking compared with thrust alone. Direct control is slightly better on average, about 10.85 versus 11.10 centimeters RMSE. Post-gust errors are nearly identical. This ranking differs from the offline prediction result. The simulator uses a hand-designed plant, shared feedback gains, and lateral learned corrections only. Consequently, this demo illustrates behavior in a controlled test, while real-flight control performance remains an open evaluation question.”

## Slide 5 — 3:20–3:55

“Residual learning gives a modest prediction benefit in the faster regime, while direct prediction remains competitive. Next I would repeat selection across flight groups, gather independent sessions, compare explicit drag terms, and validate control on a real platform after checking frames and uncertainty. GPT assisted residual-learning concepts and Isaac Sim use through Dartmouth Clusters.”

## Playback preparation

- Click the poster on slide 4 to open the video. The PDF also includes the MP4 as an attachment.
- If your PDF viewer blocks playback, open `media/drone_demo.mp4` directly and return to slide 4 afterward.
- Rehearse on the presentation computer and time the full demo once.

# Oral-defense preparation

The answers below describe the implementation and its limits. They should be checked against the final code and practiced without reading them verbatim.

## What exactly does the model predict?

The released `acc x`, `acc y`, and `acc z` channels, at the same time as the measured inputs, in the documented front-left-up body frame. This is supervised regression of recorded responses. It does not predict future commands, learn a policy, or prove stable control. The official README names these body-acceleration channels but does not explicitly spell out gravity subtraction. Avoid equating them with a world-frame velocity derivative without verifying the release processing code.

## What are the ten inputs, and which fields are excluded?

Three body velocities, three body angular rates, and the squares of four measured rotor speeds. Acceleration labels, angular acceleration, position, timestamps, quaternions, flight identity, and the authors’ prediction files do not enter the regression. Battery and rotor-speed derivatives are omitted to keep the comparison small. Timestamps and IDs remain available for grouping and diagnostic plots.

## Why does a random row split give an optimistic result?

Nearby observations from the same trajectory have similar inputs and responses. A random row split can put near-duplicates of the same maneuver in both training and test sets. Holding out entire original flights makes the test closer to predicting a new flight. It still does not guarantee independence across flights recorded in the same session or on the same platform.

## Is the test split identical to the original NeuroBEM paper’s split?

The notebook expands the authors’ listed test segments to all segments from the corresponding original flights. That prevents flight overlap. Report this closure rule rather than claiming an exact reproduction of the paper’s experiment.

## How is the physical coefficient fitted?

Let `s` be the sum of the four squared rotor speeds. The coefficient is the least-squares fit through the origin, `c = sum(s * a_z) / sum(s**2)`, using training observations only. The baseline predicts `(0, 0, c*s)`. It is an intentionally simple fitted thrust approximation, rather than the full blade-element momentum model from NeuroBEM.

## What distinguishes direct and residual learning?

The direct MLP fits `a`. The residual MLP fits `a - a0`, then adds `a0` at inference. Both receive the same standardized features and use the same 10–32–3 architecture. This architecture has 451 learned parameters. The fixed physical skip adds structure without adding learned MLP parameters. A difference in performance can reflect optimization and target scaling as well as useful physical structure, so a small residual advantage does not by itself establish the mechanism.

## How did tuning avoid test leakage?

The coefficient, input scaler, OLS, ridge candidates, and neural weights use training data. Ridge penalty and the checkpoint for each neural run use validation RMSE. Test metrics describe the selected models. The later paired-flight diagnostic is exploratory analysis of that held-out test set, so it is not a new untouched confirmatory test. Repeated use of the same test set to redesign the model would require a fresh evaluation set.

## Why did ridge lose to ordinary least squares on the test set?

Validation selected `alpha=10000` from `{1,100,10000}`. This setting can help validation while imposing too much shrinkage for a different test distribution. OLS has lower overall and fast-bin test RMSE, while ridge slightly improves the slow bin. This is a legitimate result. Do not replace the selected penalty with a test-optimal value and keep calling the result unbiased.

## Why retain every eighth observation?

It reduces training cost while preserving order and segment boundaries. It does not make all rows independent or equalize time spacing. Most original intervals are about 0.0025 seconds, but some segments have larger intervals. After thinning, the sampling interval still differs between files. Current metrics weight observations equally, rather than weighting elapsed time or giving every flight equal influence.

## How do the two RMSE definitions differ?

Offline acceleration RMSE is the square root of the mean squared error over all selected observations and all three acceleration coordinates. It has units of meters per second squared. The simulated position-tracking metric is the square root of the mean squared three-dimensional position-error norm, in meters or centimeters. Their absolute values are not directly comparable. MAE in the report complements RMSE by giving less emphasis to large errors.

## What does the standard deviation across seeds mean?

It measures sensitivity to neural initialization and minibatch order on the same fixed data partition. Three seeds give a limited estimate of optimization variability. It does not quantify uncertainty over new flight sessions, platforms, or weather. The deterministic thrust baseline repeats identically across these seeds, so zero variation is expected for it.

## Why use whole flights for the paired diagnostic?

Rows within a flight are correlated. The diagnostic first averages each model’s per-seed fast-bin RMSE for each flight, then subtracts direct from residual. Negative differences favor residual. It gives each of the twelve flights equal weight. The pooled speed table instead gives more weight to flights with more observations, so the two summaries answer slightly different questions.

## What does the sign test show?

Under a null of equally likely positive and negative independent flight differences, ten residual wins out of twelve gives the two-sided exact binomial probability 0.0386. It uses the signs, not magnitudes. Shared recording conditions, only twelve flights, and exploratory analysis limit the inference. A p-value is not the probability that residual learning is true or a guarantee of future performance.

## What does the bootstrap interval show?

The notebook resamples twelve whole-flight differences with replacement 20,000 times, using RNG seed 42, and takes the 2.5th and 97.5th percentiles of their means. The interval is approximately `[-0.0430,-0.0083] m/s^2`. It conditions on the existing flights and trained runs. It does not include all uncertainty from model selection, new platforms, or dependence across recording sessions.

## Why is the vertical prediction error larger?

The measured result is that `z` RMSE exceeds the two lateral values. Plausible reasons include stronger thrust-related variation, unmodeled rotor effects, and measurement noise. The current experiment does not isolate these causes. Inspecting high-error flight segments identifies when errors occur, but causal claims require targeted ablations or additional measurements.

## Why does better prediction fail to guarantee better control?

A controller acts on errors encountered along its own trajectory. The showcase uses the network’s lateral estimates, a prescribed force model, clipping and feedback gains. The offline metric also includes vertical error and a different input distribution. These differences can reverse the ranking. The direct model’s slightly smaller mean simulated tracking error is evidence about this task, not broad superiority in real flight.

## Is the video a new experiment or an Isaac Sim camera recording?

It is a visualization of an existing saved trial, not an additional independent trial. Isaac Sim/PhysX integrated the motion. The provided video replays saved positions and orientations with Python-rendered drone graphics and synchronized error plots. It does not use Isaac camera masks. The 13.2-second file covers 24 simulated seconds because quiet intervals use 4× playback. Report snapshots come from the same visualization.

## What would make the control evidence stronger?

Verify body/world frame handling against the original label processing, test a range of independently identified plants and disturbances, report safety failures and latency, and compare controllers on real held-out flight maneuvers. Add uncertainty or domain checks before applying corrections outside the training distribution.

## How did the final project differ from the proposal?

The main supervised pipeline and five-model comparison remain. The viewer uses notebook ipywidgets instead of Streamlit, keeping diagnostics near the experiment. The project adds a checkpoint export, Isaac Sim braking/gust trials, and a paired fast-flight diagnostic. The simulation is an extension and should not replace the principal real-log prediction result.

## How was generative AI used?

The user confirmed ChatGPT assistance with residual-learning concepts and Dartmouth cluster/Isaac Sim setup. OpenAI Codex assisted the current artifact review, numerical checks, report/deck writing, LaTeX, and presentation preparation. The proposal also reports earlier AI help with topic selection, literature discovery, and a preliminary notebook. An earlier model identity is not documented and must not be invented. Explain which generated components you inspected, revised, and understood. Do not claim an independent human review that has not happened.

# Final personal checks

- Practice deriving the thrust fit and residual target without notes.
- Explain the split, model selection, and metric definitions in plain language.
- Inspect the worst-flight plots and explain what the curves show without asserting an untested cause.
- Confirm the AI-use history accurately reflects all material tools and tasks.
- Rehearse once with the complete video against a timer, aiming to finish several seconds before four minutes.
