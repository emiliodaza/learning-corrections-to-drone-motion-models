"""Report saved PhysX results in Python and export an annotated MP4."""

import argparse
from bisect import bisect_right
import csv
from math import dist
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory

import numpy as np

from quadrotor_control import BRAKE_SECONDS, BRAKING_GUSTS
from replay_scene import COLORS as REPLAY_COLORS, VIEW_SIZE, ReplayScene, draw_view


MODELS = {
    model: (label, "#%02x%02x%02x" % color)
    for (model, label), color in zip(
        (("thrust", "Thrust baseline"), ("direct", "Direct network"), ("residual", "Residual network")),
        REPLAY_COLORS,
    )
}
FPS = 30
SPEEDS = (3, 6)
SEEDS = (42, 43, 44)
VIDEO_TRIAL = (6, 42)
BACKGROUND = "#ffffff"
PANEL = "#f5f8fa"
INK = "#152431"
MUTED = "#506170"
RULE = "#c8d2da"
GUST_SHADE = "#e6edf3"
DELTA_COLOR = "#70449b"
LINE_STYLES = {"thrust": ":", "direct": "--", "residual": "-"}


def load_results(flight_dir):
    with np.load(flight_dir / "replay.npz", allow_pickle=False) as saved:
        duration = len(saved["positions"]) / FPS
        gusts = saved["gusts"].tolist() if "gusts" in saved else BRAKING_GUSTS
    has_hover_gust = len(gusts) > 1

    # loading metrics for each initial speed, training seed, and controller
    trials = {}
    with (flight_dir / "flight_metrics.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(float(row["speed_mps"])), int(row["seed"]), row["model"])
            settle = row["settling_s_after_brake"]
            result = {
                "tracking": float(row["tracking_rmse_m"]),
                "hover": float(row["hover_rmse_m"]),
                "settling": float(settle) if settle else None,
            }
            if has_hover_gust:
                recovery = row["recovery_s_after_hover_gust"]
                result.update({
                    "peak_lateral": float(row["hover_gust_peak_lateral_m"]),
                    "post_gust": float(row["post_gust_rmse_m"]),
                    "recovery": float(recovery) if recovery else None,
                })
            trials[key] = result
    expected = {(speed, seed, model) for speed in SPEEDS for seed in SEEDS for model in MODELS}
    if trials.keys() != expected:
        raise ValueError("Expected all 18 controller, speed, and seed combinations in flight_metrics.csv")

    # computing distance to the reference and signed lateral offset from the saved positions
    traces = {key: {name: [] for name in ("time", "position", "lateral")} for key in expected}
    with (flight_dir / "trajectories.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            actual = [float(row[axis]) for axis in "xyz"]
            target = [float(row[f"reference_{axis}"]) for axis in "xyz"]
            trace = traces[int(float(row["speed_mps"])), int(row["seed"]), row["model"]]
            trace["time"].append(float(row["time_s"]))
            trace["position"].append(dist(actual, target))
            trace["lateral"].append(actual[1] - target[1])
    if any(not trace["time"] or abs(trace["time"][-1] - duration) > 1e-6 for trace in traces.values()):
        raise ValueError("Every trajectory must cover the full replay duration")
    return traces, trials, duration, gusts


def comparison_text(trials, speed, metric):
    residual = mean(trials[speed, seed, "residual"][metric] for seed in SEEDS)
    direct = mean(trials[speed, seed, "direct"][metric] for seed in SEEDS)
    difference = residual - direct
    label = "post-gust RMSE" if metric == "post_gust" else "hover RMSE"
    if abs(difference) < 0.0005:
        return f"{speed} m/s · Direct and residual mean {label} differ by less than 0.5 mm"
    direction = "lower" if difference < 0 else "higher"
    return f"{speed} m/s · Residual {label} {abs(difference):.3f} m {direction} than direct across three seeds"


def print_report(trials, duration, gusts):
    print(f"PhysX flight duration  {duration:g} s")
    for start, end, velocity in gusts:
        print(f"Lateral wind  {velocity:+g} m/s from {start:g} to {end:g} s")
    has_hover_gust = len(gusts) > 1
    columns = [("tracking", "Tracking"), ("hover", "Hover")]
    if has_hover_gust:
        columns.extend([("peak_lateral", "Peak drift"), ("post_gust", "Post-gust")])
    settled_key = "recovery" if has_hover_gust else "settling"
    settled_label = "Recovered" if has_hover_gust else "Settled"
    print("\nMeans across seeds 42, 43, and 44")
    print("Tracking, hover, and post-gust RMSE in meters. Peak drift is in meters.")
    print(f"{'Speed':<7} {'Controller':<18} " + " ".join(f"{label:>10}" for _, label in columns)
          + f" {settled_label:>9}")
    for speed in SPEEDS:
        for model, (label, _) in MODELS.items():
            rows = [trials[speed, seed, model] for seed in SEEDS]
            averages = " ".join(f"{mean(row[key] for row in rows):10.3f}" for key, _ in columns)
            settled = sum(row[settled_key] is not None for row in rows)
            speed_label = f"{speed} m/s"
            print(f"{speed_label:<7} {label:<18} {averages} {f'{settled}/{len(rows)}':>9}")
        print(comparison_text(trials, speed, "post_gust" if has_hover_gust else "hover"))
    print("\nRecovery requires remaining within 0.25 m of the target at 0.25 m/s or slower"
          " for the rest of the trial, for at least one second.")


def flight_phase(time, gusts, speed):
    for start, end, velocity in gusts:
        if start <= time < end:
            return f"{abs(velocity):g} m/s crosswind during {'braking' if start < BRAKE_SECONDS else 'hover'}"
    if time < BRAKE_SECONDS:
        return f"Braking from {speed} m/s"
    if len(gusts) > 1 and time >= gusts[-1][1]:
        return "Recovery after hover gust"
    return "Holding the hover target"


def mark_phases(axis, gusts):
    for start, end, _ in gusts:
        axis.axvspan(start, end, facecolor=GUST_SHADE, edgecolor="none", zorder=0)
    axis.axvline(BRAKE_SECONDS, color=RULE, linestyle="--", linewidth=.8)


def error_difference(traces):
    """Return paired instantaneous position-error differences in centimeters."""
    direct = traces["direct"]
    residual = traces["residual"]
    if direct["time"] != residual["time"]:
        raise ValueError("Direct and residual trajectories must have matching times")
    return 100 * (np.asarray(residual["position"]) - direct["position"])


def style_axis(axis):
    axis.set_facecolor(BACKGROUND)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=RULE, alpha=.56, linewidth=.7)
    axis.tick_params(colors=MUTED, labelsize=9, length=3, width=.65)
    for side, spine in axis.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(RULE)
        spine.set_linewidth(.75)


def add_comparison_table(axis, trials, speed, hover_gust_end):
    """Place the full-trial seed statistics beside the live trajectories."""
    axis.set_axis_off()
    axis.text(0, 1.04, "Across-seed results", fontsize=14, weight="bold", color=INK)
    axis.text(0, .94, f"{speed} m/s  ·  Seeds 42–44  ·  RMSE in cm", fontsize=9.5, color=MUTED)
    second_metric = "post_gust" if hover_gust_end is not None else "hover"
    second_heading = "Post-gust" if hover_gust_end is not None else "Hover"
    axis.text(0, .79, "Controller", fontsize=9.5, weight="bold", color=MUTED)
    axis.text(.66, .79, "Tracking", fontsize=9.5, weight="bold", color=MUTED, ha="right")
    axis.text(1, .79, second_heading, fontsize=9.5, weight="bold", color=MUTED, ha="right")
    axis.hlines(.74, 0, 1, colors=RULE, linewidth=.8)
    for index, (model, (label, color)) in enumerate(MODELS.items()):
        y = .63 - .16 * index
        rows = [trials[speed, seed, model] for seed in SEEDS]
        axis.text(0, y, label, fontsize=10, color=color, weight="medium")
        for x, key in ((.66, "tracking"), (1, second_metric)):
            values = [100 * row[key] for row in rows]
            axis.text(x, y, f"{mean(values):.1f} ± {stdev(values):.1f}", fontsize=10, color=INK, ha="right")
        axis.hlines(y - .065, 0, 1, colors=RULE, linewidth=.55)
    axis.text(0, .13, "Mean ± sample SD across seeds", fontsize=8.5, color=MUTED)
    window = (f"Post-gust window starts at {hover_gust_end:g} s" if hover_gust_end is not None
              else f"Hover window starts at {BRAKE_SECONDS:g} s")
    axis.text(0, .035, window, fontsize=8.5, color=MUTED)
    axis.set(xlim=(0, 1), ylim=(0, 1))


def show_plots(traces, duration, gusts):
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.widgets import RadioButtons, Slider

    figure, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True, facecolor=BACKGROUND,
                                gridspec_kw={"height_ratios": (1.1, 1, 1)})
    figure.subplots_adjust(left=.094, right=.762, bottom=.169, top=.814, hspace=.39)
    figure.text(.094, .948, "Quadrotor controller comparison", color=INK, fontsize=20,
                weight="bold", ha="left")
    readout = figure.text(.094, .9, "", color=MUTED, fontsize=10.5)
    figure.text(.762, .852, "Shaded intervals mark crosswinds", ha="right", color=MUTED, fontsize=8.5)
    readout_axes = figure.add_axes([.795, .265, .19, .235])
    readout_axes.axis("off")
    readings = {
        model: readout_axes.text(0, 1 - i * .29, "", color=color, fontsize=9.5, va="top")
        for i, (model, (_, color)) in enumerate(MODELS.items())
    }
    difference = readout_axes.text(0, .085, "", color=DELTA_COLOR, fontsize=9.5, va="top")
    playheads, lines, markers = [], {}, {}
    for axis, metric, title in zip(axes[:2], ("position", "lateral"),
                                   ("3D error (cm)", "Lateral offset (cm)")):
        style_axis(axis)
        mark_phases(axis, gusts)
        for model, (label, color) in MODELS.items():
            lines[metric, model], = axis.plot([], [], label=label, color=color, linewidth=1.7,
                                               linestyle=LINE_STYLES[model])
            markers[metric, model], = axis.plot([], [], marker="o", markersize=4, color=color,
                                                linestyle="none", zorder=5)
        axis.set_ylabel(title, color=INK, fontsize=10)
        axis.yaxis.set_major_locator(MaxNLocator(4))
        playheads.append(axis.axvline(0, color=MUTED, linewidth=.9))

    gap_axis = axes[2]
    style_axis(gap_axis)
    mark_phases(gap_axis, gusts)
    gap_axis.axhline(0, color=MUTED, linewidth=.8)
    gap_line, = gap_axis.plot([], [], color=DELTA_COLOR, linewidth=1.7)
    gap_marker, = gap_axis.plot([], [], marker="o", markersize=4, color=DELTA_COLOR,
                               linestyle="none", zorder=5)
    playheads.append(gap_axis.axvline(0, color=MUTED, linewidth=.9))
    gap_axis.set_ylabel("Error gap (cm)", color=INK, fontsize=10)
    gap_axis.set_title("Residual − direct · Negative favors residual", loc="left", color=DELTA_COLOR, fontsize=9.5)
    gap_axis.yaxis.set_major_locator(MaxNLocator(4, symmetric=True))
    gap_axis.set_xlabel("Simulation time (s)", color=INK, fontsize=10)
    gap_axis.xaxis.set_major_locator(MaxNLocator(8, integer=True))
    gap_axis.set_xlim(0, duration)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), borderaxespad=0,
                   frameon=False, fontsize=9, labelcolor=INK, ncol=3)

    slider = Slider(figure.add_axes([.155, .066, .587, .027]), "Time (s)", 0, duration,
                    valstep=1 / FPS, valfmt="%.2f", color=MODELS["direct"][1], track_color=GUST_SHADE)
    slider.label.set_color(MUTED)
    slider.valtext.set_color(INK)
    speed_axis = figure.add_axes([.795, .695, .172, .112], facecolor=PANEL)
    speed_axis.set_title("Initial speed", loc="left", color=INK, fontsize=10)
    speed_choice = RadioButtons(speed_axis, [f"{speed} m/s" for speed in SPEEDS], active=1,
                                activecolor=MODELS["direct"][1])
    seed_axis = figure.add_axes([.795, .533, .172, .112], facecolor=PANEL)
    seed_axis.set_title("Model seed", loc="left", color=INK, fontsize=10)
    seed_choice = RadioButtons(seed_axis, [str(seed) for seed in SEEDS], activecolor=MODELS["direct"][1])
    for choice in (speed_choice, seed_choice):
        for label in choice.labels:
            label.set_color(INK)
    for control in (speed_axis, seed_axis):
        for spine in control.spines.values():
            spine.set_color(RULE)

    deltas = {(speed, seed): error_difference({model: traces[speed, seed, model] for model in MODELS})
              for speed in SPEEDS for seed in SEEDS}

    def selected_trial():
        return int(speed_choice.value_selected.split()[0]), int(seed_choice.value_selected)

    # looking up the latest simulation sample at the selected time
    def update(time):
        speed, seed = selected_trial()
        for playhead in playheads:
            playhead.set_xdata([time, time])
        for model, (label, _) in MODELS.items():
            trace = traces[speed, seed, model]
            row = bisect_right(trace["time"], time + 1e-9) - 1
            error = 100 * trace["position"][row]
            lateral = round(100 * trace["lateral"][row], 1) or 0.0
            readings[model].set_text(f"{label}\n{error:.1f} cm error · {lateral:+.1f} cm lateral")
            for metric in ("position", "lateral"):
                markers[metric, model].set_data([trace["time"][row]], [100 * trace[metric][row]])
        times = traces[speed, seed, "direct"]["time"]
        row = bisect_right(times, time + 1e-9) - 1
        delta = deltas[speed, seed][row]
        difference.set_text(f"Residual − direct  {round(float(delta), 1) or 0.0:+.1f} cm")
        gap_marker.set_data([times[row]], [delta])
        readout.set_text(f"{time:.2f} s   ·   {flight_phase(time, gusts, speed)}")
        figure.canvas.draw_idle()

    def change_trial(_):
        speed, seed = selected_trial()
        for (metric, model), line in lines.items():
            trace = traces[speed, seed, model]
            line.set_data(trace["time"], 100 * np.asarray(trace[metric]))
        for axis in axes[:2]:
            axis.relim()
            axis.autoscale(enable=True, axis="y")
        axes[0].set_ylim(bottom=0, auto=True)
        delta = deltas[speed, seed]
        gap_line.set_data(traces[speed, seed, "direct"]["time"], delta)
        extent = max(1.0, 1.12 * np.max(np.abs(delta)))
        gap_axis.set_ylim(-extent, extent)
        update(slider.val)

    slider.on_changed(update)
    speed_choice.on_clicked(change_trial)
    seed_choice.on_clicked(change_trial)
    change_trial(None)
    plt.show()


def make_video(replay, masks_dir, output, traces, trials, gusts):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.ticker import MaxNLocator
    from PIL import Image

    if not FFMpegWriter.isAvailable():
        raise RuntimeError("Install FFmpeg and make ffmpeg available on your PATH to export the MP4")
    # requiring one consecutive camera mask per saved pose when masks are supplied
    masks = []
    if masks_dir is not None:
        masks = sorted(masks_dir.glob("mask_*.png"))
        expected = [f"mask_{index:04d}.png" for index in range(len(replay.times))]
        if [path.name for path in masks] != expected:
            raise ValueError(f"Expected {len(expected)} consecutive Isaac Sim masks in {masks_dir}")

    speed, seed = VIDEO_TRIAL
    hover_gust_end = gusts[-1][1] if len(gusts) > 1 else None
    traces = {model: traces[speed, seed, model] for model in MODELS}
    deltas = error_difference(traces)
    times = traces["direct"]["time"]
    duration = replay.times[-1]
    fig = plt.figure(figsize=(16, 9), facecolor=BACKGROUND)

    def label(x, y, text, size, *, color=INK, bold=False, align="left"):
        return fig.text(x, y, text, fontsize=size, color=color, ha=align,
                        weight="bold" if bold else "normal")

    label(.033, .949, "Quadrotor braking and gust recovery", 23, bold=True)
    source = "Isaac Sim camera silhouettes" if masks else "Python replay of Isaac Sim poses"
    label(.033, .917, f"{source}  ·  {speed} m/s initial speed  ·  Seed {seed}", 10.5, color=MUTED)
    phase = label(.967, .947, "", 13, bold=True, align="right")
    clock = label(.967, .917, "", 10, color=MUTED, align="right")

    # placing the synchronized flight views above the error plots and across-seed summary
    width, height = VIEW_SIZE
    view_width = 540
    view_height = view_width * height / width
    images, errors = {}, {}
    for index, (model, (name, color)) in enumerate(MODELS.items()):
        left = (100 + 590 * index) / 1920
        label(left, .864, name, 14, color=color, bold=True)
        errors[model] = label(left + view_width / 1920, .864, "", 10, color=color, align="right")
        view = fig.add_axes([left, .473, view_width / 1920, view_height / 1080])
        view.set(xticks=[], yticks=[])
        for spine in view.spines.values():
            spine.set_color(RULE)
            spine.set_linewidth(.7)
        images[model] = view.imshow(np.zeros((height, width, 3), dtype=np.uint8))
    label(.052, .441, "Ring  commanded position    ·    Dot and trail  ground track    ·    Grid  0.5 m", 9.5, color=MUTED)

    label(.078, .390, "3D tracking error", 13, bold=True)
    label(.678, .390, "Shaded intervals mark crosswinds", 9, color=MUTED, align="right")
    chart = fig.add_axes([.078, .232, .600, .135])
    gap_chart = fig.add_axes([.078, .070, .600, .105], sharex=chart)
    for axis in (chart, gap_chart):
        style_axis(axis)
        mark_phases(axis, gusts)
        axis.set_ylabel("cm", fontsize=9.5, color=MUTED, labelpad=7)
    peak_error_cm = 100 * max(max(trace["position"]) for trace in traces.values())
    chart.set(xlim=(0, duration), ylim=(0, 1.13 * max(5.0, peak_error_cm)))
    chart.tick_params(labelbottom=False)
    chart.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    gap_chart.set_xlabel("Simulation time (s)", color=MUTED, fontsize=9.5)
    gap_chart.xaxis.set_major_locator(MaxNLocator(8, integer=True))
    gap_chart.yaxis.set_major_locator(MaxNLocator(3, symmetric=True))
    gap_chart.axhline(0, color=MUTED, linewidth=.8)
    extent = max(1.0, 1.14 * np.max(np.abs(deltas)))
    gap_chart.set_ylim(-extent, extent)
    label(.078, .195, "Residual − direct tracking error", 12, color=DELTA_COLOR, bold=True)
    gap_readout = label(.678, .195, "", 9.5, color=DELTA_COLOR, align="right")

    markers = {}
    for model, (_, color) in MODELS.items():
        trace = traces[model]
        chart.plot(trace["time"], 100 * np.asarray(trace["position"]), color=color, linewidth=1.75,
                   linestyle=LINE_STYLES[model], label=MODELS[model][0])
        markers[model], = chart.plot([], [], marker="o", color=color, markeredgecolor=BACKGROUND,
                                     markersize=5, linestyle="none", zorder=5)
    chart.legend(loc="lower center", bbox_to_anchor=(.52, 1.10), borderaxespad=0,
                 fontsize=8, labelcolor=INK, frameon=False, ncol=3, handlelength=2.6, columnspacing=1.2)
    gap_chart.plot(times, deltas, color=DELTA_COLOR, linewidth=1.75)
    gap_marker, = gap_chart.plot([], [], marker="o", color=DELTA_COLOR, markeredgecolor=BACKGROUND,
                                 markersize=5, linestyle="none", zorder=5)
    playheads = [axis.axvline(0, color=MUTED, linewidth=.9) for axis in (chart, gap_chart)]

    add_comparison_table(fig.add_axes([.717, .070, .231, .310]), trials, speed, hover_gust_end)

    writer = FFMpegWriter(fps=FPS, codec="libx264",
                         extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart"])
    # selecting every fourth frame during quiet hover while keeping the mp4 at 30 fps
    plan = list(replay.playback(gusts))
    print(f"Exporting {len(plan)} flight frames", flush=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        draft = Path(temporary) / output.name
        with writer.saving(fig, str(draft), dpi=120):
            for number, (frame, rate) in enumerate(plan, start=1):
                time = replay.times[frame]
                phase.set_text("Quiet hover" if rate > 1 else flight_phase(time, gusts, speed))
                clock.set_text(f"Simulation {time:05.2f} / {duration:g} s  ·  {rate}× playback")
                for playhead in playheads:
                    playhead.set_xdata([time, time])
                if masks:
                    with Image.open(masks[frame]) as opened:
                        strip = opened.convert("RGBA")
                    if strip.size != (width * len(MODELS), height):
                        raise ValueError(f"Unexpected mask dimensions in {masks[frame]}")
                for i, (model, trace) in enumerate(traces.items()):
                    mask = strip.crop((i * width, 0, (i + 1) * width, height)) if masks else None
                    images[model].set_data(draw_view(replay, frame, i, mask))
                    row = bisect_right(trace["time"], time + 1e-9) - 1
                    error = 100 * trace["position"][row]
                    errors[model].set_text(f"{error:.1f} cm error")
                    markers[model].set_data([trace["time"][row]], [error])
                row = bisect_right(times, time + 1e-9) - 1
                delta = deltas[row]
                gap_marker.set_data([times[row]], [delta])
                gap_readout.set_text(f"{round(float(delta), 1) or 0.0:+.1f} cm now  ·  Negative favors residual")
                writer.grab_frame()
                if number % (4 * FPS) == 0:
                    print(f"Encoded {number}/{len(plan)} flight frames", flush=True)

        # replacing the output after ffmpeg has finished writing the full video
        draft.replace(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Downloaded Isaac Sim job folder")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--video", action="store_true", help="Export an annotated MP4")
    mode.add_argument("--plot", action="store_true", help="Open an interactive Matplotlib report")
    parser.add_argument("--masks", type=Path, help="Use Isaac Sim camera silhouettes from this directory")
    args = parser.parse_args()
    if args.masks is not None and not args.video:
        parser.error("--masks requires --video")
    directory = args.run_directory.expanduser().resolve()
    traces, trials, duration, gusts = load_results(directory / "flight")
    print_report(trials, duration, gusts)
    if args.video:
        masks = args.masks.expanduser() if args.masks is not None else None
        replay = ReplayScene(directory / "flight" / "replay.npz")
        output = directory / "quadrotor_showcase.mp4"
        make_video(replay, masks, output, traces, trials, gusts)
        print(f"Saved {output}")
    elif args.plot:
        show_plots(traces, duration, gusts)


if __name__ == "__main__":
    main()
