#!/usr/bin/env python3
"""Re-evaluate supplied NeuroBEM model weights without retraining the networks.

Inputs are the official processed_data.zip and testset.txt plus the supplied
neurobem_dynamics.npz checkpoint. NumPy implements the exported two-layer MLP.
This script also refits OLS, the three ridge candidates, and the thrust baseline
on the training partition. It deliberately leaves neural-network training alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


VELOCITY = ["vel x", "vel y", "vel z"]
RATES = ["ang vel x", "ang vel y", "ang vel z"]
MOTORS = ["mot 1", "mot 2", "mot 3", "mot 4"]
TARGETS = ["acc x", "acc y", "acc z"]
KEEP = ["t", *VELOCITY, *RATES, *MOTORS, *TARGETS]
SEEDS = (42, 43, 44)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flight_id(filename):
    return Path(filename).stem.removeprefix("merged_").split("_seg_", 1)[0]


def features(frame):
    motors = frame[MOTORS].to_numpy(dtype=float) ** 2
    return (
        np.column_stack((frame[VELOCITY + RATES], motors)),
        frame[TARGETS].to_numpy(dtype=float),
        motors.sum(axis=1),
    )


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def scores(actual, predicted):
    result = {"RMSE": rmse(actual, predicted), "MAE": float(np.abs(actual - predicted).mean())}
    for axis, name in enumerate("xyz"):
        result[f"RMSE {name}"] = rmse(actual[:, axis], predicted[:, axis])
        result[f"MAE {name}"] = float(np.abs(actual[:, axis] - predicted[:, axis]).mean())
    return result


def network(checkpoint, name, seed, scaled):
    x = scaled.astype(np.float32)
    hidden = np.maximum(x @ checkpoint[f"{name}_{seed}_0.weight"].T + checkpoint[f"{name}_{seed}_0.bias"], 0)
    return hidden @ checkpoint[f"{name}_{seed}_2.weight"].T + checkpoint[f"{name}_{seed}_2.bias"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    ckpt = dict(np.load(args.checkpoint))

    with ZipFile(args.archive) as archive:
        files = sorted(name for name in archive.namelist() if name.endswith(".csv") and not name.startswith("__MACOSX/"))
        splits = pd.DataFrame({"file": files})
        splits["flight_id"] = splits["file"].map(flight_id)
        all_ids = set(splits["flight_id"])
        test_ids = {flight_id(line.strip()) for line in args.test_list.read_text().splitlines() if line.strip()}
        assert test_ids and test_ids <= all_ids
        remainder = np.array(sorted(all_ids - test_ids))
        # sklearn train_test_split uses ShuffleSplit with ceil(test_size*n).
        order = np.random.RandomState(42).permutation(len(remainder))
        n_val = math.ceil(0.2 * len(remainder))
        val_ids = set(remainder[order[:n_val]])
        train_ids = set(remainder[order[n_val:]])
        assignment = dict.fromkeys(train_ids, "train")
        assignment.update(dict.fromkeys(val_ids, "validation"))
        assignment.update(dict.fromkeys(test_ids, "test"))
        splits["split"] = splits["flight_id"].map(assignment)
        assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
        splits.to_csv(output / "flight_splits.csv", index=False)

        frames = {name: [] for name in ("train", "validation", "test")}
        audit = []
        for row in splits.itertuples(index=False):
            with archive.open(row.file) as stream:
                original = pd.read_csv(stream)
            original.columns = original.columns.str.strip().str.lstrip("#").str.strip()
            if set(KEEP) - set(original):
                raise ValueError(f"Missing fields in {row.file}")
            used = original[KEEP]
            dt = np.diff(used["t"].to_numpy(dtype=float))
            positive_dt = dt[np.isfinite(dt) & (dt > 0)]
            audit.append({
                "file": row.file, "rows": len(original),
                "missing_values": int(used.isna().sum().sum()),
                "nonfinite_rows": int((~np.isfinite(used.to_numpy(dtype=float)).all(axis=1)).sum()),
                "duplicate_rows": int(original.duplicated().sum()),
                "repeated_timestamps": int(used["t"].dropna().duplicated().sum()),
                "backward_time_steps": int((dt < 0).sum()),
                "median_positive_dt_s": float(np.median(positive_dt)) if len(positive_dt) else None,
            })
            frame = used.iloc[::8].copy()
            if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite retained values in {row.file}")
            frame["flight_id"] = row.flight_id
            frame["segment"] = row.file
            frames[row.split].append(frame)
        frames = {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}
    pd.DataFrame(audit).to_csv(output / "data_audit.csv", index=False)

    arrays = {name: features(frame) for name, frame in frames.items()}
    x_train, y_train, s_train = arrays["train"]
    fitted_mean = x_train.mean(axis=0)
    fitted_scale = x_train.std(axis=0)
    np.testing.assert_allclose(fitted_mean, ckpt["scaler_mean"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fitted_scale, ckpt["scaler_scale"], rtol=1e-10, atol=1e-10)
    coefficient = float((s_train @ y_train[:, 2]) / (s_train @ s_train))
    np.testing.assert_allclose(coefficient, ckpt["thrust_coefficient"], rtol=1e-12)
    scaled = {name: (data[0] - ckpt["scaler_mean"]) / ckpt["scaler_scale"] for name, data in arrays.items()}
    nominal = {name: np.column_stack((np.zeros_like(data[2]), np.zeros_like(data[2]), coefficient * data[2])) for name, data in arrays.items()}

    # Match sklearn's default fit_intercept=True by centering X and Y.
    x_mean = scaled["train"].mean(axis=0)
    y_mean = y_train.mean(axis=0)
    x_centered = scaled["train"] - x_mean
    y_centered = y_train - y_mean
    ols_w = np.linalg.lstsq(x_centered, y_centered, rcond=None)[0]
    gram, response = x_centered.T @ x_centered, x_centered.T @ y_centered
    ridge_weights = {alpha: np.linalg.solve(gram + alpha * np.eye(10), response) for alpha in (1, 100, 10000)}
    ridge_validation = {alpha: rmse(arrays["validation"][1], (scaled["validation"] - x_mean) @ weight + y_mean) for alpha, weight in ridge_weights.items()}
    alpha = min(ridge_validation, key=ridge_validation.get)
    pd.DataFrame([{"alpha": a, "validation_rmse": v} for a, v in ridge_validation.items()]).to_csv(output / "ridge_validation.csv", index=False)
    np.savez_compressed(output / "linear_models.npz", ols_weight=ols_w, ridge_weight=ridge_weights[alpha], x_mean=x_mean, y_mean=y_mean, ridge_alpha=alpha)

    actual = arrays["test"][1]
    test_frame = frames["test"]
    prediction = {
        "Thrust": {None: nominal["test"]},
        "Linear regression": {None: (scaled["test"] - x_mean) @ ols_w + y_mean},
        "Ridge": {None: (scaled["test"] - x_mean) @ ridge_weights[alpha] + y_mean},
        "Direct NN": {seed: network(ckpt, "direct", seed, scaled["test"]) for seed in SEEDS},
        "Residual NN": {seed: nominal["test"] + network(ckpt, "residual", seed, scaled["test"]) for seed in SEEDS},
    }
    test_by_run = pd.DataFrame([{"Model": model, "Seed": seed, **scores(actual, values)} for model, runs in prediction.items() for seed, values in runs.items()])
    test_by_run.to_csv(output / "test_metrics_by_run.csv", index=False)
    grouped = test_by_run.drop(columns="Seed").groupby("Model", sort=False)
    metrics = grouped.mean()
    metrics["RMSE SD"] = grouped["RMSE"].std()
    metrics.to_csv(output / "test_metrics.csv")

    validation_by_run = []
    for model, prefix in [("Direct NN", "direct"), ("Residual NN", "residual")]:
        for seed in SEEDS:
            values = network(ckpt, prefix, seed, scaled["validation"])
            if prefix == "residual":
                values = values + nominal["validation"]
            validation_by_run.append({"Model": model, "Seed": seed, "RMSE": rmse(arrays["validation"][1], values)})
    pd.DataFrame(validation_by_run).to_csv(output / "validation_checkpoint_metrics.csv", index=False)

    speed = np.linalg.norm(test_frame[VELOCITY].to_numpy(dtype=float), axis=1)
    ids = test_frame["flight_id"].to_numpy()
    fast = speed >= 5
    speed_rows = []
    for label, mask in [("< 5 m/s", ~fast), (">= 5 m/s", fast)]:
        for model, runs in prediction.items():
            values = np.array([rmse(actual[mask], estimate[mask]) for estimate in runs.values()])
            speed_rows.append({"Speed group": label, "Model": model, "Observations": int(mask.sum()), "Flights": len(np.unique(ids[mask])), "RMSE": values.mean(), "RMSE SD": values.std(ddof=1) if len(values)>1 else np.nan})
    pd.DataFrame(speed_rows).to_csv(output / "metrics_by_speed.csv", index=False)
    flight_rows = []
    for identity in np.unique(ids):
        for label, mask in [("all", ids == identity), (">= 5 m/s", (ids == identity) & fast)]:
            if not mask.any():
                continue
            for model, runs in prediction.items():
                values = [rmse(actual[mask], estimate[mask]) for estimate in runs.values()]
                flight_rows.append({"Flight ID": identity, "Speed group": label, "Model": model, "Observations": int(mask.sum()), "RMSE": np.mean(values)})
    flight_metrics = pd.DataFrame(flight_rows)
    flight_metrics.to_csv(output / "metrics_by_flight.csv", index=False)
    paired = flight_metrics[flight_metrics["Speed group"] == ">= 5 m/s"].pivot(index="Flight ID", columns="Model", values="RMSE")
    paired["Residual minus direct"] = paired["Residual NN"] - paired["Direct NN"]
    paired.to_csv(output / "paired_high_speed_flights.csv")
    delta = paired["Residual minus direct"].to_numpy()
    non_ties = delta[~np.isclose(delta, 0, rtol=0, atol=1e-12)]
    wins = int((non_ties < 0).sum())
    # Two-sided exact binomial test for p=.5. Symmetric equal-tail form.
    tail = sum(math.comb(len(non_ties), k) for k in range(min(wins, len(non_ties)-wins)+1)) / 2**len(non_ties)
    p_value = min(1.0, 2 * tail)
    bootstrap = np.random.default_rng(42).choice(delta, size=(20_000, len(delta)), replace=True).mean(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975])
    diagnostics = {"paired_flights": len(delta), "residual_wins": wins, "sign_test_p": p_value, "mean_delta": float(delta.mean()), "bootstrap_95_CI": interval.tolist(), "bootstrap_resamples": 20000, "bootstrap_seed": 42}
    (output / "paired_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    np.savez_compressed(output / "verified_test_predictions.npz", actual=actual, speed=speed, flight_id=ids.astype(str), segment=test_frame["segment"].to_numpy(dtype=str), timestamps=test_frame["t"].to_numpy(), **{model.replace(" ", "_") + "_" + str(seed): values for model, runs in prediction.items() for seed, values in runs.items()})

    provenance = {
        "purpose": "Re-evaluation of supplied neural weights, with freshly fitted linear baselines; no neural retraining",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "input_sha256": {"archive": sha256(args.archive), "test_list": sha256(args.test_list), "checkpoint": sha256(args.checkpoint)},
        "rows": {name: len(frame) for name, frame in frames.items()},
        "flights": {name: int(splits.loc[splits["split"]==name, "flight_id"].nunique()) for name in frames},
        "segments": {name: int((splits["split"]==name).sum()) for name in frames},
        "thrust_coefficient": coefficient, "ridge_alpha": alpha,
        "scaler_max_abs_mean_difference": float(np.max(np.abs(fitted_mean-ckpt["scaler_mean"]))),
        "scaler_max_abs_scale_difference": float(np.max(np.abs(fitted_scale-ckpt["scaler_scale"]))),
    }
    (output / "verification_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(metrics.round(6).to_string())
    print(pd.DataFrame(speed_rows).round(6).to_string(index=False))
    print(json.dumps(diagnostics, indent=2))
    print(json.dumps(provenance, indent=2))
    if args.plots:
        make_plots(output, actual, prediction, test_frame, speed, flight_metrics, paired)


def make_plots(output, actual, prediction, frame, speed, flight_metrics, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    ids = frame["flight_id"].to_numpy()
    segments = frame["segment"].to_numpy()
    timestamps = frame["t"].to_numpy()
    residual = prediction["Residual NN"]
    ranking = flight_metrics[(flight_metrics["Speed group"]=="all") & (flight_metrics["Model"]=="Residual NN")].sort_values("RMSE", ascending=False)
    plot_map = []
    for rank, row in enumerate(ranking.head(2).itertuples(index=False), start=1):
        identity = row[0]
        in_flight = ids == identity
        segment = max(np.unique(segments[in_flight]), key=lambda seg: np.mean([rmse(actual[in_flight & (segments==seg)], estimate[in_flight & (segments==seg)]) for estimate in residual.values()]))
        indices = np.flatnonzero(in_flight & (segments==segment))
        indices = indices[np.argsort(timestamps[indices], kind="stable")]
        time = timestamps[indices]
        fig, axes = plt.subplots(4, 1, figsize=(7, 5.5), sharex=True, constrained_layout=True)
        for axis, ax in enumerate(axes[:3]):
            ax.plot(time, actual[indices, axis], color="#2D3035", linewidth=1.05, label="Recorded")
            ax.plot(time, prediction["Direct NN"][42][indices, axis], color="#256D9C", linewidth=.9, label="Direct NN")
            ax.plot(time, residual[42][indices, axis], color="#B85524", linestyle="--", linewidth=.9, label="Residual NN")
            ax.set_ylabel(f"$a_{'xyz'[axis]}$ (m/s²)")
            ax.set_title(f"RMSE: direct {rmse(actual[indices,axis],prediction['Direct NN'][42][indices,axis]):.3f}, residual {rmse(actual[indices,axis],residual[42][indices,axis]):.3f} m/s²", loc="left", fontsize=8)
        axes[0].legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
        axes[3].plot(time, speed[indices], color="#37614F", linewidth=1)
        axes[3].axhline(5, color="0.55", linestyle=":", linewidth=.8)
        axes[3].set_ylabel("Speed (m/s)")
        axes[3].set_xlabel("Recorded time (s)")
        for ax in axes:
            ax.grid(alpha=.16)
        segment_number = segment.split("_seg_", 1)[-1].removesuffix(".csv")
        fig.suptitle(f"Flight {identity}, segment {segment_number}, seed 42", fontsize=10)
        fig.savefig(output / f"flight_comparison_{rank}.png", dpi=300)
        plt.close(fig)
        plot_map.append({"rank": rank, "flight_id": identity, "segment": segment, "mean_flight_residual_rmse": float(row[-1])})
    (output / "diagnostic_plot_map.json").write_text(json.dumps(plot_map, indent=2) + "\n")
    ordered = paired.sort_values("Residual minus direct")
    fig, ax = plt.subplots(figsize=(6.4, 3.1), constrained_layout=True)
    labels = [f"{i+1:02d}" for i in range(len(ordered))]
    values = ordered["Residual minus direct"].to_numpy()
    ax.barh(labels, values, color=["#B85524" if v<0 else "#256D9C" for v in values])
    ax.axvline(0, color="0.3", linewidth=.8)
    ax.set_xlabel("High-speed RMSE change: residual minus direct (m/s²)")
    ax.set_ylabel("Held-out flight (ordered by change)")
    ax.grid(axis="x", alpha=.15)
    fig.savefig(output / "paired_high_speed.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
