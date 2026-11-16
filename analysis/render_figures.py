#!/usr/bin/env python3
"""Render report diagnostics from the independently verified prediction arrays.

Requires NumPy and Matplotlib. Run after reproduce_checkpoint.py. This does not
train any models or select any model settings.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual-predicted)**2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--figures", type=Path, default=Path(__file__).resolve().parents[1] / "report" / "figures")
    args = parser.parse_args()
    args.figures.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "axes.titlesize":9, "axes.labelsize":8, "axes.spines.top":False, "axes.spines.right":False, "savefig.facecolor":"white"})
    z = dict(np.load(args.results / "verified_test_predictions.npz"))
    actual, speed, ids, segs, times = (z[key] for key in ("actual", "speed", "flight_id", "segment", "timestamps"))
    direct = {s:z[f"Direct_NN_{s}"] for s in (42,43,44)}
    residual = {s:z[f"Residual_NN_{s}"] for s in (42,43,44)}
    ranking = sorted(np.unique(ids), key=lambda identity:np.mean([rmse(actual[ids==identity],a[ids==identity]) for a in residual.values()]), reverse=True)
    selected = []
    for identity in ranking[:2]:
        in_flight = ids==identity
        segment = max(np.unique(segs[in_flight]), key=lambda seg:np.mean([rmse(actual[in_flight & (segs==seg)],a[in_flight & (segs==seg)]) for a in residual.values()]))
        indices = np.flatnonzero(in_flight & (segs==segment))
        indices = indices[np.argsort(times[indices],kind="stable")]
        selected.append((identity,segment,indices))
    plot_map=[]
    for rank,(identity,segment,rows) in enumerate(selected,start=1):
        time=times[rows]
        fig,axes=plt.subplots(4,1,figsize=(7,5.5),sharex=True,layout="constrained")
        for axis,ax in enumerate(axes[:3]):
            ax.plot(time,actual[rows,axis],color="#2D3035",lw=1,label="Recorded")
            ax.plot(time,direct[42][rows,axis],color="#256D9C",lw=.85,label="Direct NN")
            ax.plot(time,residual[42][rows,axis],color="#B85524",lw=.85,ls="--",label="Residual NN")
            ax.set_ylabel(f"$a_{'xyz'[axis]}$ (m/s²)")
            ax.set_title(f"RMSE: direct {rmse(actual[rows,axis],direct[42][rows,axis]):.3f}, residual {rmse(actual[rows,axis],residual[42][rows,axis]):.3f} m/s²",loc="left",fontsize=8)
        axes[0].legend(ncol=3,frameon=False,loc="upper right",fontsize=8)
        axes[3].plot(time,speed[rows],color="#37614F",lw=1)
        axes[3].axhline(5,color="0.5",ls=":",lw=.8)
        axes[3].set_ylabel("Speed (m/s)")
        axes[3].set_xlabel("Recorded time (s)")
        for ax in axes:ax.grid(alpha=.16)
        number=segment.split("_seg_",1)[-1].removesuffix(".csv")
        fig.suptitle(f"Flight {identity}, segment {number}, seed 42",fontsize=10)
        fig.savefig(args.figures/f"flight_comparison_{rank}.png",dpi=300)
        plt.close(fig)
        flight_mask = ids == identity
        plot_map.append({"rank":rank,"flight_id":identity,"segment":segment,"mean_flight_residual_rmse":float(np.mean([rmse(actual[flight_mask],a[flight_mask]) for a in residual.values()]))})
    fig,axes=plt.subplots(2,2,figsize=(7.1,3.1),height_ratios=[1.7,1],layout="constrained")
    for column,(identity,segment,rows) in enumerate(selected):
        time=times[rows]
        ax=axes[0,column]
        ax.plot(time,actual[rows,2],color="#2D3035",lw=.8,label="Recorded")
        ax.plot(time,direct[42][rows,2],color="#256D9C",lw=.8,label="Direct NN")
        ax.plot(time,residual[42][rows,2],color="#B85524",lw=.8,ls="--",label="Residual NN")
        number=segment.split("_seg_",1)[-1].removesuffix(".csv")
        ax.set_title(f"{identity}, segment {number}\n$z$ RMSE: direct {rmse(actual[rows,2],direct[42][rows,2]):.3f}, residual {rmse(actual[rows,2],residual[42][rows,2]):.3f}",fontsize=8)
        ax.set_ylabel("$a_z$ (m/s²)")
        axes[1,column].plot(time,speed[rows],color="#37614F",lw=.9)
        axes[1,column].axhline(5,color="0.45",ls=":",lw=.8)
        axes[1,column].set_ylabel("Speed (m/s)")
        axes[1,column].set_xlabel("Recorded time (s)")
        axes[1,column].set_xlim(ax.get_xlim())
        for item in axes[:,column]:item.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=3,frameon=False,fontsize=8)
    fig.savefig(args.figures/"high_error_compact.png",dpi=300)
    plt.close(fig)
    (args.results/"diagnostic_plot_map.json").write_text(json.dumps(plot_map,indent=2)+"\n")
    with (args.results/"paired_high_speed_flights.csv").open() as stream:
        pairs=sorted(csv.DictReader(stream),key=lambda r:float(r["Residual minus direct"]))
    values=np.array([float(row["Residual minus direct"]) for row in pairs])
    fig,ax=plt.subplots(figsize=(6.4,3.1),layout="constrained")
    ax.barh([f"{i+1:02d}" for i in range(len(values))],values,color=["#B85524" if value<0 else "#256D9C" for value in values])
    ax.axvline(0,color="0.3",lw=.8)
    ax.set_xlabel("High-speed RMSE change: residual minus direct (m/s²)")
    ax.set_ylabel("Held-out flight (ordered by change)")
    ax.grid(axis="x",alpha=.15)
    fig.savefig(args.figures/"paired_high_speed.png",dpi=300)
    plt.close(fig)
    print("Created diagnostic figures:",*[str(path) for path in sorted(args.figures.glob("*.png"))],sep="\n")


if __name__=="__main__":main()
