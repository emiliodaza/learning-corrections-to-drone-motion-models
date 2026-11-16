#!/usr/bin/env python3
"""Plot saved 6 m/s, seed-42 simulation tracking errors with gust intervals.

Usage: python plot_simulation.py [--repo /path/to/project] [--output figures]
Requires NumPy and Matplotlib. Charts are computed from saved trajectories.
"""
import argparse
from pathlib import Path
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'cs74-matplotlib'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from verify_simulation import read_trials


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'report'/'figures')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=read_trials(args.repo/'isaac-11043212'/'flight'/'trajectories.csv')
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],
                         'mathtext.fontset':'stix','font.size':10,
                         'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
                         'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    style={'thrust':('Thrust','#777C82',':'),
           'direct':('Direct NN','#17365D','-'),
           'residual':('Residual NN','#008577','--')}
    fig,ax=plt.subplots(figsize=(6.8,2.55))
    for model in style:
        label,color,ls=style[model];x=data[6.0,42,model]
        error=np.linalg.norm(x[:,1:4]-x[:,7:10],axis=1)
        ax.plot(x[:,0],error,label=label,color=color,linestyle=ls,linewidth=1.6)
    for lo,hi,text in [(1,2,'+2 m/s'),(10,12,'−3 m/s')]:
        ax.axvspan(lo,hi,color='#DFE6EB',alpha=.72,zorder=0)
        ax.text((lo+hi)/2,1.02,text,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=8.5,color='#4E5964')
    ax.axvline(3,color='#ADB5BD',linewidth=.8,linestyle='--',zorder=0)
    ax.set(xlim=(0,24),ylim=(0,None),xlabel='Time (s)',ylabel='3D position error (m)')
    ax.set_xticks(np.arange(0,25,3))
    ax.grid(axis='y',color='#DADEE2',linewidth=.5)
    ax.legend(frameon=False,ncol=3,loc='upper right',bbox_to_anchor=(1.015,1.28),handlelength=2.5,columnspacing=1.3)
    fig.subplots_adjust(left=.095,bottom=.20,right=.985,top=.80)
    fig.savefig(args.output/'simulation_tracking.pdf')
    fig.savefig(args.output/'simulation_tracking.png',dpi=220)
    plt.close(fig)
    print('Wrote simulation_tracking.pdf and simulation_tracking.png')


if __name__=='__main__':
    main()
