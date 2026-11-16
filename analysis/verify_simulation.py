#!/usr/bin/env python3
"""Recompute archived Isaac flight metrics. Requires Python 3 and NumPy only.

Usage: python verify_simulation.py [--repo /path/to/project] [--output results]
This checks saved trajectories, not a rerun of the Isaac Sim physics engine.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def read_trials(path):
    groups = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = (float(row['speed_mps']), int(row['seed']), row['model'])
            groups.setdefault(key, []).append([float(row[c]) for c in (
                'time_s', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                'reference_x', 'reference_y', 'reference_z')])
    return {key: np.asarray(rows) for key, rows in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    repo = args.repo.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo/'quadrotor_showcase'))
    from quadrotor_control import score_flight, HOVER_GUST
    flight = repo/'isaac-11043212'/'flight'
    trials = read_trials(flight/'trajectories.csv')
    expected = {(v, s, m) for v in (3.0, 6.0) for s in (42,43,44)
                for m in ('thrust','direct','residual')}
    assert set(trials) == expected, 'Missing or unexpected trial keys'
    with (flight/'flight_metrics.csv').open(newline='') as stream:
        stored = {(float(r['speed_mps']),int(r['seed']),r['model']):r
                  for r in csv.DictReader(stream)}
    assert set(stored) == expected, 'Missing or unexpected stored score keys'
    computed = {}; discrepancy = 0.0; checks = []; windows = []
    for key, x in sorted(trials.items()):
        assert x.shape == (2881,10), f'Unexpected trial shape: {key}: {x.shape}'
        assert np.isfinite(x).all(), f'Nonfinite trajectory: {key}'
        t, p, v, target = x[:,0], x[:,1:4], x[:,4:7], x[:,7:10]
        assert np.allclose(t,np.arange(2881)/120,atol=1e-12,rtol=0), 'Unexpected timestamps'
        score = score_flight(t,p,v,target,HOVER_GUST)
        computed[key] = score
        for name,value in score.items():
            if isinstance(value,(bool,np.bool_)):
                assert (stored[key][name] == 'True') == value, f'Flag mismatch: {key}, {name}'
            else:
                discrepancy = max(discrepancy,abs(float(stored[key][name])-float(value)))
        checks.append(dict(speed_mps=key[0],seed=key[1],model=key[2],samples=len(t),
                           duration_s=float(t[-1]),min_altitude_m=float(p[:,2].min())))
        distance = np.linalg.norm(p-target,axis=1)
        for name,lo,hi in [('braking',0,3),('first_nine_seconds',0,9),
                           ('hover_before_gust',3,10),('hover_gust',10,12),('post_gust',12,24)]:
            mask = (t>=lo)&(t<=hi)
            windows.append(dict(speed_mps=key[0],seed=key[1],model=key[2],window=name,
                                samples=int(mask.sum()),rmse_m=float(np.sqrt(np.mean(distance[mask]**2)))))
    assert discrepancy < 1e-12, f'Numeric discrepancy {discrepancy} exceeds tolerance'
    numeric = [name for name in next(iter(computed.values()))
               if name not in ('settled','crossed_floor')]
    aggregate = []; paired = []
    for speed in (3.0,6.0):
        for model in ('thrust','direct','residual'):
            for metric in numeric:
                a = np.asarray([computed[speed,s,model][metric] for s in (42,43,44)])
                aggregate.append(dict(speed_mps=speed,model=model,metric=metric,n=3,
                                      mean=float(a.mean()),sample_sd=float(a.std(ddof=1))))
        for metric in numeric:
            for seed in (42,43,44):
                paired.append(dict(speed_mps=speed,seed=seed,metric=metric,
                                   residual_minus_direct=float(computed[speed,seed,'residual'][metric]
                                                               -computed[speed,seed,'direct'][metric])))
    for filename,rows in [('simulation_aggregate.csv',aggregate),
                          ('simulation_paired.csv',paired),('simulation_windows.csv',windows)]:
        with (args.output/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    inputs = [repo/'neurobem_dynamics.npz',flight/'trajectories.csv',flight/'flight_metrics.csv',
              flight/'replay.npz',repo/'quadrotor_showcase'/'quadrotor_control.py']
    hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    metadata=dict(verification='Recomputed scores from saved trajectories, physics not rerun',
                  trajectory_rows=sum(len(x) for x in trials.values()),trials=len(trials),
                  max_absolute_discrepancy=discrepancy,tolerance=1e-12,
                  all_settled=all(bool(s['settled']) for s in computed.values()),
                  any_floor_crossing=any(bool(s['crossed_floor']) for s in computed.values()),
                  summary='Sample SD is across model-training seeds, not independent plant repetitions',
                  hashes=hashes,trial_checks=checks)
    (args.output/'simulation_verification.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Verified {len(trials)} trials and {metadata["trajectory_rows"]:,} trajectory rows.')
    print(f'Maximum metric discrepancy: {discrepancy:.3g}; tolerance: 1e-12.')
    print('Wrote simulation_aggregate.csv, simulation_paired.csv, simulation_windows.csv, simulation_verification.json')


if __name__ == '__main__':
    main()
