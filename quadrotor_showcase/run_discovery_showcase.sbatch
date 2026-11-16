#!/bin/bash -l
#SBATCH --job-name=isaac-showcase
#SBATCH --partition=l40s_nova,a5500,a5500_w
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=isaac-%j.log

set -euo pipefail

nvidia-smi
# loading apptainer when it is not already available on the compute node
if ! command -v apptainer >/dev/null; then
    module load apptainer
fi

benchmark_dir="$PWD"
cache_dir="$benchmark_dir/.isaac-cache"
image="$benchmark_dir/isaac-sim-6.2.0.sif"
mkdir -p "$cache_dir"/{main,compute,kit,config,data,logs,pkg,hub}

if [[ ! -s "$image" ]]; then
    printf 'Missing Isaac Sim container: %s\n' "$image" >&2
    exit 1
fi

# mounting the project and writable caches inside the isaac sim container
isaac() {
    apptainer exec --nv --writable-tmpfs \
        --env ACCEPT_EULA=Y,PRIVACY_CONSENT=N \
        --bind "$benchmark_dir:/benchmark" \
        --bind "$cache_dir/main:/isaac-sim/.cache" \
        --bind "$cache_dir/compute:/isaac-sim/.nv/ComputeCache" \
        --bind "$cache_dir/kit:/isaac-sim/kit/cache" \
        --bind "$cache_dir/config:/isaac-sim/.nvidia-omniverse/config" \
        --bind "$cache_dir/logs:/isaac-sim/.nvidia-omniverse/logs" \
        --bind "$cache_dir/data:/isaac-sim/.local/share/ov/data" \
        --bind "$cache_dir/pkg:/isaac-sim/.local/share/ov/pkg" \
        --bind "$cache_dir/hub:/var/cache/hub" \
        --pwd /isaac-sim "$image" "$@"
}

source_job=${1:-}
if [[ $# -gt 1 || ( -n "$source_job" && ! "$source_job" =~ ^[0-9]+$ ) ]]; then
    printf 'Usage: sbatch quadrotor_showcase/run_discovery_showcase.sbatch [PHYSICS_JOB_ID]\n' >&2
    exit 1
fi

# running physics for a new job or loading the saved poses from an earlier one
scripts="/benchmark/quadrotor_showcase"
if [[ -z "$source_job" ]]; then
    run_dir="/benchmark/results/isaac-$SLURM_JOB_ID"
    output_dir="$run_dir/render"
    isaac /isaac-sim/python.sh "$scripts/run_isaac.py" --long-run \
        --checkpoint /benchmark/neurobem_dynamics.npz --output "$run_dir/flight"
else
    run_dir="/benchmark/results/isaac-$source_job"
    output_dir="$run_dir/showcase-$SLURM_JOB_ID"
fi
# capturing camera masks from the saved poses for the optional video export
isaac /isaac-sim/python.sh "$scripts/render_showcase.py" "$run_dir/flight/replay.npz" "$output_dir"
