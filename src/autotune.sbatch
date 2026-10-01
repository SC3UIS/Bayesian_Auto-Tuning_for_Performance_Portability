#!/bin/bash
#SBATCH --job-name=kernel_autotune
#SBATCH --output=autotune_%j.out
#SBATCH --error=autotune_%j.err
#SBATCH --partition=GPU                
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00               
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -Eeuo pipefail

on_error() {
    local status=$?
    echo "ERROR: job stopped at line ${BASH_LINENO[0]} with exit code ${status} ($(date))" >&2
    exit "$status"
}
trap on_error ERR

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module load devtools/nvidia/hpc_sdk/nvhpc/23.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hpc-gemm

REPO_DIR="$HOME/Bayesian_Auto-Tuning_for_Performance_Portability"             
cd "$REPO_DIR/src" || { echo "ERROR: source dir not found"; exit 1; }

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$REPO_DIR/results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR" "$REPO_DIR/logs"

echo "======================================================================"
echo "Kernel Autotuning Experiment"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Results dir: $RESULTS_DIR"
echo "Started    : $(date)"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Experiment knobs. Override when submitting, e.g.:
#   sbatch --mail-user=you@example.edu --export=ALL,KERNELS="matmul stencil",BACKENDS="sycl cuda",TRIALS=15 autotune.sbatch
# ---------------------------------------------------------------------------
read -r -a KERNEL_LIST <<< "${KERNELS:-matmul stencil}"
read -r -a BACKEND_LIST <<< "${BACKENDS:-sycl cuda}"
read -r -a SIZE_LIST <<< "${SIZES:-1024 2048 4096}"

INDEPENDENT_RUNS="${INDEPENDENT_RUNS:-10}"
TRIALS="${TRIALS:-15}"
BENCH_RUNS="${BENCH_RUNS:-10}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
VALIDATION_RUNS="${VALIDATION_RUNS:-20}"
GRID_RUNS="${GRID_RUNS:-3}"
GRID_WARMUP_RUNS="${GRID_WARMUP_RUNS:-1}"
GRID_SEARCH="${GRID_SEARCH:-1}"
STENCIL_ITERATIONS="${STENCIL_ITERATIONS:-512}"

echo "Kernels    : ${KERNEL_LIST[*]}"
echo "Backends   : ${BACKEND_LIST[*]}"
echo "Sizes      : ${SIZE_LIST[*]}"
echo "Runs       : $INDEPENDENT_RUNS independent runs"
echo "Trials     : $TRIALS per search method/run"
echo "Bench runs : $BENCH_RUNS timed repeats per configuration"
echo "Warmups    : $WARMUP_RUNS untimed repeats per configuration"
echo "Validation : $VALIDATION_RUNS independent repeats per configuration"
echo "Grid search: $GRID_SEARCH ($GRID_RUNS repeats, $GRID_WARMUP_RUNS warmups)"
echo "Stencil K  : $STENCIL_ITERATIONS iterations"

# ---------------------------------------------------------------------------
# Run pipeline for each size
# ---------------------------------------------------------------------------
for KERNEL in "${KERNEL_LIST[@]}"; do
    for SIZE in "${SIZE_LIST[@]}"; do
        SIZE_DIR="$RESULTS_DIR/${KERNEL}/size_${SIZE}"
        mkdir -p "$SIZE_DIR"

        if [ "$KERNEL" = "stencil" ]; then
            K_VALUE="$STENCIL_ITERATIONS"
        else
            K_VALUE="$SIZE"
        fi

        echo ""
        echo "======================================================================"
        echo "Kernel: ${KERNEL} | Problem: ${SIZE}x${SIZE}x${K_VALUE}   $(date)"
        echo "======================================================================"

        GRID_ARGS=()
        if [ "$GRID_SEARCH" != "1" ]; then
            GRID_ARGS+=(--skip-grid-search)
        fi

        python3 -u run_statistical_experiments.py \
            --kernel "$KERNEL" \
            --M "$SIZE" \
            --N "$SIZE" \
            --K "$K_VALUE" \
            --num-runs "$INDEPENDENT_RUNS" \
            --trials "$TRIALS" \
            --bench-runs "$BENCH_RUNS" \
            --warmup-runs "$WARMUP_RUNS" \
            --validation-runs "$VALIDATION_RUNS" \
            --grid-runs "$GRID_RUNS" \
            --grid-warmup-runs "$GRID_WARMUP_RUNS" \
            --backends "${BACKEND_LIST[@]}" \
            "${GRID_ARGS[@]}" \
            --output "$SIZE_DIR/convergence.json" \
            2>&1 | tee "$SIZE_DIR/run.log"

        echo "Completed ${KERNEL} size ${SIZE}   $(date)"
    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "All sizes finished: $(date)"
echo "Results saved to  : $RESULTS_DIR"
echo ""
echo "Per-size layout:"
for KERNEL in "${KERNEL_LIST[@]}"; do
    for SIZE in "${SIZE_LIST[@]}"; do
        echo "  $RESULTS_DIR/${KERNEL}/size_${SIZE}/"
        echo "    convergence_sycl.json / convergence_cuda.json — backend-specific curves"
        echo "    statistical_results_sycl.json / statistical_results_cuda.json — backend-specific tests"
        echo "    run.log                — full stdout/stderr for debugging"
    done
done
echo "======================================================================"
