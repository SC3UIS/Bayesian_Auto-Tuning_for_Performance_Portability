#!/bin/bash
#SBATCH --job-name=kernel_autotune
#SBATCH --output=logs/autotune_%j.out
#SBATCH --error=logs/autotune_%j.err
#SBATCH --partition=GPU                
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00               

set -o pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hpc-gemm

REPO_DIR="$HOME/hpc-gemm"             
cd "$REPO_DIR" || { echo "ERROR: repo dir not found"; exit 1; }

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$REPO_DIR/results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR" logs

echo "======================================================================"
echo "Kernel Autotuning Experiment"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Results dir: $RESULTS_DIR"
echo "Started    : $(date)"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Experiment knobs. Override when submitting, e.g.:
#   sbatch --export=ALL,KERNELS="matmul stencil",BACKENDS="sycl cuda",TRIALS=100 autotune.sbatch
# ---------------------------------------------------------------------------
read -r -a KERNEL_LIST <<< "${KERNELS:-matmul stencil}"
read -r -a BACKEND_LIST <<< "${BACKENDS:-sycl cuda}"
read -r -a SIZE_LIST <<< "${SIZES:-512 1024 2048}"

INDEPENDENT_RUNS="${INDEPENDENT_RUNS:-10}"
TRIALS="${TRIALS:-100}"
BENCH_RUNS="${BENCH_RUNS:-10}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
STENCIL_ITERATIONS="${STENCIL_ITERATIONS:-512}"

echo "Kernels    : ${KERNEL_LIST[*]}"
echo "Backends   : ${BACKEND_LIST[*]}"
echo "Sizes      : ${SIZE_LIST[*]}"
echo "Runs       : $INDEPENDENT_RUNS independent runs"
echo "Trials     : $TRIALS per search method/run"
echo "Bench runs : $BENCH_RUNS timed repeats per configuration"
echo "Warmups    : $WARMUP_RUNS untimed repeats per configuration"
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

        python3 -u run_statistical_experiments.py \
            --kernel "$KERNEL" \
            --M "$SIZE" \
            --N "$SIZE" \
            --K "$K_VALUE" \
            --num-runs "$INDEPENDENT_RUNS" \
            --trials "$TRIALS" \
            --bench-runs "$BENCH_RUNS" \
            --warmup-runs "$WARMUP_RUNS" \
            --backends "${BACKEND_LIST[@]}" \
            --output "$SIZE_DIR/convergence.json" \
            2>&1 | tee "$SIZE_DIR/run.log"

        EXIT_CODE=${PIPESTATUS[0]}

        if [ $EXIT_CODE -ne 0 ]; then
            echo "WARNING: ${KERNEL} size ${SIZE} exited with code $EXIT_CODE - continuing"
        else
            echo "Completed ${KERNEL} size ${SIZE}   $(date)"
        fi
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
