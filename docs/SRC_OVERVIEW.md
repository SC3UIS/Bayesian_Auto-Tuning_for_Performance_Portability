# Source File Overview

This document explains the purpose of each file under `src/` in the project.

## Python source

- `analyze_results.py`
  - Loads autotuning results from `results_*` directories.
  - Generates convergence plots for Bayesian vs Random search.
  - Produces SYCL vs CUDA performance visualizations for both `matmul` and `stencil`.
  - Writes CSV summaries for analysis.

- `autotune.py`
  - Implements the autotuning logic and low-level benchmark utilities.
  - Defines the search space, configuration validation, and compilation helpers.
  - Provides Bayesian and random search routines for both kernels and backends.
  - Includes benchmark timing, throughput calculation, and plausibility checks.

- `run_statistical_experiments.py`
  - Orchestrates independent autotuning experiments for statistical comparison.
  - Runs Bayesian and random search for each kernel/backend combination.
  - Saves convergence output and invokes analysis routines for significance testing.

- `statistical_analysis.py`
  - Contains statistical tools for comparing algorithm performance.
  - Computes convergence rates, Welch t-tests, Mann-Whitney U tests, and effect sizes.
  - Loads convergence data and formats statistical reports.

- `wrapper.py`
  - Provides a Python wrapper around the compiled kernel shared libraries.
  - Loads the appropriate `sycl` or `cuda` library and calls `run_kernel` with configuration parameters.
  - Is used by benchmark and autotuning code to execute the kernel variants.

## Build and orchestration files

- `autotune.sbatch`
  - SLURM batch script for submitting autotuning experiments to a GPU cluster.
  - Sets up the conda environment, output paths, and experiment parameters.
  - Iterates over kernels, backends, and problem sizes to launch `run_statistical_experiments.py`.

## Kernel sources

- `kernel_matmul_cuda.cu`
  - CUDA implementation of blocked matrix multiplication.
  - Uses compile-time tiling parameters and shared memory to evaluate different configurations.

- `kernel_matmul_sycl.cpp`
  - SYCL implementation of blocked matrix multiplication.
  - Mirrors the CUDA matmul kernel using SYCL kernels and local memory.

- `kernel_stencil_cuda.cu`
  - CUDA implementation of a tiled 5-point stencil kernel.
  - Uses shared memory and tiling parameters for the stencil update.

- `kernel_stencil_sycl.cpp`
  - SYCL implementation of the same tiled 5-point stencil kernel.
  - Uses SYCL dispatch and local accessors for stencil execution.
