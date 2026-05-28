# Bayesian Auto-Tunin for Performance Portability

This repository includes and implementation of 2D stencil and matrix multiplication (GEMM) kernels in both CUDA and SYCL, as well as an autotuning and benchmarking pipeline implemented in Python. The goal of this project is to perform autotuning (Random Search and Bayesian Optimization) on the kernels and conduct statistical experiments to determine wich tuning strategy performs the best and if there was an improvement on the performance portability of the SYCL implementation.

## How to Run the Project

### Requirements

- Python 3.8+
- CUDA Toolkit 
- Adaptivecpp SYCL toolchain 
- Python dependencies:
  - `optuna` for Bayesian inference.
  - `numpy` and `scipy` for statistical analysis.
  - `matplotlib` for plotting.

### Quickstart

Install Python dependencies:
```bash
python3 -m pip install optuna numpy matplotlib scipy
```

Run a set of autotuning experiments with the default problem size (512 x 512 x 512) and record results:

```bash
python3 run_statistical_experiments.py
```
## How to Use the Project

Change the problem size:

```bash
python3 run_statistical_experiments.py --M 2048 --N 2048 --K 2048
```

Change the configuration of the tuning experiment:

```bash
python3 run_statistical_experiments.py --num-runs 10 --trials 20 --bench-runs 10 --warmup-runs 5
```

Select the kernels:

```bash
python3 run_statistical_experiments.py --kernels matmul stencil
```
Select the backends:

```bash
python3 run_statistical_experiments.py --backends sycl cuda
```

Specify the output file:

```bash
python3 run_statistical_experiments.py --output results/convergence_analysis.json
```
