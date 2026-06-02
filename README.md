# Bayesian Auto-Tuning for Performance Portability: A Comparative Analysis of SYCL and CUDA Kernels

**Repository for the academic paper on Bayesian Optimization-driven auto-tuning of GPU kernels for performance portability between CUDA and SYCL.**

---

## Overview

This repository contains the implementation and experimental pipeline for comparative analysis of CUDA and SYCL kernel performance, with automated tuning via Bayesian Optimization (BO) and Random Search (RS). The project evaluates:

- **Performance Portability**: Quantifying the cost of porting from vendor-specific CUDA to open-standard SYCL
- **Auto-tuning Strategies**: Comparing Bayesian Optimization (TPE algorithm) vs. Random Search
- **Computational Kernels**: General Matrix Multiplication (GEMM) and 2D Stencil operations
---

## Authors 

- **Juan F. Rojas de la H.** 
- **Diego A. Arévalo Q.** 
- **Sergio A. Gélvez C.** 
- **Luis A. Torres N.** 
- **Carlos J. Barrios H.** 
---

## Quick Start

### Prerequisites

**System Requirements:**
- Linux operating system 
- NVIDIA GPU with compute capability 8.0+ 
- 40 GB+ GPU memory for largest problem sizes
- 16 GB+ CPU RAM for analysis

**Software Stack:**
- Python 3.8+
- CUDA Toolkit 11.8+
- AdaptiveCpp (SYCL toolchain)
- GCC 9.0+ with C++17 support

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/SC3UIS/Bayesian_Auto-Tuning_for_Performance_Portability.git
cd Bayesian_Auto-Tuning_for_Performance_Portability
```

2. **Install Python dependencies:**
```bash
python3 -m pip install optuna numpy scipy matplotlib pandas
```

3. **Verify CUDA and SYCL toolchains:**
```bash
nvcc --version
acpp --version
```

4. **Compile kernels (optional - auto-compilation during execution):**
```bash
cd src
make clean
make
```

---

## Usage Guide

### Running Auto-Tuning Experiments

#### Basic Execution (Default Settings)
```bash
cd src
python3 run_statistical_experiments.py
```

This runs:
- Problem sizes: 1024³, 2048³, 4096³ (GEMM) / 1024² × 512, 2048² × 512, 4096² × 512 (Stencil)
- Kernels: Both GEMM and Stencil
- Backends: Both CUDA and SYCL
- Tuning budget: 20 trials per scenario
- Statistical runs: 10 independent trials with different random seeds

#### Customized Execution

**Change problem sizes:**
```bash
python3 run_statistical_experiments.py --M 2048 --N 2048 --K 2048
```

**Select specific kernels:**
```bash
python3 run_statistical_experiments.py --kernels matmul stencil
```

**Select specific backends:**
```bash
python3 run_statistical_experiments.py --backends cuda sycl
```

**Adjust tuning configuration:**
```bash
python3 run_statistical_experiments.py \
  --num-runs 10 \
  --trials 20 \
  --bench-runs 10 \
  --warmup-runs 5
```

**Specify custom output directory:**
```bash
python3 run_statistical_experiments.py --output /path/to/results
```

**Full customization example:**
```bash
python3 run_statistical_experiments.py \
  --kernels matmul \
  --backends cuda sycl \
  --M 4096 --N 4096 --K 4096 \
  --num-runs 5 \
  --trials 30 \
  --output ./custom_results
```

### Post-Processing and Analysis

**Generate statistical analysis:**
```bash
python3 statistical_analysis.py \
  --input /path/to/results \
  --output /path/to/analysis
```

**Create visualizations:**
```bash
python3 analyze_results.py \
  --input /path/to/results \
  --plots convergence efficiency speedup
```

---

## Results and Data

The `data/` directory contains:

- **Convergence Analysis**: Tracking optimization algorithm progress across iterations
- **Statistical Results**: JSON files with execution times and efficiency metrics
- **Performance Summaries**: CSV exports of key performance indicators
- **Speedup Tables**: Comparative performance metrics (tuned vs. baseline)

### Exploring Results

All experimental data is organized by timestamp. To analyze the latest results:

```bash
cd data/results_20260522_152743
ls -la
```
---

## Documentation

Comprehensive documentation is available in the `docs/` directory:

## Acknowledgments

We gratefully acknowledge:
- **CAGE Research Group** for research support and guidance
- **Universidad Industrial de Santander** for computational resources via the GUANE cluster
- **Universidad de Cartagena** for access to the PACCA supercomputing infrastructure

---
