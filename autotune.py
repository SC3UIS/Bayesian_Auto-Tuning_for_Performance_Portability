#!/usr/bin/env python3
import subprocess
import json
import sys
import re
import os
import random
import math
import statistics
from pathlib import Path

try:
    import optuna
except ImportError:
    optuna = None

DEFAULT_M, DEFAULT_N, DEFAULT_K = 512, 512, 512
DEFAULT_SEED = 42
MAX_PLAUSIBLE_GFLOPS = float(os.environ.get("MAX_PLAUSIBLE_GFLOPS", "200000"))
CUDA_MATMUL_MAX_THREADS_PER_BLOCK = int(os.environ.get("CUDA_MATMUL_MAX_THREADS_PER_BLOCK", "512"))
CUDA_STENCIL_MAX_THREADS_PER_BLOCK = int(os.environ.get("CUDA_STENCIL_MAX_THREADS_PER_BLOCK", "512"))
SYCL_MATMUL_MAX_WORK_GROUP_SIZE = int(os.environ.get("SYCL_MATMUL_MAX_WORK_GROUP_SIZE", "1024"))
SYCL_STENCIL_MAX_WORK_GROUP_SIZE = int(os.environ.get("SYCL_STENCIL_MAX_WORK_GROUP_SIZE", "1024"))

search_spaces = {
    "matmul": {
        "BM": [32, 64, 128],
        "BN": [32, 64, 128],
        "BK": [8, 16, 32],
        "TM": [1, 2, 4],
        "TN": [1, 2, 4]
    },
    "stencil": {
        "BM": [16, 32, 64],
        "BN": [32, 64, 128],
        "BK": [0, 1, 2, 8],
        "TM": [1, 2, 4],
        "TN": [1, 2, 4]
    },
}


default_configs = {
    "matmul": (32, 32, 32, 1, 1),
    "stencil": (16, 64, 0, 1, 1),
}


def operation_count(kernel, M, N, K):
    if kernel == "stencil":
        return 6 * max(M - 2, 0) * max(N - 2, 0) * K
    return 2 * M * N * K


def throughput_gflops(kernel, M, N, K, time_ms):
    return operation_count(kernel, M, N, K) / (time_ms * 1e6)


def default_config_id(kernel):
    return "_".join(str(value) for value in default_configs[kernel])


def max_threads_for_backend(backend=None, kernel="matmul"):
    if backend == "cuda":
        if kernel == "stencil":
            return CUDA_STENCIL_MAX_THREADS_PER_BLOCK
        return CUDA_MATMUL_MAX_THREADS_PER_BLOCK
    if backend == "sycl":
        if kernel == "stencil":
            return SYCL_STENCIL_MAX_WORK_GROUP_SIZE
        return SYCL_MATMUL_MAX_WORK_GROUP_SIZE
    return 1024


def all_valid_configs(kernel="matmul", backend=None, M=None, N=None, K=None):
    space = search_spaces.get(kernel, search_spaces["matmul"])
    return [
        f"{bm}_{bn}_{bk}_{tm}_{tn}"
        for bm in space["BM"]
        for bn in space["BN"]
        for bk in space["BK"]
        for tm in space["TM"]
        for tn in space["TN"]
        if is_valid_config(bm, bn, bk, tm, tn, kernel=kernel, backend=backend, M=M, N=N, K=K)[0]
    ]


def compile_config(bm, bn, bk, tm, tn=1, backend="sycl", kernel="matmul"):
    try:
        env = os.environ.copy()
        env['BM'] = str(bm)
        env['BN'] = str(bn)
        env['BK'] = str(bk)
        env['TM'] = str(tm)
        env['TN'] = str(tn)
        
        subprocess.run(["rm", "-f", f"kernel_{kernel}_{backend}.o", f"kernel_{kernel}_{backend}.so"], capture_output=True)
        
        result = subprocess.run(
            ["make", f"kernel_{kernel}_{backend}.so", f"BM={bm}", f"BN={bn}", f"BK={bk}", f"TM={tm}", f"TN={tn}" ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        if result.returncode != 0:
            print(f"  [COMPILE ERROR] {result.stderr[-500:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False

def benchmark_config(M, N, K, bm, bn, bk, tm, tn=1, backend="sycl", kernel="matmul", num_runs=20, warmup_runs=3, seed=DEFAULT_SEED):
    try:
        if not compile_config(bm, bn, bk, tm, tn, backend=backend, kernel=kernel):
            return None, False
        
        script_name = f"_benchmark_temp_{kernel}_{bm}_{bn}_{bk}_{tm}_{tn}.py"
        benchmark_code = f"""
import sys
import json
import math
import statistics
sys.path.insert(0, '.')
from wrapper import run

M, N, K = {M}, {N}, {K}
seed = {seed}
for i in range({warmup_runs}):
    elapsed = run(M, N, K, {bm}, {bn}, {bk}, {tm}, {tn}, backend=\"{backend}\", kernel=\"{kernel}\", seed=seed + i)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError(f"invalid warmup time {{elapsed}}")

times = []
for i in range({num_runs}):
    elapsed = run(M, N, K, {bm}, {bn}, {bk}, {tm}, {tn}, backend=\"{backend}\", kernel=\"{kernel}\", seed=seed + {warmup_runs} + i)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError(f"invalid benchmark time {{elapsed}}")
    times.append(elapsed)

n = len(times)
avg = statistics.mean(times)
median = statistics.median(times)
std = statistics.stdev(times) if n > 1 else 0.0

t_critical = {{1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571,
               6:2.447, 7:2.365, 8:2.306, 9:2.262, 10:2.228,
               11:2.201, 12:2.179, 13:2.160, 14:2.145, 15:2.131}}

t_coeff = t_critical.get(n, 2.0)
ci_half = t_coeff * std / math.sqrt(n) if n > 1 else 0.0

stats = {{
    "mean": avg,
    "median": median,
    "std": std,
    "ci_half": ci_half,
    "n": n,
    "times": times
}}
print(json.dumps(stats))
"""
        
        with open(script_name, 'w') as f:
            f.write(benchmark_code)
        
        result = subprocess.run(
            ["python3", script_name],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        try:
            os.remove(script_name)
        except:
            pass
        
        if result.returncode != 0:
            return None, False
        
        try:
            stats = json.loads(result.stdout.strip().split('\n')[-1])
            median_time = stats.get("median", 0.0)
            if median_time <= 0.0:
                return None, False
            gflops = throughput_gflops(kernel, M, N, K, median_time)
            if gflops > MAX_PLAUSIBLE_GFLOPS:
                print(
                    f"  [BENCHMARK ERROR] implausible throughput {gflops:.2f} GFLOP/s "
                    f"for median={median_time:.6f} ms"
                )
                return None, False
            return stats, True
        except Exception:
            return None, False
            
    except subprocess.TimeoutExpired:
        return None, False
    except Exception as e:
        return None, False

def is_valid_config(bm, bn, bk, tm, tn=1, kernel="matmul", backend=None, M=None, N=None, K=None):
    if tm <= 0 or tn <= 0:
        return False, "TM and TN must be positive"
    if bm % tm != 0 or bn % tn != 0:
        return False, "BM must be divisible by TM and BN must be divisible by TN"

    threads_per_block = (bm // tm) * (bn // tn)
    max_threads_per_block = max_threads_for_backend(backend=backend, kernel=kernel)
    if threads_per_block <= 0 or threads_per_block > max_threads_per_block:
        return False, f"threads={threads_per_block} out of range > {max_threads_per_block}"

    if kernel == "matmul":
        if bk <= 0:
            return False, "BK must be positive for matmul"
        shared_memory = (bm * bk + bk * bn) * 4
        if shared_memory > 48 * 1024:
            return False, f"shared_mem={shared_memory/1024:.1f}KB > 48KB"

        # The CUDA matmul kernel assumes complete tiles. Keep CUDA searches on
        # tile-aligned problem sizes unless boundary guards are added there.
        if backend == "cuda":
            if M is not None and M % bm != 0:
                return False, f"M={M} is not divisible by BM={bm}"
            if N is not None and N % bn != 0:
                return False, f"N={N} is not divisible by BN={bn}"
            if K is not None and K % bk != 0:
                return False, f"K={K} is not divisible by BK={bk}"
    elif kernel == "stencil":
        if bk < 0:
            return False, "BK is row padding for stencil and must be non-negative"
        if M is not None and M < 3:
            return False, f"M={M} is too small for stencil"
        if N is not None and N < 3:
            return False, f"N={N} is too small for stencil"
        if K is not None and K < 1:
            return False, f"K={K} must be at least one stencil iteration"
        shared_memory = (bm + 2) * (bn + 2 + bk) * 4
        if shared_memory > 48 * 1024:
            return False, f"shared_mem={shared_memory/1024:.1f}KB > 48KB"

    return True, None

def random_search(M, N, K, backend="cuda", num_runs=3, num_samples=20,
                  kernel="matmul", warmup_runs=3, seed=DEFAULT_SEED):
    results = []
    
    print(f"\n{'='*80}")
    print(f"Random Search Auto-tuning {backend.upper()} {kernel.upper()} kernel")
    print(f"{'='*80}")
    print(f"Problem size: {M}x{N}x{K}")
    print(f"Samples to evaluate: {num_samples}\n")
    print(f"Benchmark repeats per config: {num_runs}, warmup runs: {warmup_runs}\n")
    
    random.seed(seed)
    
    valid_configs = all_valid_configs(kernel=kernel, backend=backend, M=M, N=N, K=K)
    if not valid_configs:
        print("No valid configurations available in search space.")
        return results

    num_samples = min(num_samples, len(valid_configs))
    default_id = default_config_id(kernel)
    if num_samples > 0 and default_id in valid_configs:
        remaining_configs = [config for config in valid_configs if config != default_id]
        sampled_configs = [default_id] + random.sample(remaining_configs, num_samples - 1)
    else:
        sampled_configs = random.sample(valid_configs, num_samples)

    for sample_idx, config in enumerate(sampled_configs, start=1):
        bm, bn, bk, tm, tn = map(int, config.split('_'))
        stats, success = benchmark_config(
            M, N, K, bm, bn, bk, tm, tn,
            backend=backend,
            kernel=kernel,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            seed=seed
        )

        if success and stats:
            median_time = stats["median"]
            mean_time = stats["mean"]
            ci_half = stats["ci_half"]
            throughput = throughput_gflops(kernel, M, N, K, median_time)
            results.append({
                "BM": bm, "BN": bn, "BK": bk, "TM": tm, "TN": tn,
                "backend": backend,
                "kernel": kernel,
                "time_ms": median_time,
                "mean_time_ms": mean_time,
                "ci_half": ci_half,
                "throughput_gflops": throughput
            })
            print(
                f"[{sample_idx:3d}/{num_samples}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}, TN={tn}: "
                f"{median_time:8.4f} ms (median), mean={mean_time:.4f} ± {ci_half:.4f} ms ({throughput:6.2f} GFLOP/s)"
            )
        else:
            print(f"[{sample_idx:3d}/{num_samples}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}, TN={tn}: FAILED")
    
    return results

def bayesian_search(M, N, K, backend="sycl", num_runs=3, num_trials=30,
                    kernel="matmul", warmup_runs=3, seed=DEFAULT_SEED):
    if optuna is None:
        raise RuntimeError("Bayesian search requires optuna. Install it with: python3 -m pip install optuna")

    print(f"\n{'='*80}")
    print(f"Bayesian Optimization Auto-tuning {backend.upper()} {kernel.upper()} kernel (Optuna)")
    print(f"{'='*80}")
    print(f"Problem size: {M}x{N}x{K}")
    print(f"Trials to evaluate: {num_trials}\n")
    print(f"Benchmark repeats per config: {num_runs}, warmup runs: {warmup_runs}\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = []
    trial_count = [0]

    def objective(trial):
        valid_configs = all_valid_configs(kernel=kernel, backend=backend, M=M, N=N, K=K)
        if not valid_configs:
            raise RuntimeError("No valid configurations available for Bayesian search.")

        config = trial.suggest_categorical('CONFIG', valid_configs)
        bm, bn, bk, tm, tn = map(int, config.split('_'))

        trial_count[0] += 1
        idx = trial_count[0]

        stats, success = benchmark_config(
            M, N, K, bm, bn, bk, tm, tn,
            backend=backend,
            kernel=kernel,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            seed=seed
        )

        if success and stats and stats["median"] > 0:
            median_time = stats["median"]
            mean_time = stats["mean"]
            ci_half = stats["ci_half"]
            throughput = throughput_gflops(kernel, M, N, K, median_time)
            results.append({
                "BM": bm, "BN": bn, "BK": bk, "TM": tm, "TN": tn,
                "backend": backend,
                "kernel": kernel,
                "time_ms": median_time,
                "mean_time_ms": mean_time,
                "ci_half": ci_half,
                "throughput_gflops": throughput,
                "trial": idx
            })
            print(f"[{idx:3d}/{num_trials}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}, TN={tn}: {median_time:8.4f} ms (median), mean={mean_time:.4f} ± {ci_half:.4f} ms ({throughput:6.2f} GFLOP/s)")
            return median_time
        else:
            print(f"[{idx:3d}/{num_trials}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}, TN={tn}: FAILED")
            return float('inf')

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=min(10, num_trials // 3),  
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")

    # Warm-start: give Optuna the corners of the valid space so it doesn't waste
    # startup trials on obviously invalid regions
    space = search_spaces.get(kernel, search_spaces["matmul"])
    valid_configs = [
        (bm, bn, bk, tm, tn)
        for bm in space["BM"]
        for bn in space["BN"]
        for bk in space["BK"]
        for tm in space["TM"]
        for tn in space["TN"]
        if is_valid_config(bm, bn, bk, tm, tn, kernel=kernel, backend=backend, M=M, N=N, K=K)[0]
    ]
    print(f"Valid configurations in search space: {len(valid_configs)}/{len(space['BM'])*len(space['BN'])*len(space['BK'])*len(space['TM'])*len(space['TN'])}\n")

    if not valid_configs:
        return results

    default_id = default_config_id(kernel)
    if default_id in all_valid_configs(kernel=kernel, backend=backend, M=M, N=N, K=K):
        study.enqueue_trial({"CONFIG": default_id})

    study.optimize(objective, n_trials=num_trials, show_progress_bar=False)

    if study.best_value != float('inf'):
        print(f"\nBest trial: Trial #{study.best_trial.number + 1}")
        print(f"Best value: {study.best_value:.4f} ms")

    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-tune kernel block sizes")
    parser.add_argument("--M", type=int, default=DEFAULT_M, help=f"Problem M dimension (default: {DEFAULT_M})")
    parser.add_argument("--N", type=int, default=DEFAULT_N, help=f"Problem N dimension (default: {DEFAULT_N})")
    parser.add_argument("--K", type=int, default=DEFAULT_K, help=f"Problem K dimension / stencil iterations (default: {DEFAULT_K})")
    parser.add_argument("--backend", default="sycl", choices=["cuda", "sycl"], help="Backend to tune (default: sycl)")
    parser.add_argument("--kernel", default="matmul", choices=["matmul", "stencil"], help="Kernel to tune (default: matmul)")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per configuration (default: 10)")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Warmup runs per configuration before timing (default: 3)")
    parser.add_argument("--strategy", default="random", choices=["random", "bayesian"], 
                       help="Search strategy (default: random)")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples for random search (default: 20)")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials for Bayesian optimization (default: 30)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--output", default="autotune_results.json", help="Output file for results")
    args = parser.parse_args()

    if args.strategy == "random":
        results = random_search(
            args.M, args.N, args.K,
            backend=args.backend,
            num_runs=args.runs,
            num_samples=args.samples,
            kernel=args.kernel,
            warmup_runs=args.warmup_runs,
            seed=args.seed
        )
    elif args.strategy == "bayesian":
        results = bayesian_search(
            args.M, args.N, args.K,
            backend=args.backend,
            num_runs=args.runs,
            num_trials=args.trials,
            kernel=args.kernel,
            warmup_runs=args.warmup_runs,
            seed=args.seed
        )
    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")

    if results:
        results_sorted = sorted(results, key=lambda x: x["time_ms"])
        
        for rank, config in enumerate(results_sorted[:10], 1):
            print(f"{rank:<6} {config['BM']:<6} {config['BN']:<6} {config['BK']:<6} {config['TM']:<6} {config.get('TN', 1):<6} "
                  f"{config['time_ms']:<15.4f} {config['throughput_gflops']:<12.2f}")
        
        print("\n" + "="*80)
        best = results_sorted[0]
        print(f"\nBEST CONFIGURATION:")
        print(f"  BM={best['BM']}, BN={best['BN']}, BK={best['BK']}, TM={best['TM']}, TN={best.get('TN', 1)}")
        print(f"  Time: {best['time_ms']:.4f} ms")
        print(f"  Throughput: {best['throughput_gflops']:.2f} GFLOP/s")
        print(f"\nTo use this configuration for compilation:")
        print(f"  make kernel_{args.kernel}_{args.backend}.so BM={best['BM']} BN={best['BN']} BK={best['BK']} TM={best['TM']} TN={best.get('TN', 1)}")
        
        with open(args.output, 'w') as f:
            json.dump(results_sorted, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")
    else:
        print("\nNo valid configurations found!")

if __name__ == "__main__":
    main()
