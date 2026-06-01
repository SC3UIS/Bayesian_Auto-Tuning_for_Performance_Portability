#!/usr/bin/env python3
"""
Run multiple independent autotuning experiments for statistical comparison.
Orchestrates Bayesian vs Random search convergence analysis.
"""

import json
import os
import argparse
from pathlib import Path
import sys
from autotune import bayesian_search, random_search
from statistical_analysis import compare_algorithms, print_statistical_report


def run_tuning_experiment(M, N, K, run_idx, backend="sycl", kernel="matmul",
                          trials=30, bench_runs=5, warmup_runs=3, seed=None):
    """
    Run one complete tuning experiment for one kernel/backend pair.
    Returns convergence data for both Bayesian and Random search.
    """
    if seed is None:
        seed = run_idx * 1000
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {run_idx + 1}: {backend.upper()} {kernel.upper()} Bayesian Search")
    print(f"{'='*80}")
    
    # Run Bayesian search
    try:
        bayesian_results = bayesian_search(
            M, N, K,
            backend=backend,
            num_runs=bench_runs,
            num_trials=trials,
            kernel=kernel,
            warmup_runs=warmup_runs,
            seed=seed
        )
        print(f"✓ Bayesian run completed: {len(bayesian_results)} configs evaluated")
    except Exception as e:
        print(f"ERROR: Bayesian search failed for run {run_idx}: {e}")
        return None, None
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {run_idx + 1}: {backend.upper()} {kernel.upper()} Random Search")
    print(f"{'='*80}")
    
    # Run Random search with same seed
    try:
        random_results = random_search(
            M, N, K,
            backend=backend,
            num_runs=bench_runs,
            num_samples=trials,
            kernel=kernel,
            warmup_runs=warmup_runs,
            seed=seed
        )
        print(f"✓ Random run completed: {len(random_results)} configs evaluated")
    except Exception as e:
        print(f"ERROR: Random search failed for run {run_idx}: {e}")
        return bayesian_results, None
    
    return bayesian_results, random_results


def compute_best_so_far(times):
    best = float('inf')
    best_so_far = []
    for t in times:
        best = min(best, t)
        best_so_far.append(best)
    return best_so_far


def experiment_output_path(base_output, kernel, backend, multi_kernel, multi_backend):
    base_output = Path(base_output)
    suffix_parts = []
    if multi_kernel:
        suffix_parts.append(kernel)
    if multi_backend:
        suffix_parts.append(backend)
    if not suffix_parts:
        return base_output
    suffix = "_".join(suffix_parts)
    return base_output.with_name(f"{base_output.stem}_{suffix}{base_output.suffix}")


def statistical_output_path(convergence_output):
    convergence_output = Path(convergence_output)
    stem = convergence_output.stem
    if stem.startswith("convergence"):
        suffix = stem[len("convergence"):]
        name = f"statistical_results{suffix}{convergence_output.suffix}"
    else:
        name = f"{stem}_statistical_results{convergence_output.suffix}"
    return convergence_output.parent / name


def run_backend_analysis(M, N, K, backend, kernel, num_runs, trials, bench_runs,
                         warmup_runs, convergence_output, statistical_output,
                         legacy_convergence_output=None,
                         legacy_statistical_output=None):
    print("\n" + "="*80)
    print(f"STATISTICAL ANALYSIS: {backend.upper()} {kernel.upper()} BAYESIAN vs RANDOM SEARCH")
    print(f"Problem size: {M}x{N}x{K}")
    print(f"Independent runs: {num_runs}")
    print(f"Trials per run: {trials}")
    print(f"Benchmark repeats per config: {bench_runs}")
    print(f"Warmup runs per config: {warmup_runs}")
    print("="*80)

    all_bayesian = []
    all_random = []
    convergence_data = {
        "kernel": kernel,
        "backend": backend,
        "problem_size": {"M": M, "N": N, "K": K},
        "num_runs": num_runs,
        "trials_per_run": trials,
        "bench_runs_per_config": bench_runs,
        "warmup_runs_per_config": warmup_runs,
        "bayesian_runs": [],
        "random_runs": []
    }

    for run_idx in range(num_runs):
        bay_results, ran_results = run_tuning_experiment(
            M, N, K, run_idx,
            backend=backend,
            kernel=kernel,
            trials=trials,
            bench_runs=bench_runs,
            warmup_runs=warmup_runs,
            seed=run_idx * 1000
        )

        if bay_results:
            bay_times = [cfg["time_ms"] for cfg in bay_results]
            all_bayesian.append(bay_times)
            bay_best_so_far = compute_best_so_far(bay_times)
            convergence_data["bayesian_runs"].append({
                "run_idx": run_idx,
                "backend": backend,
                "kernel": kernel,
                "times": bay_times,
                "evaluations": bay_results,
                "best_time_ms": min(bay_times),
                "mean_time_ms": sum(bay_times) / len(bay_times),
                "best_so_far": bay_best_so_far
            })
            print(f"\n✓ {backend.upper()} Bayesian Run {run_idx+1}: best = {min(bay_times):.4f} ms")
        else:
            print(f"\n✗ {backend.upper()} Bayesian Run {run_idx+1}: FAILED")

        if ran_results:
            ran_times = [cfg["time_ms"] for cfg in ran_results]
            all_random.append(ran_times)
            ran_best_so_far = compute_best_so_far(ran_times)
            convergence_data["random_runs"].append({
                "run_idx": run_idx,
                "backend": backend,
                "kernel": kernel,
                "times": ran_times,
                "evaluations": ran_results,
                "best_time_ms": min(ran_times),
                "mean_time_ms": sum(ran_times) / len(ran_times),
                "best_so_far": ran_best_so_far
            })
            print(f"✓ {backend.upper()} Random Run {run_idx+1}:   best = {min(ran_times):.4f} ms")
        else:
            print(f"✗ {backend.upper()} Random Run {run_idx+1}:   FAILED")

    Path(convergence_output).parent.mkdir(parents=True, exist_ok=True)
    with open(convergence_output, 'w') as f:
        json.dump(convergence_data, f, indent=2)
    print(f"\nSaved {backend.upper()} convergence data to: {convergence_output}")

    if legacy_convergence_output and Path(legacy_convergence_output) != Path(convergence_output):
        with open(legacy_convergence_output, 'w') as f:
            json.dump(convergence_data, f, indent=2)
        print(f"Legacy convergence data also saved to: {legacy_convergence_output}")

    if all_bayesian and all_random:
        bayesian_best = [min(run) for run in all_bayesian]
        random_best = [min(run) for run in all_random]

        print("\n" + "="*80)
        print(f"Running {backend.upper()} statistical analysis...")
        print("="*80)

        try:
            results = compare_algorithms(bayesian_best, random_best)
            results["kernel"] = kernel
            results["backend"] = backend
            results["problem_size"] = {"M": M, "N": N, "K": K}
            results["num_runs"] = num_runs
            results["trials_per_run"] = trials
            results["bench_runs_per_config"] = bench_runs
            results["warmup_runs_per_config"] = warmup_runs
            print_statistical_report(results)

            with open(statistical_output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"{backend.upper()} statistical results saved to: {statistical_output}")

            if legacy_statistical_output and Path(legacy_statistical_output) != Path(statistical_output):
                with open(legacy_statistical_output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Legacy statistical results also saved to: {legacy_statistical_output}")
        except Exception as e:
            print(f"ERROR running {backend.upper()} statistical analysis: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print(f"\nInsufficient {backend.upper()} data for statistical analysis")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description="Run multiple independent tuning experiments")
    parser.add_argument('--M', type=int, default=512, help='Problem M dimension')
    parser.add_argument('--N', type=int, default=512, help='Problem N dimension')
    parser.add_argument('--K', type=int, default=512, help='Problem K dimension for matmul, iteration count for stencil')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of independent runs (default: 5)')
    parser.add_argument('--trials', type=int, default=40, help='Trials per run (default: 40)')
    parser.add_argument('--bench-runs', type=int, default=5, help='Timed benchmark repeats per configuration (default: 5)')
    parser.add_argument('--warmup-runs', type=int, default=3, help='Untimed warmup runs per configuration (default: 3)')
    parser.add_argument(
        '--kernel',
        choices=['matmul', 'stencil'],
        default=None,
        help='Single kernel to tune. Alias for --kernels with one value.'
    )
    parser.add_argument(
        '--kernels',
        nargs='+',
        default=['matmul'],
        choices=['matmul', 'stencil'],
        help='Kernel(s) to tune, e.g. --kernels matmul stencil (default: matmul)'
    )
    parser.add_argument(
        '--backends',
        nargs='+',
        default=['sycl'],
        choices=['sycl', 'cuda'],
        help='Backend(s) to tune, e.g. --backends sycl cuda (default: sycl)'
    )
    parser.add_argument('--output', default="convergence_analysis.json", help="Output file for convergence data")
    
    args = parser.parse_args()
    
    M, N, K = args.M, args.N, args.K
    kernels = [args.kernel] if args.kernel else args.kernels

    multi_kernel = len(kernels) > 1
    multi_backend = len(args.backends) > 1
    exit_code = 0
    outputs = []

    for kernel_idx, kernel in enumerate(kernels):
        for backend_idx, backend in enumerate(args.backends):
            convergence_output = experiment_output_path(
                args.output, kernel, backend, multi_kernel, multi_backend
            )
            statistical_output = statistical_output_path(convergence_output)
            write_legacy = multi_backend and not multi_kernel and backend_idx == 0
            code = run_backend_analysis(
                M, N, K, backend, kernel,
                num_runs=args.num_runs,
                trials=args.trials,
                bench_runs=args.bench_runs,
                warmup_runs=args.warmup_runs,
                convergence_output=convergence_output,
                statistical_output=statistical_output,
                legacy_convergence_output=args.output if write_legacy else None,
                legacy_statistical_output=(
                    Path(args.output).parent / "statistical_results.json"
                    if write_legacy
                    else None
                )
            )
            outputs.append((kernel, backend, convergence_output, statistical_output))
            exit_code = max(exit_code, code)
    
    # Clean up temp files
    for i in range(args.num_runs):
        for prefix in ["_temp_bayesian_run", "_temp_random_run"]:
            try:
                os.remove(f"{prefix}{i}.json")
            except:
                pass

    print("\n" + "="*80)
    print("Analysis complete! Results saved to:")
    for kernel, backend, convergence_output, statistical_output in outputs:
        print(f"  - {backend.upper()} {kernel.upper()} convergence data: {convergence_output}")
        print(f"  - {backend.upper()} {kernel.upper()} statistical results: {statistical_output}")
    print("="*80 + "\n")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
