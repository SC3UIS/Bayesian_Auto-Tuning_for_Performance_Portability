#!/usr/bin/env python3
"""
Run multiple independent autotuning experiments for statistical comparison.
Orchestrates Bayesian vs Random search convergence analysis.
"""

import json
import math
import os
import argparse
import statistics
from pathlib import Path
import sys
from autotune import (
    all_valid_configs,
    bayesian_search,
    benchmark_config,
    default_configs,
    exhaustive_grid_search,
    random_search,
)
from statistical_analysis import compare_algorithms, print_statistical_report


T_CRITICAL_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def validate_best_against_default(M, N, K, results, backend, kernel,
                                  validation_runs, warmup_runs, seed,
                                  candidate_first):
    best = min(results, key=lambda result: result["time_ms"])
    tuned_config = tuple(int(best.get(key, 1)) for key in ("BM", "BN", "BK", "TM", "TN"))
    default_config = default_configs[kernel]
    validation_seed = seed + 1_000_000

    def measure(config):
        return benchmark_config(
            M, N, K, *config,
            backend=backend,
            kernel=kernel,
            num_runs=validation_runs,
            warmup_runs=warmup_runs,
            seed=validation_seed,
        )

    if tuned_config == default_config:
        default_stats, success = measure(default_config)
        tuned_stats = default_stats
        if not success:
            return {"status": "failed", "reason": "default validation benchmark failed"}
    else:
        ordered_configs = (
            [("tuned", tuned_config), ("default", default_config)]
            if candidate_first
            else [("default", default_config), ("tuned", tuned_config)]
        )
        measured = {}
        for label, config in ordered_configs:
            stats, success = measure(config)
            if not success:
                return {
                    "status": "failed",
                    "reason": f"{label} validation benchmark failed",
                    "validation_seed": validation_seed,
                    "default_config": list(default_config),
                    "tuned_config": list(tuned_config),
                }
            measured[label] = stats
        default_stats = measured["default"]
        tuned_stats = measured["tuned"]

    paired_speedups = [
        default_time / tuned_time
        for default_time, tuned_time in zip(default_stats["times"], tuned_stats["times"])
        if default_time > 0.0 and tuned_time > 0.0
    ]
    log_speedups = [math.log(speedup) for speedup in paired_speedups]
    mean_log_speedup = statistics.mean(log_speedups)
    geometric_mean_speedup = math.exp(mean_log_speedup)

    if len(log_speedups) > 1:
        degrees_of_freedom = len(log_speedups) - 1
        critical_value = T_CRITICAL_975.get(degrees_of_freedom, 1.96)
        margin = critical_value * statistics.stdev(log_speedups) / math.sqrt(len(log_speedups))
        ci_lower = math.exp(mean_log_speedup - margin)
        ci_upper = math.exp(mean_log_speedup + margin)
    else:
        ci_lower = None
        ci_upper = None

    return {
        "status": "ok",
        "validation_seed": validation_seed,
        "validation_runs": len(paired_speedups),
        "default_config": list(default_config),
        "tuned_config": list(tuned_config),
        "default_stats": default_stats,
        "tuned_stats": tuned_stats,
        "paired_speedups_default_over_tuned": paired_speedups,
        "median_paired_speedup": statistics.median(paired_speedups),
        "geometric_mean_paired_speedup": geometric_mean_speedup,
        "geometric_mean_speedup_ci95": [ci_lower, ci_upper],
        "pairs_faster_than_default": sum(speedup > 1.0 for speedup in paired_speedups),
        "improvement_supported_at_95_percent": ci_lower is not None and ci_lower > 1.0,
    }


def run_tuning_experiment(M, N, K, run_idx, backend="sycl", kernel="matmul",
                          trials=30, bench_runs=5, warmup_runs=3,
                          validation_runs=20, seed=None):
    """
    Run one complete tuning experiment for one kernel/backend pair.
    Returns tuning results and independent validation results for both searches.
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
        return None, None, {}
    
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
        return bayesian_results, None, {}
    
    validation_results = {}
    for strategy_idx, (strategy, results) in enumerate((
        ("bayesian", bayesian_results),
        ("random", random_results),
    )):
        if not results:
            continue
        validation = validate_best_against_default(
            M, N, K, results, backend, kernel,
            validation_runs, warmup_runs, seed,
            candidate_first=(run_idx + strategy_idx) % 2 == 1,
        )
        validation_results[strategy] = validation
        if validation["status"] == "ok":
            lower, upper = validation["geometric_mean_speedup_ci95"]
            ci_text = f"[{lower:.3f}, {upper:.3f}]" if lower is not None else "unavailable"
            print(
                f"{strategy.upper()} holdout: geometric-mean speedup="
                f"{validation['geometric_mean_paired_speedup']:.3f}x, "
                f"95% CI={ci_text}, "
                f"{validation['pairs_faster_than_default']}/"
                f"{validation['validation_runs']} paired seeds faster"
            )
        else:
            print(f"{strategy.upper()} holdout validation failed: {validation['reason']}")

    return bayesian_results, random_results, validation_results


def compute_best_so_far(times):
    best = float('inf')
    best_so_far = []
    for t in times:
        best = min(best, t)
        best_so_far.append(best)
    return best_so_far


def compute_grid_regret(results, grid_search_result):
    if grid_search_result["status"] != "complete":
        return {"status": "unavailable", "reason": "exhaustive grid search incomplete"}

    best_result = min(results, key=lambda result: result["time_ms"])
    config_id = "_".join(
        str(int(best_result.get(key, 1)))
        for key in ("BM", "BN", "BK", "TM", "TN")
    )
    grid_times = {
        evaluation["config"]: evaluation["time_ms"]
        for evaluation in grid_search_result["evaluations"]
    }
    if config_id not in grid_times:
        return {
            "status": "unavailable",
            "reason": "selected configuration missing from grid results",
            "selected_config": config_id,
        }

    optimum_time = grid_search_result["optimum_time_ms"]
    selected_time = grid_times[config_id]
    return {
        "status": "ok",
        "selected_config": config_id,
        "selected_grid_time_ms": selected_time,
        "optimum_config": grid_search_result["optimum_config"],
        "optimum_time_ms": optimum_time,
        "regret_ms": selected_time - optimum_time,
        "regret_percent": 100.0 * (selected_time / optimum_time - 1.0),
    }


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
                         warmup_runs, validation_runs, grid_runs,
                         grid_warmup_runs, skip_grid_search, convergence_output,
                         statistical_output,
                         legacy_convergence_output=None,
                         legacy_statistical_output=None):
    print("\n" + "="*80)
    print(f"STATISTICAL ANALYSIS: {backend.upper()} {kernel.upper()} BAYESIAN vs RANDOM SEARCH")
    print(f"Problem size: {M}x{N}x{K}")
    print(f"Independent runs: {num_runs}")
    print(f"Trials per run: {trials}")
    print(f"Benchmark repeats per config: {bench_runs}")
    print(f"Warmup runs per config: {warmup_runs}")
    print(f"Independent validation repeats per config: {validation_runs}")
    print(f"Grid-search repeats per config: {grid_runs}")
    print("="*80)

    valid_config_count = len(all_valid_configs(
        kernel=kernel, backend=backend, M=M, N=N, K=K
    ))
    if skip_grid_search:
        grid_result = {
            "status": "skipped",
            "valid_config_count": valid_config_count,
            "evaluated_config_count": 0,
            "failed_config_count": 0,
            "failed_configs": [],
            "num_runs_per_config": grid_runs,
            "warmup_runs_per_config": grid_warmup_runs,
            "seed": None,
            "optimum_config": None,
            "optimum_time_ms": None,
            "evaluations": [],
        }
    else:
        grid_result = exhaustive_grid_search(
            M, N, K,
            backend=backend,
            kernel=kernel,
            num_runs=grid_runs,
            warmup_runs=grid_warmup_runs,
            seed=2_000_000,
        )
    if grid_result["status"] == "complete":
        print(
            f"Grid optimum: {grid_result['optimum_config']} at "
            f"{grid_result['optimum_time_ms']:.4f} ms "
            f"({grid_result['valid_config_count']} configs)"
        )
    elif grid_result["status"] == "skipped":
        print(
            f"Grid skipped for smoke test; "
            f"{grid_result['valid_config_count']} valid configs, regrets unavailable"
        )
    else:
        print(
            f"Grid incomplete: measured {grid_result['evaluated_config_count']}/"
            f"{grid_result['valid_config_count']} configs; regrets unavailable"
        )

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
        "validation_runs_per_config": validation_runs,
        "grid_search": grid_result,
        "bayesian_runs": [],
        "random_runs": []
    }

    for run_idx in range(num_runs):
        bay_results, ran_results, validation_results = run_tuning_experiment(
            M, N, K, run_idx,
            backend=backend,
            kernel=kernel,
            trials=trials,
            bench_runs=bench_runs,
            warmup_runs=warmup_runs,
            validation_runs=validation_runs,
            seed=run_idx * 1000
        )

        if bay_results:
            bay_times = [cfg["time_ms"] for cfg in bay_results]
            all_bayesian.append(bay_times)
            bay_best_so_far = compute_best_so_far(bay_times)
            bay_regret = compute_grid_regret(bay_results, grid_result)
            convergence_data["bayesian_runs"].append({
                "run_idx": run_idx,
                "backend": backend,
                "kernel": kernel,
                "times": bay_times,
                "evaluations": bay_results,
                "best_time_ms": min(bay_times),
                "mean_time_ms": sum(bay_times) / len(bay_times),
                "best_so_far": bay_best_so_far,
                "validation": validation_results.get("bayesian"),
                "grid_regret": bay_regret,
            })
            print(f"\n✓ {backend.upper()} Bayesian Run {run_idx+1}: best = {min(bay_times):.4f} ms")
        else:
            print(f"\n✗ {backend.upper()} Bayesian Run {run_idx+1}: FAILED")

        if ran_results:
            ran_times = [cfg["time_ms"] for cfg in ran_results]
            all_random.append(ran_times)
            ran_best_so_far = compute_best_so_far(ran_times)
            random_regret = compute_grid_regret(ran_results, grid_result)
            convergence_data["random_runs"].append({
                "run_idx": run_idx,
                "backend": backend,
                "kernel": kernel,
                "times": ran_times,
                "evaluations": ran_results,
                "best_time_ms": min(ran_times),
                "mean_time_ms": sum(ran_times) / len(ran_times),
                "best_so_far": ran_best_so_far,
                "validation": validation_results.get("random"),
                "grid_regret": random_regret,
            })
            print(f"✓ {backend.upper()} Random Run {run_idx+1}:   best = {min(ran_times):.4f} ms")
        else:
            print(f"✗ {backend.upper()} Random Run {run_idx+1}:   FAILED")

    for strategy in ("bayesian", "random"):
        run_key = f"{strategy}_runs"
        validations = [
            run.get("validation")
            for run in convergence_data[run_key]
            if (run.get("validation") or {}).get("status") == "ok"
        ]
        if validations:
            supported = sum(
                validation["improvement_supported_at_95_percent"]
                for validation in validations
            )
            median_speedup = statistics.median(
                validation["geometric_mean_paired_speedup"]
                for validation in validations
            )
            print(
                f"\n{strategy.upper()} independent validation: "
                f"median campaign speedup={median_speedup:.3f}x; "
                f"95% CI above 1.0 in {supported}/{len(validations)} campaigns"
            )
        else:
            print(f"\n{strategy.upper()} independent validation: no successful campaigns")

        regrets = [
            run["grid_regret"]["regret_percent"]
            for run in convergence_data[run_key]
            if run.get("grid_regret", {}).get("status") == "ok"
        ]
        if regrets:
            print(
                f"{strategy.upper()} grid regret: "
                f"median={statistics.median(regrets):.3f}%, "
                f"mean={statistics.mean(regrets):.3f}% "
                f"across {len(regrets)} campaigns"
            )
        else:
            print(f"{strategy.upper()} grid regret: unavailable")

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
            results["grid_runs_per_config"] = grid_runs
            results["grid_warmup_runs_per_config"] = grid_warmup_runs
            results["grid_search_status"] = grid_result["status"]
            results["grid_valid_config_count"] = grid_result["valid_config_count"]
            results["grid_optimum_config"] = grid_result["optimum_config"]
            results["grid_optimum_time_ms"] = grid_result["optimum_time_ms"]
            bayesian_regrets = [
                run["grid_regret"]["regret_percent"]
                for run in convergence_data["bayesian_runs"]
                if run.get("grid_regret", {}).get("status") == "ok"
            ]
            random_regrets = [
                run["grid_regret"]["regret_percent"]
                for run in convergence_data["random_runs"]
                if run.get("grid_regret", {}).get("status") == "ok"
            ]
            if bayesian_regrets and random_regrets:
                results["grid_regret_comparison"] = {
                    "metric": "relative_simple_regret_percent",
                    "bayesian_median_percent": statistics.median(bayesian_regrets),
                    "random_median_percent": statistics.median(random_regrets),
                    "bayesian_runs": bayesian_regrets,
                    "random_runs": random_regrets,
                }
                if len(bayesian_regrets) >= 2 and len(random_regrets) >= 2:
                    results["grid_regret_comparison"]["statistical_tests"] = (
                        compare_algorithms(bayesian_regrets, random_regrets)
                    )
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
        '--validation-runs', type=int, default=20,
        help='Independent paired validation repeats per config (default: 20)'
    )
    parser.add_argument(
        '--grid-runs', type=int, default=3,
        help='Timed exhaustive-grid repeats per configuration (default: 3)'
    )
    parser.add_argument(
        '--grid-warmup-runs', type=int, default=1,
        help='Untimed warmup repeats per exhaustive-grid config (default: 1)'
    )
    parser.add_argument(
        '--skip-grid-search', action='store_true',
        help='Skip exhaustive grid search (useful for a quick pipeline smoke test)'
    )
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

    if args.validation_runs < 2:
        parser.error('--validation-runs must be at least 2 to estimate a confidence interval')
    if args.grid_runs < 1 or args.grid_warmup_runs < 0:
        parser.error('--grid-runs must be positive and --grid-warmup-runs non-negative')
    
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
                validation_runs=args.validation_runs,
                grid_runs=args.grid_runs,
                grid_warmup_runs=args.grid_warmup_runs,
                skip_grid_search=args.skip_grid_search,
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
