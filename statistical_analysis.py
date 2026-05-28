#!/usr/bin/env python3
"""
Statistical analysis tools for comparing optimization algorithms.
Computes convergence rates, significance tests, and publishes results.
"""

import json
import math
import statistics
from pathlib import Path

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

try:
    import numpy as np
except ImportError:
    np = None

plt = None


def require_stats_deps():
    if np is None or sp_stats is None:
        raise RuntimeError(
            "Statistical analysis requires numpy and scipy. "
            "Install them with: python3 -m pip install numpy scipy"
        )


def require_plot_deps():
    global plt
    if plt is None:
        try:
            import matplotlib.pyplot as matplotlib_pyplot
        except ImportError as exc:
            raise RuntimeError(
                "Plotting requires matplotlib. Install it with: python3 -m pip install matplotlib"
            ) from exc
        plt = matplotlib_pyplot


def load_convergence_data(filename):
    """Load convergence data from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def compute_convergence_rate(times, window=5):
    """
    Compute convergence rate as the slope of best-so-far times.
    Uses linear regression on log-scale for rate estimation.
    """
    require_stats_deps()
    best_so_far = []
    best = float('inf')
    for t in times:
        best = min(best, t)
        best_so_far.append(best)
    
    if len(best_so_far) < window:
        return None
    
    # Compute slope over last window evaluations
    x = np.arange(len(best_so_far) - window, len(best_so_far))
    y = np.array(best_so_far[-window:])
    
    # Avoid log of zero or negative
    y_safe = np.maximum(y, 1e-6)
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, np.log(y_safe))
    
    return slope, r_value**2, std_err


def welch_ttest(group1, group2):
    """
    Welch's t-test for unequal variances.
    Returns t-statistic, p-value, and mean difference.
    """
    require_stats_deps()
    t_stat, p_value = sp_stats.ttest_ind(group1, group2, equal_var=False)
    mean_diff = np.mean(group1) - np.mean(group2)
    ci_95 = 1.96 * np.sqrt(np.var(group1)/len(group1) + np.var(group2)/len(group2))
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_difference": float(mean_diff),
        "ci_95": float(ci_95),
        "significant_at_0_05": float(p_value) < 0.05,
        "significant_at_0_01": float(p_value) < 0.01
    }


def mann_whitney_u(group1, group2):
    """
    Mann-Whitney U test (non-parametric alternative to t-test).
    More robust for non-normal distributions.
    """
    require_stats_deps()
    u_stat, p_value = sp_stats.mannwhitneyu(group1, group2, alternative='two-sided')
    return {
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "significant_at_0_05": float(p_value) < 0.05,
        "significant_at_0_01": float(p_value) < 0.01
    }


def cohens_d(group1, group2):
    """Compute Cohen's d with a guard for zero pooled variance."""
    require_stats_deps()
    if len(group1) < 2 or len(group2) < 2:
        return float("nan")
    pooled_var = (
        ((len(group1) - 1) * np.var(group1, ddof=1) +
         (len(group2) - 1) * np.var(group2, ddof=1)) /
        (len(group1) + len(group2) - 2)
    )
    if pooled_var <= 0:
        return 0.0 if np.mean(group1) == np.mean(group2) else float("inf")
    return float((np.mean(group1) - np.mean(group2)) / np.sqrt(pooled_var))


def compare_algorithms(bayesian_best_times, random_best_times):
    """
    Comprehensive comparison of two optimization algorithms.
    
    Args:
        bayesian_best_times: list of best times from Bayesian runs
        random_best_times: list of best times from Random runs
    
    Returns:
        dict with statistical analysis results
    """
    require_stats_deps()
    results = {
        "sample_sizes": {
            "bayesian": len(bayesian_best_times),
            "random": len(random_best_times)
        },
        "descriptive_stats": {
            "bayesian": {
                "mean": float(np.mean(bayesian_best_times)),
                "median": float(np.median(bayesian_best_times)),
                "std": float(np.std(bayesian_best_times)),
                "min": float(np.min(bayesian_best_times)),
                "max": float(np.max(bayesian_best_times)),
                "q1": float(np.percentile(bayesian_best_times, 25)),
                "q3": float(np.percentile(bayesian_best_times, 75))
            },
            "random": {
                "mean": float(np.mean(random_best_times)),
                "median": float(np.median(random_best_times)),
                "std": float(np.std(random_best_times)),
                "min": float(np.min(random_best_times)),
                "max": float(np.max(random_best_times)),
                "q1": float(np.percentile(random_best_times, 25)),
                "q3": float(np.percentile(random_best_times, 75))
            }
        },
        "effect_size_cohens_d": cohens_d(bayesian_best_times, random_best_times),
        "welch_ttest": welch_ttest(bayesian_best_times, random_best_times),
        "mann_whitney_u": mann_whitney_u(bayesian_best_times, random_best_times),
        "winner": "Bayesian" if np.mean(bayesian_best_times) < np.mean(random_best_times) else "Random"
    }
    
    return results


def plot_convergence_comparison(convergence_data, output_file="convergence_comparison.png"):
    """
    Plot convergence curves for all runs side-by-side.
    """
    require_plot_deps()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bayesian convergence
    ax_bay = axes[0]
    for run_idx, run_data in enumerate(convergence_data.get("bayesian_runs", [])):
        times = run_data["times"]
        best_so_far = []
        best = float('inf')
        for t in times:
            best = min(best, t)
            best_so_far.append(best)
        ax_bay.plot(best_so_far, alpha=0.6, label=f"Run {run_idx+1}")
    
    ax_bay.set_xlabel("Evaluation #", fontsize=12, fontweight='bold')
    ax_bay.set_ylabel("Best Time So Far (ms)", fontsize=12, fontweight='bold')
    ax_bay.set_title("Bayesian Optimization Convergence", fontsize=14, fontweight='bold')
    ax_bay.grid(True, alpha=0.3)
    ax_bay.legend()
    
    # Random search convergence
    ax_ran = axes[1]
    for run_idx, run_data in enumerate(convergence_data.get("random_runs", [])):
        times = run_data["times"]
        best_so_far = []
        best = float('inf')
        for t in times:
            best = min(best, t)
            best_so_far.append(best)
        ax_ran.plot(best_so_far, alpha=0.6, label=f"Run {run_idx+1}")
    
    ax_ran.set_xlabel("Evaluation #", fontsize=12, fontweight='bold')
    ax_ran.set_ylabel("Best Time So Far (ms)", fontsize=12, fontweight='bold')
    ax_ran.set_title("Random Search Convergence", fontsize=14, fontweight='bold')
    ax_ran.grid(True, alpha=0.3)
    ax_ran.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved convergence comparison plot: {output_file}")


def print_statistical_report(comparison_results):
    """Pretty-print statistical analysis results."""
    print("\n" + "="*80)
    kernel = comparison_results.get("kernel")
    backend = comparison_results.get("backend")
    label = "STATISTICAL COMPARISON"
    if backend or kernel:
        label += f": {str(backend or '').upper()} {str(kernel or '').upper()}".rstrip()
    print(f"{label}: BAYESIAN vs RANDOM SEARCH")
    print("="*80)
    
    print(f"\nSample Sizes:")
    print(f"  Bayesian: {comparison_results['sample_sizes']['bayesian']} runs")
    print(f"  Random:   {comparison_results['sample_sizes']['random']} runs")
    
    print(f"\nDescriptive Statistics (Best Time Found):")
    for algo in ["bayesian", "random"]:
        stats = comparison_results["descriptive_stats"][algo]
        print(f"\n  {algo.upper()}:")
        print(f"    Mean:   {stats['mean']:.4f} ms")
        print(f"    Median: {stats['median']:.4f} ms")
        print(f"    Std:    {stats['std']:.4f} ms")
        print(f"    Range:  [{stats['min']:.4f}, {stats['max']:.4f}] ms")
        print(f"    IQR:    [{stats['q1']:.4f}, {stats['q3']:.4f}] ms")
    
    effect_size = comparison_results['effect_size_cohens_d']
    print(f"\nEffect Size (Cohen's d): {effect_size:.3f}")
    if not math.isfinite(effect_size):
        print("  → Not available with the current samples")
    elif abs(effect_size) < 0.2:
        print("  → Negligible effect size")
    elif abs(effect_size) < 0.5:
        print("  → Small effect size")
    elif abs(effect_size) < 0.8:
        print("  → Medium effect size")
    else:
        print("  → Large effect size")
    
    welch = comparison_results["welch_ttest"]
    print(f"\nWelch's t-test:")
    print(f"  t-statistic: {welch['t_statistic']:.4f}")
    print(f"  p-value:     {welch['p_value']:.4f}")
    print(f"  Mean diff:   {welch['mean_difference']:.4f} ms (95% CI: ±{welch['ci_95']:.4f})")
    if welch['significant_at_0_01']:
        print(f"  Result:      SIGNIFICANT (p < 0.01) ***")
    elif welch['significant_at_0_05']:
        print(f"  Result:      SIGNIFICANT (p < 0.05) **")
    else:
        print(f"  Result:      Not significant")
    
    mw = comparison_results["mann_whitney_u"]
    print(f"\nMann-Whitney U test (non-parametric):")
    print(f"  U-statistic: {mw['u_statistic']:.4f}")
    print(f"  p-value:     {mw['p_value']:.4f}")
    if mw['significant_at_0_01']:
        print(f"  Result:      SIGNIFICANT (p < 0.01) ***")
    elif mw['significant_at_0_05']:
        print(f"  Result:      SIGNIFICANT (p < 0.05) **")
    else:
        print(f"  Result:      Not significant")
    
    print(f"\nWINNER: {comparison_results['winner']}")
    print("="*80 + "\n")


def extract_best_times(data, algorithm=None):
    """Extract best-time samples from either combined convergence data or a list."""
    if isinstance(data, dict):
        if algorithm is None:
            raise ValueError("algorithm is required when reading combined convergence data")
        return [run["best_time_ms"] for run in data.get(f"{algorithm}_runs", [])]
    return [run["best_time_ms"] for run in data]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical analysis of optimization algorithms")
    parser.add_argument("--convergence-results", help="Combined convergence JSON produced by run_statistical_experiments.py")
    parser.add_argument("--bayesian-results", help="JSON file with Bayesian run results")
    parser.add_argument("--random-results", help="JSON file with Random run results")
    parser.add_argument("--output", default="statistical_analysis.json", help="Output JSON file")
    parser.add_argument("--plot-output", default=None, help="Optional convergence plot output path")
    
    args = parser.parse_args()
    
    metadata = {}
    if args.convergence_results:
        convergence_data = load_convergence_data(args.convergence_results)
        bayesian_best_times = extract_best_times(convergence_data, "bayesian")
        random_best_times = extract_best_times(convergence_data, "random")
        metadata = {
            key: convergence_data[key]
            for key in ("kernel", "backend", "problem_size", "num_runs",
                        "trials_per_run", "bench_runs_per_config",
                        "warmup_runs_per_config")
            if key in convergence_data
        }
        if args.plot_output:
            plot_convergence_comparison(convergence_data, args.plot_output)
    else:
        if not args.bayesian_results or not args.random_results:
            parser.error("Provide --convergence-results or both --bayesian-results and --random-results")
        bayesian_data = load_convergence_data(args.bayesian_results)
        random_data = load_convergence_data(args.random_results)
        bayesian_best_times = extract_best_times(bayesian_data)
        random_best_times = extract_best_times(random_data)
    
    # Run analysis
    comparison = compare_algorithms(bayesian_best_times, random_best_times)
    comparison.update(metadata)
    
    # Print report
    print_statistical_report(comparison)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved analysis to: {args.output}")
