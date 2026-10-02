#!/usr/bin/env python3
"""
Expected input layout, as produced by autotune.sbatch:

    results_YYYYMMDD_HHMMSS/
      matmul/
        size_512/
          convergence_sycl.json
          convergence_cuda.json
      stencil/
        size_512/
          convergence_sycl.json
          convergence_cuda.json

The script creates:
  - convergence plots for Bayesian vs Random search
  - SYCL vs CUDA performance plots for matmul and stencil
  - CSV summaries 
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import TYPE_CHECKING, Any

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
else:
    plt: Any = None

KERNELS = ("matmul", "stencil")
BACKENDS = ("sycl", "cuda")
ALGORITHMS = ("bayesian", "random")

ALGORITHM_LABELS = {
    "bayesian": "Bayesian",
    "random": "Random",
}

BACKEND_LABELS = {
    "sycl": "SYCL",
    "cuda": "CUDA",
}

KERNEL_LABELS = {
    "matmul": "Matmul",
    "stencil": "Stencil",
}


@dataclass(frozen=True)
class ResultEntry:
    kernel: str
    backend: str
    size: int
    path: Path
    data: dict

    @property
    def problem_size(self) -> tuple[int, int, int]:
        problem = self.data.get("problem_size", {})
        return (
            int(problem.get("M", self.size)),
            int(problem.get("N", self.size)),
            int(problem.get("K", self.size)),
        )


def require_plot_deps() -> None:
    global plt
    if plt is None:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"),
        )
        try:
            import matplotlib.pyplot as matplotlib_pyplot
        except ImportError as exc:
            raise RuntimeError(
                "Plotting requires matplotlib. Install it with: python3 -m pip install matplotlib"
            ) from exc
        plt = matplotlib_pyplot

        plt.rcParams.update({
            'font.size': 14,             # Base font size for all text
            'axes.titlesize': 16,        # Size of individual subplot titles (e.g., "N = 512")
            'axes.labelsize': 13,        # Size of x and y axis labels (e.g., "Evaluation")
            'xtick.labelsize': 11,       # Size of x-axis tick labels
            'ytick.labelsize': 11,       # Size of y-axis tick labels
            'legend.fontsize': 11,       # Size of the legend labels
            'figure.titlesize': 16,      # Size of the overall figure super title
        })


def parse_size(size_dir: Path) -> int:
    try:
        return int(size_dir.name.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse matrix size from {size_dir}") from exc


def latest_results_dir(base_dir: Path) -> Path:
    candidates = [path for path in base_dir.glob("results_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No results_* directories were found in {base_dir}. "
            "Pass a results directory explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def convergence_candidates(size_dir: Path, kernel: str, backend: str) -> list[Path]:
    return [
        size_dir / f"convergence_{kernel}_{backend}.json",
        size_dir / f"convergence_{backend}.json",
        size_dir / f"convergence_{kernel}.json",
        size_dir / "convergence.json",
    ]


def load_convergence(
    size_dir: Path, kernel: str, backend: str
) -> tuple[Path, dict] | None:
    for path in convergence_candidates(size_dir, kernel, backend):
        if not path.exists():
            continue
        data = load_json(path)
        data_kernel = data.get("kernel")
        data_backend = data.get("backend")
        if data_kernel and data_kernel != kernel:
            continue
        if data_backend and data_backend != backend:
            continue
        return path, data
    return None


def discover_entries(
    results_dir: Path, kernels: list[str], backends: list[str]
) -> list[ResultEntry]:
    entries: list[ResultEntry] = []
    for kernel in kernels:
        kernel_dir = results_dir / kernel
        search_root = kernel_dir if kernel_dir.exists() else results_dir
        size_dirs = sorted(search_root.glob("size_*"), key=parse_size)

        for size_dir in size_dirs:
            size = parse_size(size_dir)
            for backend in backends:
                loaded = load_convergence(size_dir, kernel, backend)
                if loaded is None:
                    continue
                path, data = loaded
                entries.append(
                    ResultEntry(
                        kernel=kernel,
                        backend=backend,
                        size=size,
                        path=path,
                        data=data,
                    )
                )
    return entries


def finite_positive(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value) and value > 0.0]


def operation_count(kernel: str, M: int, N: int, K: int) -> float:
    if kernel == "stencil":
        return 6.0 * max(M - 2, 0) * max(N - 2, 0) * K
    return 2.0 * M * N * K


def throughput_gflops(kernel: str, M: int, N: int, K: int, time_ms: float) -> float:
    if not math.isfinite(time_ms) or time_ms <= 0.0:
        return math.nan
    return operation_count(kernel, M, N, K) / (time_ms * 1e6)


def best_so_far(times: list[float]) -> list[float]:
    best = math.inf
    trace = []
    for time_ms in times:
        if not math.isfinite(time_ms) or time_ms <= 0.0:
            continue
        best = min(best, time_ms)
        trace.append(best)
    return trace


def run_trace(run: dict) -> list[float]:
    trace = run.get("best_so_far")
    if trace:
        return finite_positive([float(value) for value in trace])
    return best_so_far([float(value) for value in run.get("times", [])])


def run_best_time(run: dict) -> float:
    if "best_time_ms" in run:
        value = float(run["best_time_ms"])
        if math.isfinite(value) and value > 0.0:
            return value
    trace = run_trace(run)
    return trace[-1] if trace else math.nan


def algorithm_runs(entry: ResultEntry, algorithm: str) -> list[dict]:
    return entry.data.get(f"{algorithm}_runs", [])


def default_time_ms(entry: ResultEntry) -> float:
    """Untuned/default execution time for this (kernel, backend, size).

    Looked up under a few common key names so this works whether the
    orchestrator wrote it as a flat field or nested under a "default"-style
    object:
        {"default_baseline": {"time_ms": 64.34, "mean_time_ms": 64.30}}
        {"default_time_ms": 64.34}
        {"default": {"time_ms": 64.34}}
        {"baseline_time_ms": 64.34}
        {"default": {"best_time_ms": 64.34}}
    """
    data = entry.data
    candidates = [
        data.get("default_time_ms"),
        data.get("baseline_time_ms"),
    ]
    for key in ("default_baseline", "default"):
        obj = data.get(key)
        if isinstance(obj, dict):
            candidates.append(obj.get("time_ms"))
            candidates.append(obj.get("mean_time_ms"))
            candidates.append(obj.get("best_time_ms"))
        elif isinstance(obj, (int, float)):
            candidates.append(obj)

    for value in candidates:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return math.nan


def percentile(values: list[float], pct: float) -> float:
    values = finite_positive(values)
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def sample_summary(values: list[float]) -> dict[str, float | int]:
    values = finite_positive(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * sd / math.sqrt(len(values)) if values else math.nan
    return {
        "n": len(values),
        "mean": mean(values) if values else math.nan,
        "median": median(values) if values else math.nan,
        "std": sd if values else math.nan,
        "ci95": ci95,
        "min": min(values) if values else math.nan,
        "max": max(values) if values else math.nan,
        "q1": percentile(values, 25),
        "q3": percentile(values, 75),
    }


GOOD_CONFIG_REGRET_PCT = 10.0
TOST_MARGIN_PCT = 10.0
CONFIG_FIELDS = ("BM", "BN", "BK", "TM", "TN")


def config_id(evaluation: dict) -> str | None:
    if evaluation.get("config") is not None:
        return str(evaluation["config"])
    if all(key in evaluation for key in CONFIG_FIELDS):
        return "_".join(str(int(evaluation[key])) for key in CONFIG_FIELDS)
    return None


def grid_evaluations(entry: ResultEntry) -> tuple[dict, list[dict]]:
    grid = entry.data.get("grid_search", {})
    evaluations = grid.get("evaluations", []) if isinstance(grid, dict) else []
    by_config = {
        config_id(item): item
        for item in evaluations
        if config_id(item) is not None
        and math.isfinite(float(item.get("time_ms", math.nan)))
        and float(item.get("time_ms", 0.0)) > 0.0
    }
    complete = (
        isinstance(grid, dict)
        and grid.get("status") == "complete"
        and bool(by_config)
        and len(by_config) == int(grid.get("valid_config_count", len(by_config)))
    )
    return {**(grid if isinstance(grid, dict) else {}), "complete": complete}, list(by_config.values())


def evaluation_times(evaluation: dict) -> list[float]:
    stats = evaluation.get("stats", {})
    raw = stats.get("times", []) if isinstance(stats, dict) else []
    return finite_positive([float(value) for value in raw])


def build_grid_analysis(entries: list[ResultEntry]) -> tuple[list[dict], list[dict]]:
    space_rows: list[dict] = []
    config_rows: list[dict] = []
    for entry in entries:
        grid, evaluations = grid_evaluations(entry)
        by_config = {config_id(item): item for item in evaluations}
        valid_count = int(grid.get("valid_config_count", 0) or 0)
        optimum = min((float(item["time_ms"]) for item in evaluations), default=math.nan)
        if not grid["complete"]:
            optimum = math.nan
        optimum_item = min(evaluations, key=lambda item: float(item["time_ms"])) if evaluations else {}
        good_count = 0
        for item in evaluations:
            time_ms = float(item["time_ms"])
            times = evaluation_times(item)
            stats = sample_summary(times)
            _, ci_low, ci_high = mean_ci(times)
            defect_ms = time_ms - optimum if math.isfinite(optimum) else math.nan
            defect_pct = 100.0 * defect_ms / optimum if math.isfinite(optimum) and optimum > 0 else math.nan
            is_good = math.isfinite(defect_pct) and defect_pct <= GOOD_CONFIG_REGRET_PCT
            good_count += int(is_good)
            config_rows.append({
                "kernel": entry.kernel,
                "backend": entry.backend,
                "size": entry.size,
                "config": config_id(item),
                "time_ms": time_ms,
                "measurement_runs": stats["n"],
                "measurement_mean_ms": stats["mean"],
                "measurement_std_ms": stats["std"],
                "measurement_ci95_ms": (ci_high - ci_low) / 2 if math.isfinite(ci_low) and math.isfinite(ci_high) else math.nan,
                "grid_optimum_ms": optimum,
                "defect_ms": defect_ms,
                "defect_pct": defect_pct,
                "good_at_10pct": is_good,
                "grid_complete": grid["complete"],
            })
        optimum_times = evaluation_times(optimum_item) if optimum_item and grid["complete"] else []
        optimum_stats = sample_summary(optimum_times)
        _, optimum_ci_low, optimum_ci_high = mean_ci(optimum_times)
        space_rows.append({
            "kernel": entry.kernel,
            "backend": entry.backend,
            "size": entry.size,
            "M": entry.problem_size[0],
            "N": entry.problem_size[1],
            "K": entry.problem_size[2],
            "grid_status": grid.get("status", "missing"),
            "grid_complete": grid["complete"],
            "valid_config_count": valid_count,
            "evaluated_config_count": int(grid.get("evaluated_config_count", len(evaluations)) or 0),
            "failed_config_count": int(grid.get("failed_config_count", 0) or 0),
            "optimum_config": config_id(optimum_item) if optimum_item else "",
            "optimum_ms": optimum,
            "optimum_noise_runs": optimum_stats["n"],
            "optimum_noise_std_ms": optimum_stats["std"],
            "optimum_noise_ci95_ms": (optimum_ci_high - optimum_ci_low) / 2 if math.isfinite(optimum_ci_low) and math.isfinite(optimum_ci_high) else math.nan,
            "good_config_threshold_pct": GOOD_CONFIG_REGRET_PCT,
            "good_config_count": good_count if grid["complete"] else "",
            "good_config_fraction": good_count / valid_count if grid["complete"] and valid_count else math.nan,
        })
    return space_rows, config_rows


def build_regret_rows(entries: list[ResultEntry]) -> tuple[list[dict], list[dict]]:
    evaluation_rows: list[dict] = []
    campaign_rows: list[dict] = []
    for entry in entries:
        grid, evaluations = grid_evaluations(entry)
        grid_by_config = {config_id(item): float(item["time_ms"]) for item in evaluations}
        optimum = min(grid_by_config.values(), default=math.nan)
        for algorithm in ALGORITHMS:
            for run_index, run in enumerate(algorithm_runs(entry, algorithm)):
                run_idx = int(run.get("run_idx", run_index))
                running_best = math.inf
                seen = 0
                campaign_config = ""
                for evaluation_index, evaluation in enumerate(run.get("evaluations", []), start=1):
                    key = config_id(evaluation)
                    grid_time = grid_by_config.get(key)
                    available = grid["complete"] and grid_time is not None and optimum > 0
                    regret_ms = grid_time - optimum if available else math.nan
                    regret_pct = 100.0 * (grid_time / optimum - 1.0) if available else math.nan
                    if available:
                        seen += 1
                        if regret_ms < running_best:
                            running_best = regret_ms
                            campaign_config = key or ""
                    evaluation_rows.append({
                        "kernel": entry.kernel,
                        "backend": entry.backend,
                        "size": entry.size,
                        "algorithm": algorithm,
                        "run_idx": run_idx,
                        "evaluation": evaluation_index,
                        "config": key or "",
                        "grid_time_ms": grid_time if available else math.nan,
                        "regret_ms": regret_ms,
                        "regret_pct": regret_pct,
                        "best_regret_pct_so_far": 100.0 * (running_best / optimum) if available and math.isfinite(running_best) else math.nan,
                        "grid_status": "ok" if available else "unavailable",
                    })
                campaign_rows.append({
                    "kernel": entry.kernel,
                    "backend": entry.backend,
                    "size": entry.size,
                    "algorithm": algorithm,
                    "run_idx": run_idx,
                    "grid_complete": grid["complete"],
                    "evaluations_with_grid_match": seen,
                    "best_config_by_grid": campaign_config,
                    "best_regret_pct": 100.0 * (running_best / optimum) if seen and optimum > 0 else math.nan,
                })
    return evaluation_rows, campaign_rows


def mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan, math.nan, math.nan
    center = mean(values)
    if len(values) < 2:
        return center, math.nan, math.nan
    sd = stdev(values)
    if scipy_stats is None:
        critical = 1.96
    else:
        critical = float(scipy_stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    half_width = critical * sd / math.sqrt(len(values))
    return center, center - half_width, center + half_width


def paired_test(values: list[float], null: float = 0.0, alternative: str = "two-sided") -> float:
    if scipy_stats is None or len(values) < 2:
        return math.nan
    return float(scipy_stats.ttest_1samp(values, popmean=null, alternative=alternative).pvalue)


def holm_adjust(rows: list[dict], p_field: str, adjusted_field: str) -> None:
    valid = [(index, float(row[p_field])) for index, row in enumerate(rows)
             if isinstance(row.get(p_field), (int, float)) and math.isfinite(float(row[p_field]))]
    ordered = sorted(valid, key=lambda item: item[1])
    count = len(ordered)
    running = 0.0
    adjusted = {}
    for rank, (index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_value))
        adjusted[index] = running
    for index, row in enumerate(rows):
        row[adjusted_field] = adjusted.get(index, math.nan)


def build_algorithm_comparisons(campaign_rows: list[dict]) -> list[dict]:
    comparisons: list[dict] = []
    groups = sorted({(row["kernel"], row["backend"], row["size"]) for row in campaign_rows})
    for kernel, backend, size in groups:
        grouped = {
            algorithm: {int(row["run_idx"]): float(row["best_regret_pct"])
                        for row in campaign_rows
                        if (row["kernel"], row["backend"], row["size"], row["algorithm"]) == (kernel, backend, size, algorithm)
                        and math.isfinite(float(row["best_regret_pct"]))}
            for algorithm in ALGORITHMS
        }
        common = sorted(grouped["bayesian"].keys() & grouped["random"].keys())
        differences = [grouped["bayesian"][idx] - grouped["random"][idx] for idx in common]
        center, ci90_low, ci90_high = mean_ci(differences, confidence=0.90)
        _, ci95_low, ci95_high = mean_ci(differences, confidence=0.95)
        sd = stdev(differences) if len(differences) > 1 else math.nan
        dz = center / sd if len(differences) > 1 and sd > 0 else (0.0 if differences and center == 0 else math.nan)
        p_difference = paired_test(differences)
        p_lower = paired_test(differences, null=-TOST_MARGIN_PCT, alternative="greater")
        p_upper = paired_test(differences, null=TOST_MARGIN_PCT, alternative="less")
        p_tost = max(p_lower, p_upper) if math.isfinite(p_lower) and math.isfinite(p_upper) else math.nan
        comparisons.append({
            "kernel": kernel,
            "backend": backend,
            "size": size,
            "paired_campaigns": len(common),
            "run_indices": ";".join(str(value) for value in common),
            "bayesian_mean_regret_pct": mean(grouped["bayesian"].values()) if grouped["bayesian"] else math.nan,
            "random_mean_regret_pct": mean(grouped["random"].values()) if grouped["random"] else math.nan,
            "mean_difference_bo_minus_rs_pp": center,
            "difference_ci95_low_pp": ci95_low,
            "difference_ci95_high_pp": ci95_high,
            "difference_cohen_dz": dz,
            "difference_p_value": p_difference,
            "difference_holm_p_value": math.nan,
            "tost_margin_pp": TOST_MARGIN_PCT,
            "difference_ci90_low_pp": ci90_low,
            "difference_ci90_high_pp": ci90_high,
            "tost_lower_p_value": p_lower,
            "tost_upper_p_value": p_upper,
            "tost_p_value": p_tost,
            "tost_holm_p_value": math.nan,
            "difference_conclusion": "test_unavailable" if not math.isfinite(p_difference) else ("difference" if p_difference < 0.05 else "no_difference_detected"),
            "equivalence_conclusion": "test_unavailable" if not math.isfinite(p_tost) else ("equivalent" if p_tost < 0.05 else "not_equivalent_or_inconclusive"),
        })
    holm_adjust(comparisons, "difference_p_value", "difference_holm_p_value")
    holm_adjust(comparisons, "tost_p_value", "tost_holm_p_value")
    for row in comparisons:
        row["difference_conclusion"] = (
            "test_unavailable" if not math.isfinite(row["difference_holm_p_value"])
            else ("difference" if row["difference_holm_p_value"] < 0.05 else "no_difference_detected")
        )
        row["equivalence_conclusion"] = (
            "test_unavailable" if not math.isfinite(row["tost_holm_p_value"])
            else ("equivalent" if row["tost_holm_p_value"] < 0.05 else "not_equivalent_or_inconclusive")
        )
    return comparisons


def validation_speedups(entry: ResultEntry, algorithm: str) -> list[dict]:
    rows = []
    for index, run in enumerate(algorithm_runs(entry, algorithm)):
        validation = run.get("validation", {})
        if validation.get("status") != "ok":
            continue
        default_times = validation.get("default_stats", {}).get("times", [])
        tuned_times = validation.get("tuned_stats", {}).get("times", [])
        paired = [(float(default), float(tuned)) for default, tuned in zip(default_times, tuned_times)
                  if math.isfinite(float(default)) and float(default) > 0
                  and math.isfinite(float(tuned)) and float(tuned) > 0]
        speedups = [default / tuned for default, tuned in paired]
        if not speedups:
            continue
        logs = [math.log(value) for value in speedups]
        log_mean, lower, upper = mean_ci(logs, confidence=0.95)
        rows.append({
            "kernel": entry.kernel,
            "backend": entry.backend,
            "size": entry.size,
            "algorithm": algorithm,
            "run_idx": int(run.get("run_idx", index)),
            "validation_seed": validation.get("validation_seed", ""),
            "paired_repetitions": len(speedups),
            "geometric_mean_speedup": math.exp(log_mean),
            "speedup_ci95_low": math.exp(lower) if math.isfinite(lower) else math.nan,
            "speedup_ci95_high": math.exp(upper) if math.isfinite(upper) else math.nan,
            "median_speedup": median(speedups),
            "pairs_faster_than_default": sum(value > 1 for value in speedups),
        })
    return rows


def validation_by_run(entry: ResultEntry, algorithm: str) -> dict[int, dict]:
    return {
        int(run.get("run_idx", index)): run.get("validation", {})
        for index, run in enumerate(algorithm_runs(entry, algorithm))
        if run.get("validation", {}).get("status") == "ok"
    }


def validation_pairs(
    sycl_entry: ResultEntry, cuda_entry: ResultEntry, algorithm: str | None, field: str
) -> list[tuple[float, float]]:
    algorithms = (algorithm,) if algorithm else ("bayesian",)
    pairs: list[tuple[float, float]] = []
    for selected_algorithm in algorithms:
        sycl_runs = validation_by_run(sycl_entry, selected_algorithm)
        cuda_runs = validation_by_run(cuda_entry, selected_algorithm)
        for run_idx in sorted(sycl_runs.keys() & cuda_runs.keys()):
            sycl_validation = sycl_runs[run_idx]
            cuda_validation = cuda_runs[run_idx]
            if sycl_validation.get("validation_seed") != cuda_validation.get("validation_seed"):
                continue
            sycl_times = sycl_validation.get(f"{field}_stats", {}).get("times", [])
            cuda_times = cuda_validation.get(f"{field}_stats", {}).get("times", [])
            pairs.extend(
                (float(sycl_time), float(cuda_time))
                for sycl_time, cuda_time in zip(sycl_times, cuda_times)
                if math.isfinite(float(sycl_time)) and float(sycl_time) > 0
                and math.isfinite(float(cuda_time)) and float(cuda_time) > 0
            )
    return pairs


def efficiency_row(
    kernel: str, size: int, config: str, pairs: list[tuple[float, float]], source: str
) -> dict:
    ratios = [100.0 * cuda_time / sycl_time for sycl_time, cuda_time in pairs]
    stats = sample_summary(ratios)
    return {
        "kernel": kernel,
        "size": size,
        "config": config,
        "source": source,
        "paired_measurements": len(ratios),
        "sycl_time_ms": mean(sycl_time for sycl_time, _ in pairs) if pairs else math.nan,
        "cuda_time_ms": mean(cuda_time for _, cuda_time in pairs) if pairs else math.nan,
        "efficiency_pct": stats["mean"],
        "median_efficiency_pct": stats["median"],
        "std_efficiency_pct": stats["std"],
        "ci95_efficiency_pct": stats["ci95"],
    }


def grid_time_pairs(sycl_entry: ResultEntry, cuda_entry: ResultEntry, config: str) -> list[tuple[float, float]]:
    sycl_grid, sycl_evals = grid_evaluations(sycl_entry)
    cuda_grid, cuda_evals = grid_evaluations(cuda_entry)
    if not sycl_grid["complete"] or not cuda_grid["complete"]:
        return []
    sycl_eval = next((item for item in sycl_evals if config_id(item) == config), None)
    cuda_eval = next((item for item in cuda_evals if config_id(item) == config), None)
    if sycl_eval is None or cuda_eval is None:
        return []
    return list(zip(evaluation_times(sycl_eval), evaluation_times(cuda_eval)))


def build_transferability_rows(entries: list[ResultEntry]) -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    transfer_rows: list[dict] = []
    entry_by_key = {(entry.kernel, entry.size, entry.backend): entry for entry in entries}
    scopes = sorted({(entry.kernel, entry.size) for entry in entries})
    for kernel, size in scopes:
        sycl_entry = entry_by_key.get((kernel, size, "sycl"))
        cuda_entry = entry_by_key.get((kernel, size, "cuda"))
        if sycl_entry is None or cuda_entry is None:
            summary_rows.append({"kernel": kernel, "size": size, "status": "backend_missing"})
            continue
        sycl_grid, sycl_evals = grid_evaluations(sycl_entry)
        cuda_grid, cuda_evals = grid_evaluations(cuda_entry)
        sycl_by_config = {config_id(item): float(item["time_ms"]) for item in sycl_evals}
        cuda_by_config = {config_id(item): float(item["time_ms"]) for item in cuda_evals}
        shared = sorted(sycl_by_config.keys() & cuda_by_config.keys())
        same_space = (
            sycl_grid["complete"] and cuda_grid["complete"]
            and len(shared) == len(sycl_by_config) == len(cuda_by_config)
        )
        rho = math.nan
        if same_space and len(shared) > 1 and scipy_stats is not None:
            rho = float(scipy_stats.spearmanr(
                [sycl_by_config[key] for key in shared],
                [cuda_by_config[key] for key in shared],
            ).statistic)
        summary_rows.append({
            "kernel": kernel,
            "size": size,
            "status": "ok" if same_space else "spaces_not_identical_or_grid_incomplete",
            "sycl_config_count": len(sycl_by_config),
            "cuda_config_count": len(cuda_by_config),
            "shared_config_count": len(shared),
            "same_config_space": same_space,
            "spearman_time_correlation": rho,
        })
        if not same_space:
            continue
        for source_backend, source_times, target_backend, target_times in (
            ("sycl", sycl_by_config, "cuda", cuda_by_config),
            ("cuda", cuda_by_config, "sycl", sycl_by_config),
        ):
            source_optimum_config = min(source_times, key=source_times.get)
            target_optimum_time = min(target_times.values())
            transferred_time = target_times[source_optimum_config]
            transfer_rows.append({
                "kernel": kernel,
                "size": size,
                "source_backend": source_backend,
                "target_backend": target_backend,
                "transferred_config": source_optimum_config,
                "source_optimum_ms": source_times[source_optimum_config],
                "target_transferred_ms": transferred_time,
                "target_optimum_ms": target_optimum_time,
                "target_regret_ms": transferred_time - target_optimum_time,
                "target_regret_pct": 100.0 * (transferred_time / target_optimum_time - 1.0),
                "target_over_source_time_ratio": transferred_time / source_times[source_optimum_config],
            })
    return summary_rows, transfer_rows


def convergence_summary(entry: ResultEntry, algorithm: str) -> list[dict]:
    traces = [run_trace(run) for run in algorithm_runs(entry, algorithm)]
    traces = [trace for trace in traces if trace]
    if not traces:
        return []

    width = min(len(trace) for trace in traces)
    rows = []
    for idx in range(width):
        values = [trace[idx] for trace in traces]
        stats = sample_summary(values)
        rows.append(
            {
                "kernel": entry.kernel,
                "backend": entry.backend,
                "size": entry.size,
                "algorithm": algorithm,
                "trial": idx + 1,
                "mean_best_ms": stats["mean"],
                "median_best_ms": stats["median"],
                "std_best_ms": stats["std"],
                "ci95_ms": stats["ci95"],
                "min_best_ms": stats["min"],
                "max_best_ms": stats["max"],
            }
        )
    return rows


def performance_summary(entry: ResultEntry, algorithm: str) -> dict:
    M, N, K = entry.problem_size
    best_times = [run_best_time(run) for run in algorithm_runs(entry, algorithm)]
    best_times = finite_positive(best_times)
    gflops_values = [
        throughput_gflops(entry.kernel, M, N, K, time_ms) for time_ms in best_times
    ]
    time_stats = sample_summary(best_times)
    gflops_stats = sample_summary(gflops_values)
    return {
        "kernel": entry.kernel,
        "backend": entry.backend,
        "size": entry.size,
        "M": M,
        "N": N,
        "K": K,
        "algorithm": algorithm,
        "runs": time_stats["n"],
        "mean_best_ms": time_stats["mean"],
        "median_best_ms": time_stats["median"],
        "std_best_ms": time_stats["std"],
        "ci95_best_ms": time_stats["ci95"],
        "best_observed_ms": time_stats["min"],
        "mean_gflops": gflops_stats["mean"],
        "median_gflops": gflops_stats["median"],
        "std_gflops": gflops_stats["std"],
        "ci95_gflops": gflops_stats["ci95"],
        "best_observed_gflops": gflops_stats["max"],
        "default_ms": default_time_ms(entry),
    }


def build_tables(entries: list[ResultEntry]) -> tuple[list[dict], list[dict]]:
    performance_rows = []
    convergence_rows = []
    for entry in entries:
        for algorithm in ALGORITHMS:
            if not algorithm_runs(entry, algorithm):
                continue
            performance_rows.append(performance_summary(entry, algorithm))
            convergence_rows.extend(convergence_summary(entry, algorithm))
    return performance_rows, convergence_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grid_shape(count: int) -> tuple[int, int]:
    if count <= 1:
        return 1, 1
    cols = 2
    rows = math.ceil(count / cols)
    return rows, cols


def rows_for(rows: list[dict], **filters: object) -> list[dict]:
    out = rows
    for key, value in filters.items():
        out = [row for row in out if row.get(key) == value]
    return sorted(out, key=lambda row: row.get("trial", row.get("size", 0)))


def plot_convergence(
    convergence_rows: list[dict], output_dir: Path, log_y: bool
) -> list[Path]:
    require_plot_deps()
    outputs = []
    colors = {"bayesian": "#1f77b4", "random": "#d62728"}
    markers = {"bayesian": "o", "random": "s"}

    for kernel in KERNELS:
        for backend in BACKENDS:
            subset = rows_for(convergence_rows, kernel=kernel, backend=backend)
            sizes = sorted({row["size"] for row in subset})
            if not sizes:
                continue

            nrows, ncols = grid_shape(len(sizes))
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(6.2 * ncols, 4.2 * nrows),
                squeeze=False,
            )
            axes_flat = [ax for row_axes in axes for ax in row_axes]

            for ax, size in zip(axes_flat, sizes):
                for algorithm in ALGORITHMS:
                    algo_rows = rows_for(subset, size=size, algorithm=algorithm)
                    if not algo_rows:
                        continue
                    trials = [row["trial"] for row in algo_rows]
                    means = [row["mean_best_ms"] for row in algo_rows]
                    errors = [row["ci95_ms"] for row in algo_rows]
                    ax.errorbar(
                        trials,
                        means,
                        yerr=errors,
                        color=colors[algorithm],
                        marker=markers[algorithm],
                        markevery=max(len(trials) // 8, 1),
                        linewidth=1.8,
                        markersize=4,
                        capsize=2,
                        label=ALGORITHM_LABELS[algorithm],
                    )

                ax.set_title(f"N = {size}")
                ax.set_xlabel("Evaluation")
                ax.set_ylabel("Best-so-far time (ms)")
                if log_y:
                    ax.set_yscale("log")
                ax.grid(True, linestyle=":", linewidth=0.7)

            for ax in axes_flat[len(sizes) :]:
                ax.axis("off")

            handles, labels = axes_flat[0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.945),
                    ncol=2,
                    frameon=False,
                    borderaxespad=0.0,
                )
            fig.suptitle(
                f"{BACKEND_LABELS[backend]} convergence - {KERNEL_LABELS[kernel]}",
                y=0.99,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            output_path = output_dir / f"convergence_{kernel}_{backend}.pdf"
            fig.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            outputs.append(output_path)

    return outputs


def plot_performance(performance_rows: list[dict], output_dir: Path) -> list[Path]:
    require_plot_deps()
    outputs = []
    colors = {"sycl": "#1f77b4", "cuda": "#d62728"}
    linestyles = {"bayesian": "-", "random": "--"}
    markers = {"bayesian": "o", "random": "s"}

    for kernel in KERNELS:
        subset = rows_for(performance_rows, kernel=kernel)
        sizes = sorted({row["size"] for row in subset})
        if not sizes:
            continue

        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        x_positions = list(range(len(sizes)))

        for backend in BACKENDS:
            for algorithm in ALGORITHMS:
                series_rows = rows_for(subset, backend=backend, algorithm=algorithm)
                by_size = {row["size"]: row for row in series_rows}
                y_values = [
                    by_size[size]["mean_gflops"] if size in by_size else math.nan
                    for size in sizes
                ]
                y_errors = [
                    by_size[size]["ci95_gflops"] if size in by_size else math.nan
                    for size in sizes
                ]
                if all(not math.isfinite(value) for value in y_values):
                    continue
                ax.errorbar(
                    x_positions,
                    y_values,
                    yerr=y_errors,
                    color=colors[backend],
                    linestyle=linestyles[algorithm],
                    marker=markers[algorithm],
                    linewidth=2.0,
                    markersize=5,
                    capsize=3,
                    label=f"{BACKEND_LABELS[backend]} {ALGORITHM_LABELS[algorithm]}",
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.set_xlabel("Matrix size N x N")
        ax.set_ylabel("Performance (GFLOP/s)")
        ax.set_title(f"SYCL vs CUDA performance - {KERNEL_LABELS[kernel]}")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7)
        ax.legend(frameon=False)
        fig.tight_layout()

        output_path = output_dir / f"performance_sycl_vs_cuda_{kernel}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        outputs.append(output_path)

    return outputs


def plot_time(
    performance_rows: list[dict], output_dir: Path, log_y: bool
) -> list[Path]:
    require_plot_deps()
    outputs = []
    colors = {"sycl": "#1f77b4", "cuda": "#d62728"}
    linestyles = {"bayesian": "-", "random": "--"}
    markers = {"bayesian": "o", "random": "s"}

    for kernel in KERNELS:
        subset = rows_for(performance_rows, kernel=kernel)
        sizes = sorted({row["size"] for row in subset})
        if not sizes:
            continue

        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        x_positions = list(range(len(sizes)))

        for backend in BACKENDS:
            for algorithm in ALGORITHMS:
                series_rows = rows_for(subset, backend=backend, algorithm=algorithm)
                by_size = {row["size"]: row for row in series_rows}
                y_values = [
                    by_size[size]["mean_best_ms"] if size in by_size else math.nan
                    for size in sizes
                ]
                y_errors = [
                    by_size[size]["ci95_best_ms"] if size in by_size else math.nan
                    for size in sizes
                ]
                if all(not math.isfinite(value) for value in y_values):
                    continue
                ax.errorbar(
                    x_positions,
                    y_values,
                    yerr=y_errors,
                    color=colors[backend],
                    linestyle=linestyles[algorithm],
                    marker=markers[algorithm],
                    linewidth=2.0,
                    markersize=5,
                    capsize=3,
                    label=f"{BACKEND_LABELS[backend]} {ALGORITHM_LABELS[algorithm]}",
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.set_xlabel("Matrix size N x N")
        ax.set_ylabel("Mean final time (ms)")
        ax.set_title(f"SYCL vs CUDA time - {KERNEL_LABELS[kernel]}")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7)
        ax.legend(frameon=False)
        fig.tight_layout()

        output_path = output_dir / f"time_sycl_vs_cuda_{kernel}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        outputs.append(output_path)

    return outputs


def plot_cuda_sycl_ratio(performance_rows: list[dict], output_dir: Path) -> list[Path]:
    require_plot_deps()
    outputs = []
    colors = {"bayesian": "#2ca02c", "random": "#9467bd"}

    for kernel in KERNELS:
        subset = rows_for(performance_rows, kernel=kernel)
        sizes = sorted({row["size"] for row in subset})
        if not sizes:
            continue

        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        width = 0.34
        base_positions = list(range(len(sizes)))
        plotted = False

        for offset_idx, algorithm in enumerate(ALGORITHMS):
            ratios = []
            for size in sizes:
                sycl_rows = rows_for(
                    subset, size=size, backend="sycl", algorithm=algorithm
                )
                cuda_rows = rows_for(
                    subset, size=size, backend="cuda", algorithm=algorithm
                )
                if not sycl_rows or not cuda_rows:
                    ratios.append(math.nan)
                    continue
                sycl_perf = sycl_rows[0]["mean_gflops"]
                cuda_perf = cuda_rows[0]["mean_gflops"]
                ratios.append(
                    cuda_perf / sycl_perf
                    if sycl_perf and math.isfinite(sycl_perf)
                    else math.nan
                )

            if all(not math.isfinite(value) for value in ratios):
                continue

            positions = [pos + (offset_idx - 0.5) * width for pos in base_positions]
            ax.bar(
                positions,
                [value if math.isfinite(value) else 0.0 for value in ratios],
                width=width,
                color=colors[algorithm],
                label=ALGORITHM_LABELS[algorithm],
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.axhline(1.0, color="#444444", linewidth=1.0)
        ax.set_xticks(base_positions)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.set_xlabel("Matrix size N x N")
        ax.set_ylabel("CUDA / SYCL performance ratio")
        ax.set_title(f"CUDA/SYCL performance ratio - {KERNEL_LABELS[kernel]}")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7)
        ax.legend(frameon=False)
        fig.tight_layout()

        output_path = output_dir / f"cuda_over_sycl_ratio_{kernel}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        outputs.append(output_path)

    return outputs


EFFICIENCY_CONFIGS = ("default", "random", "bayesian", "grid_optimum")

EFFICIENCY_LABELS = {
    "default": "Default / Default",
    "random": "Random / Random",
    "bayesian": "BO / BO",
    "grid_optimum": "Grid optimum / Grid optimum",
}

EFFICIENCY_COLORS = {
    "default": "#7f7f7f",
    "random": "#2ca02c",
    "bayesian": "#1f77b4",
    "grid_optimum": "#d62728",
}


def exact_sign_test_p_value(above: int, below: int) -> float:
    """Two-sided exact sign-test p-value, excluding ties."""
    n = above + below
    if n == 0:
        return math.nan
    tail = min(above, below)
    probability = sum(math.comb(n, index) for index in range(tail + 1)) / 2**n
    return min(1.0, 2.0 * probability)


def build_efficiency_rows(
    performance_rows: list[dict], entries: list[ResultEntry]
) -> list[dict]:
    """Compare default/default, tuned/tuned, and grid-optimum/grid-optimum."""
    rows: list[dict] = []
    entry_by_key = {
        (entry.kernel, entry.size, entry.backend): entry for entry in entries
    }
    for kernel in KERNELS:
        sizes = sorted({entry.size for entry in entries if entry.kernel == kernel})

        for size in sizes:
            sycl_entry = entry_by_key.get((kernel, size, "sycl"))
            cuda_entry = entry_by_key.get((kernel, size, "cuda"))
            if sycl_entry is None or cuda_entry is None:
                continue
            default_pairs = validation_pairs(sycl_entry, cuda_entry, None, "default")
            if default_pairs:
                rows.append(efficiency_row(kernel, size, "default", default_pairs, "20-pair validation"))
            for algorithm in ("random", "bayesian"):
                tuned_pairs = validation_pairs(sycl_entry, cuda_entry, algorithm, "tuned")
                if tuned_pairs:
                    rows.append(efficiency_row(kernel, size, algorithm, tuned_pairs, "20-pair validation"))
            sycl_grid, sycl_evals = grid_evaluations(sycl_entry)
            cuda_grid, cuda_evals = grid_evaluations(cuda_entry)
            if sycl_grid["complete"] and cuda_grid["complete"]:
                sycl_optimum = min(sycl_evals, key=lambda item: float(item["time_ms"]))
                cuda_optimum = min(cuda_evals, key=lambda item: float(item["time_ms"]))
                sycl_times = evaluation_times(sycl_optimum)
                cuda_times = evaluation_times(cuda_optimum)
                optimum_pairs = list(zip(sycl_times, cuda_times))
                if optimum_pairs:
                    rows.append(efficiency_row(kernel, size, "grid_optimum", optimum_pairs, "independent grid optima; grid repeats"))
    return rows


def plot_application_efficiency(
    efficiency_rows: list[dict], output_dir: Path
) -> list[Path]:
    """Plot symmetric CUDA/SYCL configuration comparisons per matrix size."""
    require_plot_deps()
    outputs = []

    for kernel in KERNELS:
        subset = [row for row in efficiency_rows if row["kernel"] == kernel]
        sizes = sorted({row["size"] for row in subset})
        if not sizes:
            continue

        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        width = 0.26
        base_positions = list(range(len(sizes)))
        plotted = False

        for offset_idx, config in enumerate(EFFICIENCY_CONFIGS):
            by_size = {
                row["size"]: row["efficiency_pct"]
                for row in subset
                if row["config"] == config
            }
            values = [by_size.get(size, math.nan) for size in sizes]
            if all(not math.isfinite(value) for value in values):
                continue

            positions = [
                pos + (offset_idx - 1) * width for pos in base_positions
            ]
            bars = ax.bar(
                positions,
                [value if math.isfinite(value) else 0.0 for value in values],
                width=width,
                color=EFFICIENCY_COLORS[config],
                label=EFFICIENCY_LABELS[config],
            )
            for bar, value in zip(bars, values):
                if math.isfinite(value):
                    ax.annotate(
                        f"{value:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        max_value = max(
            (
                row["efficiency_pct"]
                for row in subset
                if math.isfinite(row["efficiency_pct"])
            ),
            default=100.0,
        )
        ax.set_ylim(0, max(105.0, max_value * 1.12))
        ax.axhline(100.0, color="#444444", linewidth=1.0)
        ax.set_xticks(base_positions)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.set_xlabel("Matrix size N x N")
        ax.set_ylabel("Application efficiency E_app = CUDA / SYCL (%)")
        ax.set_title(
            f"Matched CUDA/SYCL Application Efficiency - {KERNEL_LABELS[kernel]}"
        )
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        fig.tight_layout()

        output_path = output_dir / f"application_efficiency_{kernel}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        outputs.append(output_path)

    return outputs


def print_summary(
    results_dir: Path, output_dir: Path, entries: list[ResultEntry], outputs: list[Path]
) -> None:
    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Loaded JSON : {len(entries)} convergence files")
    print("\nInput files:")
    for entry in sorted(
        entries, key=lambda item: (item.kernel, item.size, item.backend)
    ):
        print(
            f"  - {entry.kernel:7s} size={entry.size:<6d} {entry.backend:4s} {entry.path}"
        )
    print("\nGenerated artifacts:")
    for path in outputs:
        print(f"  - {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze SYCL/CUDA autotuning results and generate PDF plots."
    )
    parser.add_argument(
        "--input",
        nargs="?",
        type=Path,
        help="Results directory. Defaults to the newest results_* folder in the current directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for CSV and PDF artifacts. Default: RESULTS_DIR/eps_analysis",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=KERNELS,
        default=list(KERNELS),
        help="Kernels to analyze.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS,
        default=list(BACKENDS),
        help="Backends to analyze.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic y axis for convergence and time plots.",
    )
    args = parser.parse_args()

    results_dir = args.input or latest_results_dir(Path.cwd())
    results_dir = results_dir.resolve()
    output_dir = (args.output or (results_dir / "eps_analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = discover_entries(results_dir, args.kernels, args.backends)
    if not entries:
        raise FileNotFoundError(
            f"No convergence JSON files were found under {results_dir} for "
            f"kernels={args.kernels} and backends={args.backends}."
        )

    performance_rows, convergence_rows = build_tables(entries)

    performance_csv = output_dir / "performance_summary.csv"
    convergence_csv = output_dir / "convergence_summary.csv"
    write_csv(
        performance_csv,
        performance_rows,
        [
            "kernel",
            "backend",
            "size",
            "M",
            "N",
            "K",
            "algorithm",
            "runs",
            "mean_best_ms",
            "median_best_ms",
            "std_best_ms",
            "ci95_best_ms",
            "best_observed_ms",
            "mean_gflops",
            "median_gflops",
            "std_gflops",
            "ci95_gflops",
            "best_observed_gflops",
            "default_ms",
        ],
    )
    write_csv(
        convergence_csv,
        convergence_rows,
        [
            "kernel",
            "backend",
            "size",
            "algorithm",
            "trial",
            "mean_best_ms",
            "median_best_ms",
            "std_best_ms",
            "ci95_ms",
            "min_best_ms",
            "max_best_ms",
        ],
    )

    efficiency_rows = build_efficiency_rows(performance_rows, entries)
    efficiency_csv = output_dir / "application_efficiency.csv"
    write_csv(
        efficiency_csv,
        efficiency_rows,
        [
            "kernel",
            "size",
            "config",
            "source",
            "paired_measurements",
            "sycl_time_ms",
            "cuda_time_ms",
            "efficiency_pct",
            "median_efficiency_pct",
            "std_efficiency_pct",
            "ci95_efficiency_pct",
        ],
    )

    space_rows, grid_config_rows = build_grid_analysis(entries)
    regret_evaluation_rows, campaign_regret_rows = build_regret_rows(entries)
    comparison_rows = build_algorithm_comparisons(campaign_regret_rows)
    transfer_rows, transfer_detail_rows = build_transferability_rows(entries)
    speedup_rows = [
        row
        for entry in entries
        for algorithm in ALGORITHMS
        for row in validation_speedups(entry, algorithm)
    ]

    analysis_csvs = [
        ("search_space_summary.csv", space_rows, [
            "kernel", "backend", "size", "M", "N", "K", "grid_status",
            "grid_complete", "valid_config_count", "evaluated_config_count",
            "failed_config_count", "optimum_config", "optimum_ms",
            "optimum_noise_runs", "optimum_noise_std_ms", "optimum_noise_ci95_ms",
            "good_config_threshold_pct", "good_config_count", "good_config_fraction",
        ]),
        ("grid_configuration_defects.csv", grid_config_rows, [
            "kernel", "backend", "size", "config", "time_ms", "measurement_runs",
            "measurement_mean_ms", "measurement_std_ms", "measurement_ci95_ms",
            "grid_optimum_ms", "defect_ms", "defect_pct", "good_at_10pct", "grid_complete",
        ]),
        ("grid_regret_evaluations.csv", regret_evaluation_rows, [
            "kernel", "backend", "size", "algorithm", "run_idx", "evaluation",
            "config", "grid_time_ms", "regret_ms", "regret_pct",
            "best_regret_pct_so_far", "grid_status",
        ]),
        ("grid_regret_campaigns.csv", campaign_regret_rows, [
            "kernel", "backend", "size", "algorithm", "run_idx", "grid_complete",
            "evaluations_with_grid_match", "best_config_by_grid", "best_regret_pct",
        ]),
        ("bo_vs_rs_grid_regret.csv", comparison_rows, [
            "kernel", "backend", "size", "paired_campaigns", "run_indices",
            "bayesian_mean_regret_pct", "random_mean_regret_pct",
            "mean_difference_bo_minus_rs_pp", "difference_ci95_low_pp",
            "difference_ci95_high_pp", "difference_cohen_dz", "difference_p_value",
            "difference_holm_p_value", "tost_margin_pp", "difference_ci90_low_pp",
            "difference_ci90_high_pp", "tost_lower_p_value", "tost_upper_p_value",
            "tost_p_value", "tost_holm_p_value", "difference_conclusion",
            "equivalence_conclusion",
        ]),
        ("transferability_summary.csv", transfer_rows, [
            "kernel", "size", "status", "sycl_config_count", "cuda_config_count",
            "shared_config_count", "same_config_space", "spearman_time_correlation",
        ]),
        ("optimum_transfer.csv", transfer_detail_rows, [
            "kernel", "size", "source_backend", "target_backend", "transferred_config",
            "source_optimum_ms", "target_transferred_ms", "target_optimum_ms",
            "target_regret_ms", "target_regret_pct", "target_over_source_time_ratio",
        ]),
        ("validated_speedups.csv", speedup_rows, [
            "kernel", "backend", "size", "algorithm", "run_idx", "validation_seed",
            "paired_repetitions", "geometric_mean_speedup", "speedup_ci95_low",
            "speedup_ci95_high", "median_speedup", "pairs_faster_than_default",
        ]),
    ]
    analysis_csv_paths = []
    for filename, rows, fields in analysis_csvs:
        path = output_dir / filename
        write_csv(path, rows, fields)
        analysis_csv_paths.append(path)

    outputs = [performance_csv, convergence_csv, efficiency_csv, *analysis_csv_paths]
    outputs.extend(plot_convergence(convergence_rows, output_dir, args.log_y))
    outputs.extend(plot_performance(performance_rows, output_dir))
    outputs.extend(plot_time(performance_rows, output_dir, args.log_y))
    outputs.extend(plot_cuda_sycl_ratio(performance_rows, output_dir))
    outputs.extend(plot_application_efficiency(efficiency_rows, output_dir))

    if not efficiency_rows:
        print(
            "\nWARNING: no paired default/tuned validation samples or complete grid "
            "optima were found, so application-efficiency comparisons are unavailable."
        )

    print_summary(results_dir, output_dir, entries, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
