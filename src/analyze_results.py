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

plt = None

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
            output_path = output_dir / f"convergence_{kernel}_{backend}.eps"
            fig.savefig(output_path, format="eps", bbox_inches="tight")
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

        output_path = output_dir / f"performance_sycl_vs_cuda_{kernel}.eps"
        fig.savefig(output_path, format="eps", bbox_inches="tight")
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

        output_path = output_dir / f"time_sycl_vs_cuda_{kernel}.eps"
        fig.savefig(output_path, format="eps", bbox_inches="tight")
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

        output_path = output_dir / f"cuda_over_sycl_ratio_{kernel}.eps"
        fig.savefig(output_path, format="eps", bbox_inches="tight")
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
        description="Analyze SYCL/CUDA autotuning results and generate EPS plots."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        help="Results directory. Defaults to the newest results_* folder in the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV and EPS artifacts. Default: RESULTS_DIR/eps_analysis",
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

    results_dir = args.results_dir or latest_results_dir(Path.cwd())
    results_dir = results_dir.resolve()
    output_dir = (args.output_dir or (results_dir / "eps_analysis")).resolve()
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

    outputs = [performance_csv, convergence_csv]
    outputs.extend(plot_convergence(convergence_rows, output_dir, args.log_y))
    outputs.extend(plot_performance(performance_rows, output_dir))
    outputs.extend(plot_time(performance_rows, output_dir, args.log_y))
    outputs.extend(plot_cuda_sycl_ratio(performance_rows, output_dir))

    print_summary(results_dir, output_dir, entries, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
