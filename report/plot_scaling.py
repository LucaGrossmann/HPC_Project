"""Generate scaling and comparison plots from report/results.csv.

Per the project spec, each implementation owns its own results +
plots, and the top-level report/ holds cross-implementation figures.
This script writes:

  Per-implementation:
    serial/results/serial_size_scaling.pdf
    openmp/results/omp_strong.pdf
    openmp/results/omp_weak.pdf
    mpi/results/mpi_strong.pdf
    mpi/results/mpi_weak.pdf
    additional/results/julia_size_scaling.pdf

  Cross-implementation (report/figures/):
    strong_scaling.pdf
    weak_scaling.pdf
    efficiency_bars.pdf
    cross_impl_comparison.pdf

Run from the repo root:
    conda run -n claude python report/plot_scaling.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "report" / "results.csv"

# Consistent styling across plots
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLOR_OPENMP = "#1f77b4"
COLOR_MPI    = "#d62728"
COLOR_SERIAL = "#2ca02c"
COLOR_JULIA  = "#9467bd"
COLOR_CUPY   = "#ff7f0e"
COLOR_IDEAL  = "#888888"


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            r["m"]         = int(r["m"])
            r["p"]         = int(r["p"])
            r["n"]         = int(r["n"])
            r["workers"]   = int(r["workers"])
            r["reps"]      = int(r["reps"])
            r["rel_err"]   = float(r["rel_err"])
            r["median_ms"] = float(r["median_ms"])
            r["min_ms"]    = float(r["min_ms"])
            r["max_ms"]    = float(r["max_ms"])
            r["is_weak"]   = (r["is_weak"] == "True")
            rows.append(r)
    return rows


def filt(rows, **conds) -> list[dict]:
    out = list(rows)
    for k, v in conds.items():
        out = [r for r in out if r[k] == v]
    return out


def by_workers(rows, sort_key="workers") -> tuple[list[int], list[float]]:
    rows = sorted(rows, key=lambda r: r[sort_key])
    return [r["workers"] for r in rows], [r["median_ms"] for r in rows]


# ---------------------------------------------------------------------------
# Per-impl plots
# ---------------------------------------------------------------------------

def plot_omp_strong(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, fixture, title in (
        (axes[0], "medium", "OpenMP strong scaling — medium (256³)"),
        (axes[1], "large",  "OpenMP strong scaling — large (512×512×256)"),
    ):
        sub = filt(rows, impl="openmp", fixture=fixture, is_weak=False)
        ts, ys = by_workers(sub)
        if not ts:
            continue
        ax.plot(ts, ys, "o-", color=COLOR_OPENMP, lw=2, ms=7, label="measured")
        ideal = [ys[0] / t for t in ts]
        ax.plot(ts, ideal, "--", color=COLOR_IDEAL, lw=1, label="ideal speedup")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(ts); ax.set_xticklabels([str(t) for t in ts])
        ax.set_xlabel("threads"); ax.set_ylabel("median wall time (ms)")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    out = REPO / "openmp" / "results" / "omp_strong.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def plot_omp_weak(rows: list[dict]) -> None:
    sub = filt(rows, impl="openmp", is_weak=True)
    sub = sorted(sub, key=lambda r: r["workers"])
    if not sub:
        return
    ts = [r["workers"] for r in sub]
    ys = [r["median_ms"] for r in sub]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, ys, "o-", color=COLOR_OPENMP, lw=2, ms=7, label="measured")
    ax.axhline(ys[0], ls="--", color=COLOR_IDEAL, lw=1, label="ideal (flat)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ts); ax.set_xticklabels([str(t) for t in ts])
    ax.set_xlabel("threads (n scales 32·T)"); ax.set_ylabel("median wall time (ms)")
    ax.set_title("OpenMP weak scaling (256×256×n)")
    ax.legend()
    fig.tight_layout()
    out = REPO / "openmp" / "results" / "omp_weak.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def plot_mpi_strong(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, fixture, title in (
        (axes[0], "medium", "MPI strong scaling — medium (256³)"),
        (axes[1], "large",  "MPI strong scaling — large (512×512×256)"),
    ):
        sub = filt(rows, impl="mpi", fixture=fixture, is_weak=False)
        ts, ys = by_workers(sub)
        if not ts:
            continue
        ax.plot(ts, ys, "s-", color=COLOR_MPI, lw=2, ms=7, label="measured")
        ideal = [ys[0] / t for t in ts]
        ax.plot(ts, ideal, "--", color=COLOR_IDEAL, lw=1, label="ideal speedup")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(ts); ax.set_xticklabels([str(t) for t in ts])
        ax.set_xlabel("ranks (1 per node)"); ax.set_ylabel("median wall time (ms)")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    out = REPO / "mpi" / "results" / "mpi_strong.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def plot_mpi_weak(rows: list[dict]) -> None:
    sub = sorted(filt(rows, impl="mpi", is_weak=True), key=lambda r: r["workers"])
    if not sub:
        return
    ts = [r["workers"] for r in sub]
    ys = [r["median_ms"] for r in sub]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, ys, "s-", color=COLOR_MPI, lw=2, ms=7, label="measured")
    ax.axhline(ys[0], ls="--", color=COLOR_IDEAL, lw=1, label="ideal (flat)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ts); ax.set_xticklabels([str(t) for t in ts])
    ax.set_xlabel("ranks (n scales 32·P)"); ax.set_ylabel("median wall time (ms)")
    ax.set_title("MPI weak scaling (256×256×n)")
    ax.legend()
    fig.tight_layout()
    out = REPO / "mpi" / "results" / "mpi_weak.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def plot_serial_size_scaling(rows: list[dict]) -> None:
    sub = sorted(filt(rows, impl="serial", is_weak=False),
                 key=lambda r: r["m"] * r["p"] * r["n"])
    if not sub:
        return
    labels = [r["fixture"] for r in sub]
    ys = [r["median_ms"] for r in sub]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, ys, color=COLOR_SERIAL)
    for i, (lbl, y) in enumerate(zip(labels, ys)):
        ax.text(i, y, f"{y:.0f} ms", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("median wall time (ms)")
    ax.set_title("Serial C++ — runtime by problem size")
    # Annotate that 'large' was not measured (timed out at 5 min).
    ax.text(0.98, 0.98,
            "large (512×512×256): timed out at the 5 min wall cap",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, style="italic", color="#555")
    fig.tight_layout()
    out = REPO / "serial" / "results" / "serial_size_scaling.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def plot_julia_size_scaling(rows: list[dict]) -> None:
    sub = sorted(filt(rows, impl="julia", is_weak=False),
                 key=lambda r: r["m"] * r["p"] * r["n"])
    # Drop the cross-check row (reps=1) by keeping the first entry per fixture.
    seen: set[str] = set()
    uniq: list[dict] = []
    for r in sub:
        if r["fixture"] in seen:
            continue
        seen.add(r["fixture"])
        uniq.append(r)
    if not uniq:
        return
    labels = [r["fixture"] for r in uniq]
    ys     = [r["median_ms"] for r in uniq]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, ys, color=COLOR_JULIA)
    for i, (_, y) in enumerate(zip(labels, ys)):
        ax.text(i, y, f"{y:.0f} ms", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("median wall time (ms, log)")
    ax.set_title("Julia (stdlib) — runtime by problem size")
    fig.tight_layout()
    out = REPO / "additional" / "results" / "julia_size_scaling.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# Cross-impl plots (report/figures/)
# ---------------------------------------------------------------------------

def plot_strong_combined(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, fixture, title in (
        (axes[0], "medium", "Strong scaling — medium (256³)"),
        (axes[1], "large",  "Strong scaling — large (512×512×256)"),
    ):
        omp = filt(rows, impl="openmp", fixture=fixture, is_weak=False)
        mpi = filt(rows, impl="mpi",    fixture=fixture, is_weak=False)
        omp_ts, omp_ys = by_workers(omp)
        mpi_ts, mpi_ys = by_workers(mpi)
        baseline = omp_ys[0] if omp_ys else (mpi_ys[0] if mpi_ys else None)
        if omp_ts:
            ax.plot(omp_ts, omp_ys, "o-", color=COLOR_OPENMP, lw=2, ms=7, label="OpenMP")
        if mpi_ts:
            ax.plot(mpi_ts, mpi_ys, "s-", color=COLOR_MPI, lw=2, ms=7, label="MPI")
        if baseline:
            ts_ref = sorted(set(omp_ts + mpi_ts))
            ideal = [baseline / t for t in ts_ref]
            ax.plot(ts_ref, ideal, "--", color=COLOR_IDEAL, lw=1, label="ideal")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        all_ts = sorted(set(omp_ts + mpi_ts))
        ax.set_xticks(all_ts); ax.set_xticklabels([str(t) for t in all_ts])
        ax.set_xlabel("workers (threads or ranks)")
        ax.set_ylabel("median wall time (ms)")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    out = REPO / "report" / "figures" / "strong_scaling.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def plot_weak_combined(rows: list[dict]) -> None:
    omp = sorted(filt(rows, impl="openmp", is_weak=True), key=lambda r: r["workers"])
    mpi = sorted(filt(rows, impl="mpi",    is_weak=True), key=lambda r: r["workers"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if omp:
        ax.plot([r["workers"] for r in omp], [r["median_ms"] for r in omp],
                "o-", color=COLOR_OPENMP, lw=2, ms=7,
                label="OpenMP (n=32·T)")
        ax.axhline(omp[0]["median_ms"], ls="--", color=COLOR_OPENMP, alpha=0.4,
                   label="OpenMP ideal (flat)")
    if mpi:
        ax.plot([r["workers"] for r in mpi], [r["median_ms"] for r in mpi],
                "s-", color=COLOR_MPI, lw=2, ms=7,
                label="MPI (n=32·P)")
        ax.axhline(mpi[0]["median_ms"], ls="--", color=COLOR_MPI, alpha=0.4,
                   label="MPI ideal (flat)")
    all_ws = sorted({r["workers"] for r in omp + mpi})
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_ws); ax.set_xticklabels([str(w) for w in all_ws])
    ax.set_xlabel("workers (threads or ranks)")
    ax.set_ylabel("median wall time (ms)")
    ax.set_title("Weak scaling (256×256×n, per-worker work fixed)")
    ax.legend()
    fig.tight_layout()
    out = REPO / "report" / "figures" / "weak_scaling.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def plot_efficiency_bars(rows: list[dict]) -> None:
    """Side-by-side parallel efficiency bar chart at the medium fixture."""
    omp = sorted(filt(rows, impl="openmp", fixture="medium", is_weak=False),
                 key=lambda r: r["workers"])
    mpi = sorted(filt(rows, impl="mpi",    fixture="medium", is_weak=False),
                 key=lambda r: r["workers"])
    if not omp and not mpi:
        return
    omp_t1 = omp[0]["median_ms"] if omp else 1.0
    mpi_t1 = mpi[0]["median_ms"] if mpi else 1.0
    omp_eff = [(omp_t1 / r["median_ms"]) / r["workers"] * 100 for r in omp]
    mpi_eff = [(mpi_t1 / r["median_ms"]) / r["workers"] * 100 for r in mpi]
    omp_w   = [r["workers"] for r in omp]
    mpi_w   = [r["workers"] for r in mpi]

    all_w = sorted(set(omp_w + mpi_w))
    x = np.arange(len(all_w))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if omp_eff:
        omp_y = [omp_eff[omp_w.index(w)] if w in omp_w else 0 for w in all_w]
        ax.bar(x - width/2, omp_y, width, color=COLOR_OPENMP, label="OpenMP")
    if mpi_eff:
        mpi_y = [mpi_eff[mpi_w.index(w)] if w in mpi_w else 0 for w in all_w]
        ax.bar(x + width/2, mpi_y, width, color=COLOR_MPI, label="MPI")
    ax.axhline(100, ls="--", color=COLOR_IDEAL, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([str(w) for w in all_w])
    ax.set_xlabel("workers (threads or ranks)")
    ax.set_ylabel("parallel efficiency (%)")
    ax.set_title("Parallel efficiency at medium fixture (256³)")
    ax.set_ylim(0, 110)
    ax.legend()
    fig.tight_layout()
    out = REPO / "report" / "figures" / "efficiency_bars.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def plot_cross_impl(rows: list[dict]) -> None:
    """Bar chart at the medium fixture, normalized to serial baseline."""
    medium = filt(rows, fixture="medium", is_weak=False)
    serial = filt(medium, impl="serial")
    if not serial:
        print("WARNING: no serial-medium row; skipping cross-impl bar chart")
        return
    serial_ms = serial[0]["median_ms"]

    # Pick the best row (lowest median) per implementation at medium.
    best: dict[str, dict] = {}
    for r in medium:
        cur = best.get(r["impl"])
        if cur is None or r["median_ms"] < cur["median_ms"]:
            best[r["impl"]] = r

    # Order: serial, openmp, mpi, julia, cupy.
    order = ["serial", "openmp", "mpi", "julia", "cupy"]
    labels: list[str] = []
    times:  list[float] = []
    colors: list[str]   = []
    for impl in order:
        if impl in best:
            r = best[impl]
            if impl == "openmp":
                labels.append(f"OpenMP\n(T={r['workers']})")
            elif impl == "mpi":
                labels.append(f"MPI\n(P={r['workers']})")
            else:
                labels.append({
                    "serial": "Serial C++",
                    "julia":  "Julia",
                    "cupy":   "CuPy",
                }[impl])
            times.append(r["median_ms"])
            colors.append({
                "serial": COLOR_SERIAL,
                "openmp": COLOR_OPENMP,
                "mpi":    COLOR_MPI,
                "julia":  COLOR_JULIA,
                "cupy":   COLOR_CUPY,
            }[impl])

    speedups = [serial_ms / t for t in times]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, speedups, color=colors)
    ax.axhline(1.0, ls="--", color=COLOR_IDEAL, lw=1, label="serial baseline (1×)")
    for bar, s, t in zip(bars, speedups, times):
        ax.text(bar.get_x() + bar.get_width()/2, s,
                f"{s:.2f}×\n({t:.0f} ms)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("speedup over serial C++")
    ax.set_title("Cross-implementation comparison at medium (256³)")
    ax.set_ylim(0, max(speedups) * 1.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = REPO / "report" / "figures" / "cross_impl_comparison.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


# ---------------------------------------------------------------------------
def main() -> int:
    if not CSV_PATH.exists():
        print(f"missing {CSV_PATH}; run report/parse_results.py first",
              file=sys.stderr)
        return 1
    rows = load_rows()
    print(f"loaded {len(rows)} rows from {CSV_PATH.relative_to(REPO)}")

    plot_omp_strong(rows)
    plot_omp_weak(rows)
    plot_mpi_strong(rows)
    plot_mpi_weak(rows)
    plot_serial_size_scaling(rows)
    plot_julia_size_scaling(rows)
    plot_strong_combined(rows)
    plot_weak_combined(rows)
    plot_efficiency_bars(rows)
    plot_cross_impl(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
