"""Plot VTune v1-vs-v2 hotspots as a side-by-side bar chart for the report.

Reads CSV hotspot reports produced by `openmp/profile.slurm`:
    openmp/results/vtune_hotspots_v1.csv
    openmp/results/vtune_hotspots_v2.csv
and writes:
    report/figures/profile_vtune.pdf

Also prints a tidy comparison table to stdout, plus the v1/v2 elapsed
wall-time pulled from `vtune_summary_v{1,2}.txt` if present.

Run from the repo root:
    conda run -n claude python report/plot_vtune.py

Override paths or the number of top functions:
    python report/plot_vtune.py --top 10 \
        --v1-csv openmp/results/vtune_hotspots_v1.csv \
        --v2-csv openmp/results/vtune_hotspots_v2.csv \
        --out report/figures/profile_vtune.pdf
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "openmp" / "results"

NUM_RE = re.compile(r"^\s*([\d.]+(?:[eE][+-]?\d+)?)\s*(ms|s)?\s*$")


_TXT_ROW_RE = re.compile(r"^(\S(?:.*?\S)?)\s{2,}([\d.]+)s\b")


def parse_csv_hotspots(path: Path) -> dict[str, float]:
    """Return ``{function: cpu_seconds}`` from a VTune hotspots report.

    Reads either the comma-CSV (``-format csv``) or the whitespace-padded
    text output (``-report hotspots`` default; columns separated by 2+
    spaces). Falls back to the .txt sibling if the .csv path is missing
    (the original v1/v2 runs only wrote .txt — CSV emit was added
    afterwards in profile.slurm).
    """
    if not path.exists():
        alt = path.with_suffix(".txt")
        if alt.exists():
            path = alt
        else:
            sys.exit(f"missing: {path} (and no .txt sibling)")

    text = path.read_text()

    if path.suffix == ".csv":
        return _parse_csv_text(text, path)
    return _parse_txt_text(text, path)


def _parse_csv_text(text: str, path: Path) -> dict[str, float]:
    """Comma-CSV path (`vtune -format csv`). Header lookup by column name."""
    reader = csv.reader(text.splitlines())
    header = None
    for row in reader:
        lc = [c.lower().strip() for c in row]
        if any("function" in c for c in lc) and any("time" in c for c in lc):
            header = lc
            break
    if header is None:
        sys.exit(f"no recognizable header in {path}")

    func_idx = next(i for i, c in enumerate(header) if "function" in c)
    self_idx = next(
        (i for i, c in enumerate(header) if "self" in c and "time" in c),
        None,
    )
    time_idx = self_idx if self_idx is not None else next(
        i for i, c in enumerate(header) if "time" in c
    )

    rows: dict[str, float] = {}
    for row in reader:
        if len(row) <= max(func_idx, time_idx):
            continue
        func = row[func_idx].strip().strip('"')
        time_str = row[time_idx].strip().strip('"')
        m = NUM_RE.match(time_str)
        if m and func:
            val = float(m.group(1)) / (1000.0 if m.group(2) == "ms" else 1.0)
            rows.setdefault(func, val)

    if not rows:
        sys.exit(f"parsed 0 rows from {path}")
    return rows


def _parse_txt_text(text: str, path: Path) -> dict[str, float]:
    """Whitespace-padded path. The first column is the function name and
    the second is CPU Time formatted as e.g. ``4.427s``. Skip the header
    and dash separator rows; ignore VTune's trailing crash log."""
    rows: dict[str, float] = {}
    for line in text.splitlines():
        # Stop at VTune's footer banner.
        if line.startswith("Intel(R) VTune"):
            break
        m = _TXT_ROW_RE.match(line)
        if not m:
            continue
        func = m.group(1).strip()
        if func.lower() in ("function", "----", "-" * len(func)):
            continue
        rows.setdefault(func, float(m.group(2)))

    if not rows:
        sys.exit(f"parsed 0 rows from {path}; first 3 lines:\n"
                 + "\n".join(text.splitlines()[:3]))
    return rows


def parse_summary_walltime(path: Path) -> float | None:
    """Pull the elapsed wall-time (seconds) out of a VTune summary text file."""
    if not path.exists():
        return None
    m = re.search(
        r"Elapsed Time:?\s*([\d.]+)\s*s",
        path.read_text(),
        re.IGNORECASE,
    )
    return float(m.group(1)) if m else None


def parse_summary_breakdown(path: Path) -> dict[str, float]:
    """Extract Effective / Spin (imbalance) / Overhead times from the VTune
    summary so we can plot the CPU-time breakdown alongside the hotspots."""
    if not path.exists():
        return {}
    text = path.read_text()
    out: dict[str, float] = {}
    for key, pat in [
        ("effective", r"Effective Time:\s*([\d.]+)\s*s"),
        ("spin",      r"Spin Time:\s*([\d.]+)\s*s"),
        ("overhead",  r"Overhead Time:\s*([\d.]+)\s*s"),
        ("util_pct",  r"Effective Physical Core Utilization:\s*([\d.]+)\s*%"),
    ]:
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def short(name: str, n: int = 32) -> str:
    return name if len(name) <= n else name[: n - 3] + "..."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v1-csv", type=Path,
                    default=RESULTS / "vtune_hotspots_v1.csv")
    ap.add_argument("--v2-csv", type=Path,
                    default=RESULTS / "vtune_hotspots_v2.csv")
    ap.add_argument("--v1-summary", type=Path,
                    default=RESULTS / "vtune_summary_v1.txt")
    ap.add_argument("--v2-summary", type=Path,
                    default=RESULTS / "vtune_summary_v2.txt")
    ap.add_argument("--out", type=Path,
                    default=REPO / "report" / "figures" / "profile_vtune.pdf")
    ap.add_argument("--top", type=int, default=8,
                    help="how many top hotspots to plot (default 8)")
    ap.add_argument("--v1-label", default="pass 1 (SVD pragma only)")
    ap.add_argument("--v2-label", default="pass 2 (helpers parallelized)")
    args = ap.parse_args()

    v1 = parse_csv_hotspots(args.v1_csv)
    v2 = parse_csv_hotspots(args.v2_csv)
    s1 = parse_summary_breakdown(args.v1_summary)
    s2 = parse_summary_breakdown(args.v2_summary)

    # Rank by combined time across both runs so the figure highlights the
    # functions where the change matters most.
    funcs = sorted(set(v1) | set(v2),
                   key=lambda f: -(v1.get(f, 0.0) + v2.get(f, 0.0)))[: args.top]

    t1 = [v1.get(f, 0.0) for f in funcs]
    t2 = [v2.get(f, 0.0) for f in funcs]
    labels = [short(f) for f in funcs]

    fig, (ax_h, ax_b) = plt.subplots(
        1, 2, figsize=(12, max(3.5, 0.55 * len(funcs) + 1.0)),
        gridspec_kw={"width_ratios": [2.4, 1.0]},
    )

    # Left: hotspot bars.
    y = list(range(len(funcs)))
    ax_h.barh([i - 0.2 for i in y], t1, height=0.4, label=args.v1_label,
              color="#4C72B0")
    ax_h.barh([i + 0.2 for i in y], t2, height=0.4, label=args.v2_label,
              color="#C44E52")
    ax_h.set_yticks(y)
    ax_h.set_yticklabels(labels)
    ax_h.invert_yaxis()
    ax_h.set_xlabel("CPU time (s)")
    ax_h.set_title(f"Top-{len(funcs)} hotspots")
    ax_h.grid(axis="x", alpha=0.3)
    ax_h.legend(loc="lower right")

    # Right: stacked CPU-time breakdown (effective + spin + overhead).
    cats = ["effective", "spin", "overhead"]
    cat_labels = ["effective", "spin (imbalance)", "overhead"]
    colors = ["#55A868", "#C44E52", "#8172B2"]
    runs = ["pass 1", "pass 2"]
    summaries = [s1, s2]

    bottom = [0.0, 0.0]
    for cat, lbl, col in zip(cats, cat_labels, colors):
        vals = [s.get(cat, 0.0) for s in summaries]
        ax_b.bar(runs, vals, bottom=bottom, color=col, label=lbl)
        for i, v in enumerate(vals):
            if v > 0.05:  # annotate sizable segments
                ax_b.text(i, bottom[i] + v / 2, f"{v:.2f}s",
                          ha="center", va="center", fontsize=8, color="white")
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax_b.set_ylabel("CPU time (s)")
    ax_b.set_title("CPU-time breakdown")
    ax_b.legend(loc="upper right", fontsize=8)
    ax_b.grid(axis="y", alpha=0.3)

    fig.suptitle("VTune — OpenMP, T=16, medium fixture", y=1.0)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")

    # Tidy comparison table.
    print()
    print(f"{'function':<32}  {'v1 (s)':>10}  {'v2 (s)':>10}  {'delta':>10}")
    print("-" * 68)
    for f, a, b in zip(funcs, t1, t2):
        print(f"{short(f):<32}  {a:>10.3f}  {b:>10.3f}  {b - a:>+10.3f}")

    w1 = parse_summary_walltime(args.v1_summary)
    w2 = parse_summary_walltime(args.v2_summary)
    if w1 is not None and w2 is not None:
        delta_pct = (w1 - w2) / w1 * 100.0
        print(f"\nelapsed wall time: v1 = {w1:.3f} s, v2 = {w2:.3f} s "
              f"({delta_pct:+.1f}% faster)")
    elif w1 is not None or w2 is not None:
        print("\n(only one summary file found; skipping wall-time comparison)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
