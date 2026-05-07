"""Parse every */results/*.out into report/results.csv.

The C++ baselines, OpenMP, MPI, CuPy, and Julia drivers all emit one
result line per run with the same shape, e.g.

    m=256 p=256 n=256 threads=16 reps=3 rel_err=3.585e-15 \\
        median_ms=721.279 min_ms=719.061 max_ms=723.471

This script grovels through every .out file under <impl>/results/,
matches the result line, and writes a single tidy CSV.

Run from the repo root:
    conda run -n claude python report/parse_results.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIRS = {
    "serial": REPO / "serial"     / "results",
    "openmp": REPO / "openmp"     / "results",
    "mpi":    REPO / "mpi"        / "results",
    "julia":  REPO / "additional" / "results",
    "cupy":   REPO / "cuda"       / "results",
}

# Match the canonical result line. The worker key varies by impl
# (threads/ranks/backend), so we accept any token followed by =.
LINE_RE = re.compile(
    r"m=(?P<m>\d+)\s+p=(?P<p>\d+)\s+n=(?P<n>\d+)"
    r"(?:\s+(?P<wkind>threads|ranks|backend)=(?P<workers>\S+))?"
    r"\s+reps=(?P<reps>\d+)"
    r"\s+rel_err=(?P<rel_err>\S+)"
    r"\s+median_ms=(?P<median_ms>\S+)"
    r"\s+min_ms=(?P<min_ms>\S+)"
    r"\s+max_ms=(?P<max_ms>\S+)"
)

# Detect which sweep a file belongs to from its name.
WEAK_RE = re.compile(r"_weak[-_]")


def fixture_label(m: int, p: int, n: int, weak: bool) -> str:
    """Human-friendly fixture label so plots can group consistently."""
    if weak:
        return f"weak_n{n}"
    if (m, p, n) == (64, 64, 64):
        return "small"
    if (m, p, n) == (256, 256, 256):
        return "medium"
    if (m, p, n) == (512, 512, 256):
        return "large"
    return f"{m}x{p}x{n}"


def normalize_workers(impl: str, wkind: str | None, workers: str | None) -> tuple[str, int]:
    """Return (worker_kind, worker_count) for the row.

    serial / julia have no parallelism token in the line; treat as 1 worker.
    """
    if wkind is None or workers is None:
        return ("none", 1)
    if wkind == "backend":
        # julia driver writes backend=julia; not a worker count.
        return ("none", 1)
    return (wkind, int(workers))


def parse_file(path: Path, impl: str) -> list[dict]:
    rows: list[dict] = []
    weak = bool(WEAK_RE.search(path.name))
    with path.open() as f:
        for line in f:
            mobj = LINE_RE.search(line)
            if not mobj:
                continue
            d = mobj.groupdict()
            m, p, n = int(d["m"]), int(d["p"]), int(d["n"])
            wkind, workers = normalize_workers(impl, d.get("wkind"), d.get("workers"))
            rows.append({
                "impl":         impl,
                "fixture":      fixture_label(m, p, n, weak),
                "m":            m,
                "p":            p,
                "n":            n,
                "worker_kind":  wkind,
                "workers":      workers,
                "reps":         int(d["reps"]),
                "rel_err":      float(d["rel_err"]),
                "median_ms":    float(d["median_ms"]),
                "min_ms":       float(d["min_ms"]),
                "max_ms":       float(d["max_ms"]),
                "source":       str(path.relative_to(REPO)),
                "is_weak":      weak,
            })
    return rows


def main() -> int:
    out_path = REPO / "report" / "results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for impl, dirpath in RESULTS_DIRS.items():
        if not dirpath.is_dir():
            continue
        for outfile in sorted(dirpath.glob("*.out")):
            file_rows = parse_file(outfile, impl)
            all_rows.extend(file_rows)
            print(f"  parsed {outfile.relative_to(REPO)}: {len(file_rows)} rows",
                  file=sys.stderr)

    if not all_rows:
        print("no data found — did you run any benchmarks?", file=sys.stderr)
        return 1

    fieldnames = [
        "impl", "fixture", "m", "p", "n",
        "worker_kind", "workers", "reps",
        "rel_err", "median_ms", "min_ms", "max_ms",
        "source", "is_weak",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {out_path.relative_to(REPO)} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
