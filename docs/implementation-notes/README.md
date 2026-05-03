# Implementation Notes

Per-implementation design notes for the HPC study of t-SVDM / t-SVDMII.
Each file here explains **how** and **why** one implementation is
structured the way it is, and how to reproduce its results. The
course-level plan lives in `../plan.md`; the formal algorithms live in
`../kilmer-et-al-2021-tensor-tensor-algebra...pdf`.

## Index

| Part | Implementation | Notes | Status |
|---|---|---|---|
| 0 | Python reference (NumPy) | [`00-python-reference.md`](./00-python-reference.md) | ✅ Complete |
| 1 | Serial C++ | [`01-serial-cpp.md`](./01-serial-cpp.md) | ✅ Complete |
| 2 | OpenMP C++ | _not started_ | — |
| 3 | MPI C++ | _not started_ | — |
| 4 | CUDA via CuPy | _not started_ | — |
| 5 | Julia | _not started_ | — |

Each file should cover: purpose, module layout, key design decisions
(with the alternative that was rejected), test strategy, known
limitations, and handoff expectations for downstream parts.
