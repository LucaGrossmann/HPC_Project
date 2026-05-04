// File:        tsvdm_mpi.cpp
// Description: MPI distributed-memory implementation of t-SVDM
//              (Kilmer, Horesh, Avron, Newman, 2021, Algorithm 2).
// Copyright 2026 Harvard University.
//
// Strategy
// --------
// Diff from serial: replicate the cheap mode-3 transforms on every rank,
// partition the (expensive) per-slice dgesdd loop across ranks, and
// MPI_Allgatherv the per-factor outputs so every rank ends with the same
// U, S, V. Only rank 0 reads the fixture, dumps, and reports.
//
// Communication pattern
// ---------------------
//   1× MPI_Bcast       — distribute A (m·p·n doubles) and M (n² doubles)
//   3× MPI_Allgatherv  — assemble Uhat, sHat, VtHat from per-rank slabs
//   1× MPI_Reduce      — collect the slowest rank's wall time
//
// No derived datatypes. No MPI_Alltoallv. No nested parallelism.
//
// Layout convention: identical to serial/tsvdm_serial.cpp — see that file
// for details on the (m, p, n) column-major data layout.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <mpi.h>

extern "C" {
    void dgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const double* alpha,
                const double* A, const int* lda,
                const double* B, const int* ldb,
                const double* beta,
                double* C, const int* ldc);

    void dgesdd_(const char* jobz,
                 const int* m, const int* n,
                 double* A, const int* lda,
                 double* S,
                 double* U, const int* ldu,
                 double* VT, const int* ldvt,
                 double* work, const int* lwork,
                 int* iwork, int* info);
}

static const char MAGIC[5] = "TSVD";

struct Fixture {
    int m = 0;
    int p = 0;
    int n = 0;
    std::vector<double> A;
    std::vector<double> M;
};

/** @brief Read a fixture (rank 0 only). */
static Fixture loadFixture(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char magic[4];
    std::fread(magic, sizeof(char), 4, f);
    if (std::memcmp(magic, MAGIC, 4) != 0) {
        std::fprintf(stderr, "error: bad magic in %s (expected TSVD)\n",
                     path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int32_t m = 0, p = 0, n = 0;
    std::fread(&m, sizeof(int32_t), 1, f);
    std::fread(&p, sizeof(int32_t), 1, f);
    std::fread(&n, sizeof(int32_t), 1, f);

    Fixture fx;
    fx.m = m;
    fx.p = p;
    fx.n = n;
    fx.A.resize(static_cast<size_t>(m) * p * n);
    fx.M.resize(static_cast<size_t>(n) * n);

    std::fread(fx.A.data(), sizeof(double), fx.A.size(), f);
    std::fread(fx.M.data(), sizeof(double), fx.M.size(), f);

    std::fclose(f);
    return fx;
}

/** @brief Compute this rank's slice range [k0, k0 + count).
 *
 *  The first (n % size) ranks each get one extra slice — handles arbitrary
 *  (n, size) pairs without requiring size | n.
 */
static void blockRange(int n, int size, int rank, int* k0, int* count) {
    const int base = n / size;
    const int rem  = n % size;
    *count = base + (rank < rem ? 1 : 0);
    *k0    = rank * base + std::min(rank, rem);
}

/** @brief Mode-3 product, single dgemm. Identical to serial. */
static void modeThree(const double* B, const double* M,
                      double* Bhat, int m, int p, int n, bool forward) {
    const int mp = m * p;
    const char noTrans = 'N';
    const char trans   = 'T';
    const double alpha = 1.0;
    const double beta  = 0.0;
    const char* transb;
    if (forward) {
        transb = &trans;    // Bhat = B . M^T
    } else {
        transb = &noTrans;  // Bhat = B . M
    }
    
    dgemm_(&noTrans, transb,
           &mp, &n, &n,
           &alpha, B, &mp,
           M, &n,
           &beta, Bhat, &mp);
}

/** @brief Run dgesdd on slices [k0, k0+count) only.
 *
 *  Writes into the per-rank section of the (global-shaped) output buffers,
 *  i.e. Uhat at offset (k0 * m * r), etc. The remaining slabs are filled
 *  by MPI_Allgatherv after this call.
 */
static void sliceSvdRange(double* Ahat,
                          double* Uhat, double* sHat, double* VtHat,
                          int m, int p, int k0, int count) {
    const int r = std::min(m, p);
    const char jobz = 'S';

    // Workspace query (same on every rank, so they all do it independently).
    int lwork = -1;
    double workQuery = 0.0;
    int info = 0;
    {
        std::vector<int> iwork_q(8 * r);
        dgesdd_(&jobz, &m, &p,
                Ahat, &m,
                sHat,
                Uhat, &m,
                VtHat, &r,
                &workQuery, &lwork,
                iwork_q.data(), &info);
        if (info != 0) {
            std::fprintf(stderr, "dgesdd workspace query failed, info=%d\n", info);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    lwork = static_cast<int>(workQuery);
    std::vector<double> work(lwork);
    std::vector<int> iwork(8 * r);

    for (int k = k0; k < k0 + count; ++k) {
        double* aPtr = Ahat  + static_cast<size_t>(k) * m * p;
        double* uPtr = Uhat  + static_cast<size_t>(k) * m * r;
        double* sPtr = sHat  + static_cast<size_t>(k) * r;
        double* vPtr = VtHat + static_cast<size_t>(k) * r * p;
        dgesdd_(&jobz, &m, &p,
                aPtr, &m,
                sPtr,
                uPtr, &m,
                vPtr, &r,
                work.data(), &lwork,
                iwork.data(), &info);
        if (info != 0) {
            std::fprintf(stderr, "dgesdd slice %d failed, info=%d\n", k, info);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

/** @brief Build a dense f-diagonal Shat (r, r, n) from packed sHat (r, n). */
static void assembleSDense(const double* sHat, double* Shat, int n, int r) {
    std::fill(Shat, Shat + static_cast<size_t>(r) * r * n, 0.0);
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < r; ++i) {
            Shat[static_cast<size_t>(i) + static_cast<size_t>(i) * r
                 + static_cast<size_t>(k) * r * r] = sHat[i + k * r];
        }
    }
}

/** @brief Distributed t-SVDM. Every rank exits with the same U, S, V. */
static void tsvdmMpi(const double* A, const double* M,
                    int m, int p, int n,
                    double* U, double* S, double* V,
                    MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const int r = std::min(m, p);

    // 1. Forward mode-3 — redundant on every rank (one dgemm).
    std::vector<double> Ahat(static_cast<size_t>(m) * p * n);
    modeThree(A, M, Ahat.data(), m, p, n, /*forward=*/true);

    // 2. Per-rank slice range.
    int k0 = 0, count = 0;
    blockRange(n, size, rank, &k0, &count);

    // 3. Local SVDs on owned slices, written into the global-shaped buffers.
    std::vector<double> Uhat (static_cast<size_t>(m) * r * n);
    std::vector<double> sHat (static_cast<size_t>(r) * n);
    std::vector<double> VtHat(static_cast<size_t>(r) * p * n);
    sliceSvdRange(Ahat.data(), Uhat.data(), sHat.data(), VtHat.data(),
                  m, p, k0, count);

    // 4. Allgatherv per factor — each rank fills displs[r]..displs[r]+counts[r].
    std::vector<int> recvU(size), dispU(size);
    std::vector<int> recvS(size), dispS(size);
    std::vector<int> recvV(size), dispV(size);
    for (int rk = 0; rk < size; ++rk) {
        int k0_r, cnt_r;
        blockRange(n, size, rk, &k0_r, &cnt_r);
        recvU[rk] = cnt_r * m * r;   dispU[rk] = k0_r * m * r;
        recvS[rk] = cnt_r * r;       dispS[rk] = k0_r * r;
        recvV[rk] = cnt_r * r * p;   dispV[rk] = k0_r * r * p;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   Uhat.data(), recvU.data(), dispU.data(),
                   MPI_DOUBLE, comm);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   sHat.data(), recvS.data(), dispS.data(),
                   MPI_DOUBLE, comm);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   VtHat.data(), recvV.data(), dispV.data(),
                   MPI_DOUBLE, comm);

    // 5. Densify S and transpose V — redundant on every rank.
    std::vector<double> Shat(static_cast<size_t>(r) * r * n);
    assembleSDense(sHat.data(), Shat.data(), n, r);

    std::vector<double> Vhat(static_cast<size_t>(p) * r * n);
    for (int k = 0; k < n; ++k) {
        const double* src = VtHat.data() + static_cast<size_t>(k) * r * p;
        double* dst       = Vhat.data()  + static_cast<size_t>(k) * p * r;
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < p; ++j) {
                dst[j + i * p] = src[i + j * r];
            }
        }
    }

    // 6. Inverse mode-3 — redundant on every rank.
    modeThree(Uhat.data(), M, U, m, r, n, /*forward=*/false);
    modeThree(Shat.data(), M, S, r, r, n, /*forward=*/false);
    modeThree(Vhat.data(), M, V, p, r, n, /*forward=*/false);
}

/** @brief Reconstruct A_approx from (U, S, V, M). Used only for verification. */
static void reconstructApprox(const double* U, const double* S, const double* V,
                              const double* M,
                              int m, int p, int n, int r,
                              double* Aapprox) {
    std::vector<double> Uhat (static_cast<size_t>(m) * r * n);
    std::vector<double> Shat (static_cast<size_t>(r) * r * n);
    std::vector<double> Vhat (static_cast<size_t>(p) * r * n);
    modeThree(U, M, Uhat.data(), m, r, n, /*forward=*/true);
    modeThree(S, M, Shat.data(), r, r, n, /*forward=*/true);
    modeThree(V, M, Vhat.data(), p, r, n, /*forward=*/true);

    std::vector<double> AhatApprox(static_cast<size_t>(m) * p * n);
    std::vector<double> tmp(static_cast<size_t>(m) * r);
    const char noTrans = 'N';
    const char trans   = 'T';
    const double one  = 1.0;
    const double zero = 0.0;
    for (int k = 0; k < n; ++k) {
        const double* uPtr = Uhat.data() + static_cast<size_t>(k) * m * r;
        const double* sPtr = Shat.data() + static_cast<size_t>(k) * r * r;
        const double* vPtr = Vhat.data() + static_cast<size_t>(k) * p * r;
        double*       aPtr = AhatApprox.data() + static_cast<size_t>(k) * m * p;
        dgemm_(&noTrans, &noTrans, &m, &r, &r,
               &one, uPtr, &m, sPtr, &r,
               &zero, tmp.data(), &m);
        dgemm_(&noTrans, &trans, &m, &p, &r,
               &one, tmp.data(), &m, vPtr, &p,
               &zero, aPtr, &m);
    }
    modeThree(AhatApprox.data(), M, Aapprox, m, p, n, /*forward=*/false);
}

static double relativeError(const double* A, const double* Aapprox, size_t n) {
    double numer = 0.0;
    double denom = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double diff = A[i] - Aapprox[i];
        numer += diff * diff;
        denom += A[i] * A[i];
    }
    return std::sqrt(numer) / std::sqrt(denom);
}

// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char* fixturePath = nullptr;
    const char* dumpPath    = nullptr;
    int reps = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dumpPath = argv[++i];
        } else if (argv[i][0] != '-') {
            fixturePath = argv[i];
        } else {
            if (rank == 0) std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            MPI_Finalize();
            return 2;
        }
    }
    if (!fixturePath || reps < 1) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "usage: %s <fixture.bin> [--reps N] [--dump out.bin]\n",
                         argv[0]);
        }
        MPI_Finalize();
        return 2;
    }

    // Rank 0 reads the fixture; broadcast dimensions then payload.
    Fixture fx;
    int dims[3] = {0, 0, 0};
    if (rank == 0) {
        fx = loadFixture(fixturePath);
        dims[0] = fx.m;
        dims[1] = fx.p;
        dims[2] = fx.n;
    }
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    const int m = dims[0];
    const int p = dims[1];
    const int n = dims[2];
    const int r = std::min(m, p);

    if (rank != 0) {
        fx.m = m; fx.p = p; fx.n = n;
        fx.A.resize(static_cast<size_t>(m) * p * n);
        fx.M.resize(static_cast<size_t>(n) * n);
    }
    MPI_Bcast(fx.A.data(), static_cast<int>(fx.A.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(fx.M.data(), static_cast<int>(fx.M.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> U(static_cast<size_t>(m) * r * n);
    std::vector<double> S(static_cast<size_t>(r) * r * n);
    std::vector<double> V(static_cast<size_t>(p) * r * n);

    // Warm up — not counted.
    tsvdmMpi(fx.A.data(), fx.M.data(), m, p, n,
             U.data(), S.data(), V.data(), MPI_COMM_WORLD);

    std::vector<double> times(reps);
    for (int rep = 0; rep < reps; ++rep) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();
        tsvdmMpi(fx.A.data(), fx.M.data(), m, p, n,
                 U.data(), S.data(), V.data(), MPI_COMM_WORLD);
        const double localMs = (MPI_Wtime() - t0) * 1000.0;

        double maxMs = 0.0;
        MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) times[rep] = maxMs;
    }

    // Verification + reporting on rank 0 only (every rank holds identical
    // U, S, V after the allgatherv, so this is non-redundant).
    if (rank == 0) {
        std::sort(times.begin(), times.end());
        const double minMs    = times.front();
        const double maxMs    = times.back();
        const double medianMs = times[reps / 2];

        std::vector<double> Aapprox(static_cast<size_t>(m) * p * n);
        reconstructApprox(U.data(), S.data(), V.data(), fx.M.data(),
                          m, p, n, r, Aapprox.data());
        const double relErr = relativeError(fx.A.data(), Aapprox.data(),
                                            fx.A.size());

        std::printf("m=%d p=%d n=%d ranks=%d reps=%d rel_err=%.3e "
                    "median_ms=%.3f min_ms=%.3f max_ms=%.3f\n",
                    m, p, n, size, reps, relErr, medianMs, minMs, maxMs);

        if (dumpPath) {
            std::FILE* out = std::fopen(dumpPath, "wb");
            if (!out) {
                std::fprintf(stderr, "error: cannot open %s for writing\n",
                             dumpPath);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::fwrite(Aapprox.data(), sizeof(double), Aapprox.size(), out);
            std::fclose(out);
        }

        const int rc = (relErr < 1e-10) ? 0 : 1;
        MPI_Finalize();
        return rc;
    }

    MPI_Finalize();
    return 0;
}
