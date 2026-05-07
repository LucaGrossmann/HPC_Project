// File:        tsvdm_serial.cpp
// Description: Serial C++ baseline implementation of t-SVDM
//              (Kilmer, Horesh, Avron, Newman, 2021, Algorithm 2).
//              Parts 2-3 (OpenMP, MPI) start from this file.
// Copyright 2026 Harvard University.
//
// Layout convention
// -----------------
// A has shape (m, p, n), column-major (Fortran order):
//     A(i, j, k) == A[i + j*m + k*m*p]      0 <= i < m, 0 <= j < p, 0 <= k < n
// Frontal slice k is the contiguous m*p-element block starting at A + k*m*p.
// M is (n, n) column-major and orthogonal.
//
// This is the *opposite* of Part 0's Python layout ((n, m, p), C-order).
// The fixture generator (cuda/gen_fixture.py) does the conversion at write
// time so this code can fread slices directly into LAPACK.
//
// Build: see Makefile.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// LAPACK / BLAS prototypes (Fortran-style; OpenBLAS via Spack)
// -----------------------------------------------------------------------------
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
    std::vector<double> A;   // size m*p*n, column-major (m, p, n)
    std::vector<double> M;   // size n*n,   column-major (n, n)
};

/** @brief Read a fixture written by cuda/gen_fixture.py. */
static Fixture loadFixture(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
        std::exit(1);
    }

    char magic[4];
    std::fread(magic, sizeof(char), 4, f);
    if (std::memcmp(magic, MAGIC, 4) != 0) {
        std::fprintf(stderr, "error: bad magic in %s (expected TSVD)\n",
                     path.c_str());
        std::exit(1);
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

/** @brief Mode-3 product: Bhat = B x_3 M if forward, else B x_3 M^T.
 *
 *  View B as an (m*p) x n column-major matrix. Then the mode-3 product
 *  along the tube axis is a single dense matmul with M (or M^T).
 */
static void modeThree(const double* B, const double* M,
                      double* Bhat, int m, int p, int n, bool forward) {
    const int mp = m * p;
    const char noTrans = 'N';
    const char trans   = 'T';
    const double alpha = 1.0;
    const double beta  = 0.0;
    // Forward:  Bhat = B . M^T    (because we want Bhat(r, k) = sum_l B(r, l) M(k, l))
    // Inverse:  Bhat = B . M      (applies M^T to tube fibers, the inverse of forward)
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

/** @brief Per-slice thin SVD via LAPACK dgesdd.
 *
 *  Destroys Ahat (as dgesdd requires). Writes:
 *      Uhat   : (m, r, n) column-major, r = min(m, p)
 *      sHat   : (r, n)    column-major (one column per slice)
 *      VtHat  : (r, p, n) column-major (each slice is V^T, r x p)
 *  iwork size is 8 * min(m, p) per the LAPACK reference.
 */
static void sliceSvdAll(double* Ahat,
                        double* Uhat, double* sHat, double* VtHat,
                        int m, int p, int n) {
    const int r = std::min(m, p);
    const char jobz = 'S';   // thin SVD

    // Workspace query: lwork = -1 returns the optimal size in work[0].
    int lwork = -1;
    double workQuery = 0.0;
    int info = 0;
    std::vector<int> iwork(8 * r);
    dgesdd_(&jobz, &m, &p,
            Ahat, &m,
            sHat,
            Uhat, &m,
            VtHat, &r,
            &workQuery, &lwork,
            iwork.data(), &info);
    if (info != 0) {
        std::fprintf(stderr, "dgesdd workspace query failed, info=%d\n", info);
        std::exit(1);
    }
    lwork = static_cast<int>(workQuery);
    std::vector<double> work(lwork);

    for (int k = 0; k < n; ++k) {
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
            std::exit(1);
        }
    }
}

/** @brief Build a dense f-diagonal tensor Shat (r, r, n) from packed sHat (r, n). */
static void assembleSDense(const double* sHat, double* Shat, int n, int r) {
    std::fill(Shat, Shat + static_cast<size_t>(r) * r * n, 0.0);
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < r; ++i) {
            // Shat(i, i, k) = s(i, k)
            Shat[static_cast<size_t>(i) + static_cast<size_t>(i) * r
                 + static_cast<size_t>(k) * r * r] = sHat[i + k * r];
        }
    }
}

/** @brief t-SVDM (Algorithm 2). Writes U, S, V in the spatial domain.
 *
 *  U is (m, r, n), S is (r, r, n) f-diagonal, V is (p, r, n), r = min(m, p).
 *  A is not modified.
 */
static void tsvdm(const double* A, const double* M,
                  int m, int p, int n,
                  double* U, double* S, double* V) {
    const int r = std::min(m, p);

    std::vector<double> Ahat(static_cast<size_t>(m) * p * n);
    modeThree(A, M, Ahat.data(), m, p, n, /*forward=*/true);

    // Transform-domain factors.
    std::vector<double> Uhat (static_cast<size_t>(m) * r * n);
    std::vector<double> sHat (static_cast<size_t>(r) * n);
    std::vector<double> VtHat(static_cast<size_t>(r) * p * n);
    sliceSvdAll(Ahat.data(), Uhat.data(), sHat.data(), VtHat.data(), m, p, n);

    std::vector<double> Shat(static_cast<size_t>(r) * r * n);
    assembleSDense(sHat.data(), Shat.data(), n, r);

    // V wants shape (p, r, n), not (r, p, n). Transpose each slice.
    std::vector<double> Vhat(static_cast<size_t>(p) * r * n);
    for (int k = 0; k < n; ++k) {
        const double* src = VtHat.data() + static_cast<size_t>(k) * r * p;
        double* dst       = Vhat.data()  + static_cast<size_t>(k) * p * r;
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < p; ++j) {
                // VtHat(i, j, k) -> Vhat(j, i, k)
                dst[j + i * p] = src[i + j * r];
            }
        }
    }

    // Inverse mode-3 (apply M^T) on each factor to land in spatial domain.
    modeThree(Uhat.data(), M, U, m, r, n, /*forward=*/false);
    modeThree(Shat.data(), M, S, r, r, n, /*forward=*/false);
    modeThree(Vhat.data(), M, V, p, r, n, /*forward=*/false);
}

/** @brief Reconstruct A_approx from (U, S, V, M). */
static void reconstructApprox(const double* U, const double* S, const double* V,
                              const double* M,
                              int m, int p, int n, int r,
                              double* Aapprox) {
    // Lift U, S, V back to the transform domain.
    std::vector<double> Uhat (static_cast<size_t>(m) * r * n);
    std::vector<double> Shat (static_cast<size_t>(r) * r * n);
    std::vector<double> Vhat (static_cast<size_t>(p) * r * n);
    modeThree(U, M, Uhat.data(), m, r, n, /*forward=*/true);
    modeThree(S, M, Shat.data(), r, r, n, /*forward=*/true);
    modeThree(V, M, Vhat.data(), p, r, n, /*forward=*/true);

    // Ahat_approx[k] = Uhat[k] . Shat[k] . Vhat[k]^T  (facewise matmul)
    std::vector<double> AhatApprox(static_cast<size_t>(m) * p * n);
    std::vector<double> tmp(static_cast<size_t>(m) * r);  // reused per slice
    const char noTrans = 'N';
    const char trans   = 'T';
    const double one  = 1.0;
    const double zero = 0.0;
    for (int k = 0; k < n; ++k) {
        const double* uPtr = Uhat.data() + static_cast<size_t>(k) * m * r;
        const double* sPtr = Shat.data() + static_cast<size_t>(k) * r * r;
        const double* vPtr = Vhat.data() + static_cast<size_t>(k) * p * r;
        double*       aPtr = AhatApprox.data() + static_cast<size_t>(k) * m * p;

        // tmp = Uhat . Shat   (m x r) = (m x r) . (r x r)
        dgemm_(&noTrans, &noTrans, &m, &r, &r,
               &one, uPtr, &m, sPtr, &r,
               &zero, tmp.data(), &m);
        // Ahat_approx = tmp . Vhat^T   (m x p) = (m x r) . (r x p)
        dgemm_(&noTrans, &trans, &m, &p, &r,
               &one, tmp.data(), &m, vPtr, &p,
               &zero, aPtr, &m);
    }

    // Inverse mode-3 to return to spatial domain.
    modeThree(AhatApprox.data(), M, Aapprox, m, p, n, /*forward=*/false);
}

/** @brief Relative Frobenius error ||A - Aapprox||_F / ||A||_F. */
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
    const int reps = 30;
    const char* fixturePath = nullptr;
    const char* dumpPath    = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dumpPath = argv[++i];
        } else if (argv[i][0] != '-') {
            fixturePath = argv[i];
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!fixturePath) {
        std::fprintf(stderr, "usage: %s <fixture.bin> [--dump out.bin]\n",
                     argv[0]);
        return 2;
    }

    Fixture fx = loadFixture(fixturePath);
    const int m = fx.m;
    const int p = fx.p;
    const int n = fx.n;
    const int r = std::min(m, p);

    std::vector<double> U(static_cast<size_t>(m) * r * n);
    std::vector<double> S(static_cast<size_t>(r) * r * n);
    std::vector<double> V(static_cast<size_t>(p) * r * n);

    // Warm up once (pays for any lazy BLAS init) — not counted.
    tsvdm(fx.A.data(), fx.M.data(), m, p, n, U.data(), S.data(), V.data());

    // Time N reps of the decomposition itself, per plan.md §4 Part 1.
    std::vector<double> times(reps);
    for (int rep = 0; rep < reps; ++rep) {
        auto tStart = std::chrono::steady_clock::now();
        tsvdm(fx.A.data(), fx.M.data(), m, p, n, U.data(), S.data(), V.data());
        auto tEnd = std::chrono::steady_clock::now();
        times[rep] = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    }
    std::sort(times.begin(), times.end());
    const double minMs    = times.front();
    const double maxMs    = times.back();
    const double medianMs = times[reps / 2];

    // Verify via reconstruction (not timed).
    std::vector<double> Aapprox(static_cast<size_t>(m) * p * n);
    reconstructApprox(U.data(), S.data(), V.data(), fx.M.data(),
                      m, p, n, r, Aapprox.data());
    const double relErr = relativeError(fx.A.data(), Aapprox.data(), fx.A.size());

    std::printf("m=%d p=%d n=%d reps=%d rel_err=%.3e "
                "median_ms=%.3f min_ms=%.3f max_ms=%.3f\n",
                m, p, n, reps, relErr, medianMs, minMs, maxMs);

    if (dumpPath) {
        std::FILE* out = std::fopen(dumpPath, "wb");
        if (!out) {
            std::fprintf(stderr, "error: cannot open %s for writing\n", dumpPath);
            return 1;
        }
        std::fwrite(Aapprox.data(), sizeof(double), Aapprox.size(), out);
        std::fclose(out);
    }

    return (relErr < 1e-10) ? 0 : 1;
}
