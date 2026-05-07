#!/usr/bin/env julia
# Part 5 — t-SVDM in Julia (stdlib only).
#
# Layout (matches cuda/run_cupy.py):
#   A : (n, m, p)         tensor
#   M : (n, n)            orthogonal matrix
# Factors returned by tsvdm:
#   U : (n, m, r)
#   S : (n, r, r)         f-diagonal
#   V : (n, p, r)
# where r = min(m, p).
#
# Run:
#   julia tsvdm.jl <fixture.bin> [--reps N] [--dump out.bin]
#   julia tsvdm.jl --gen M P N   [--reps N] [--dump out.bin]

using LinearAlgebra
using Random
using Printf

# Mode-3 product: (X ×₃ A)[i,j,k] = Σₗ X[i,l] · A[l,j,k].
# Reshape A as (n, m*p), multiply by X, reshape back.
function mode3(X, A)
    n, m, p = size(A)
    return reshape(X * reshape(A, n, m*p), n, m, p)
end

function tsvdm(A, M)
    n, m, p = size(A)
    r = min(m, p)

    Ahat = mode3(M, A)

    Uhat = zeros(n, m, r)
    Shat = zeros(n, r, r)
    Vhat = zeros(n, p, r)
    @views for i in 1:n                       # @views: slices stay as views, no copy
        F = svd(Ahat[i, :, :])                # F.U (m,r), F.S (r,), F.V (p,r)
        Uhat[i, :, :] = F.U
        Shat[i, :, :] = Diagonal(F.S)         # off-diagonals stay zero
        Vhat[i, :, :] = F.V
    end

    return mode3(M', Uhat), mode3(M', Shat), mode3(M', Vhat)
end

function reconstruct(U, S, V, M)
    Uhat = mode3(M, U)
    Shat = mode3(M, S)
    Vhat = mode3(M, V)
    n, m, _ = size(Uhat)
    p = size(Vhat, 2)
    Ahat = zeros(n, m, p)
    @views for i in 1:n
        Ahat[i, :, :] = Uhat[i, :, :] * Shat[i, :, :] * Vhat[i, :, :]'
    end
    return mode3(M', Ahat)
end

function load_fixture(path)
    open(path, "r") do f
        @assert String(read(f, 4)) == "TSVD" "bad fixture magic"
        m = Int(read(f, Int32))
        p = Int(read(f, Int32))
        n = Int(read(f, Int32))
        # Disk format: column-major (m, p, n). Permute into our (n, m, p) layout.
        A_disk = Array{Float64,3}(undef, m, p, n)
        read!(f, A_disk)
        A = permutedims(A_disk, (3, 1, 2))
        M = Matrix{Float64}(undef, n, n)
        read!(f, M)
        return A, M, (m, p, n)
    end
end

function generate(m, p, n; seed=0)
    Random.seed!(seed)
    A = randn(n, m, p)
    Q = Matrix(qr(randn(n, n)).Q)             # materialize Q out of the QR-packed form
    return A, Q, (m, p, n)
end

function main(args)
    reps_idx = findfirst(==("--reps"), args)
    reps = reps_idx === nothing ? 1 : parse(Int, args[reps_idx + 1])

    dump_idx = findfirst(==("--dump"), args)
    dump = dump_idx === nothing ? nothing : args[dump_idx + 1]

    A, M, (m, p, n) =
        if !isempty(args) && args[1] == "--gen"
            generate(parse(Int, args[2]), parse(Int, args[3]), parse(Int, args[4]))
        elseif !isempty(args) && !startswith(args[1], "-")
            load_fixture(args[1])
        else
            error("usage: tsvdm.jl <fixture.bin|--gen M P N> [--reps N] [--dump out.bin]")
        end

    tsvdm(A, M)                               # warmup — pays Julia's first-call JIT cost

    times = Float64[]
    for _ in 1:reps
        t0 = time()
        tsvdm(A, M)
        push!(times, (time() - t0) * 1000.0)
    end
    sort!(times)

    U, S, V = tsvdm(A, M)
    A_approx = reconstruct(U, S, V, M)
    rel_err = norm(A_approx .- A) / norm(A)

    @printf("m=%d p=%d n=%d backend=julia reps=%d rel_err=%.3e median_ms=%.3f min_ms=%.3f max_ms=%.3f\n",
            m, p, n, reps, rel_err, times[(length(times) ÷ 2) + 1],
            minimum(times), maximum(times))

    if dump !== nothing
        # Match C++ dump format: column-major (m, p, n).
        A_dump = permutedims(A_approx, (2, 3, 1))
        open(dump, "w") do f
            write(f, A_dump)
        end
    end

    return rel_err < 1e-10 ? 0 : 1
end

exit(main(ARGS))
