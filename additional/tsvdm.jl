#!/usr/bin/env julia
# Part 5 — additional implementation in Julia.
#
# Single-file algorithm + driver. Uses only Julia stdlib (LinearAlgebra,
# Random, Printf). Layout convention matches cuda/run_cupy.py:
#   A : (n, m, p)         tensor, slices stacked along axis 1
#   M : (n, n)            orthogonal matrix
# Returned factors:
#   U : (n, m, r)
#   S : (n, r, r)         f-diagonal
#   V : (n, p, r)
# where r = min(m, p).
#
# Run locally:
#   julia tsvdm.jl ../serial/fixtures/small.bin
#   julia tsvdm.jl --gen 256 256 256 --reps 3
#
# Run on cluster: submit.slurm (CS-2050 Spack env).

using LinearAlgebra
using Random
using Printf

# --- Algorithm (the "Julia part") --------------------------------------------

function tsvdm(A::Array{Float64,3}, M::Matrix{Float64})
    n, m, p = size(A)
    r = min(m, p)

    # Forward mode-3 transform: A_hat[i,j,k] = sum_l M[i,l] * A[l,j,k].
    # Trick: reshape A as (n, m*p), multiply by M, reshape back.
    Ahat = reshape(M * reshape(A, n, m*p), n, m, p)

    # Per-slice thin SVD (stdlib has no batched variant).
    Uhat  = Array{Float64,3}(undef, n, m, r)
    shat  = Array{Float64,2}(undef, n, r)
    Vthat = Array{Float64,3}(undef, n, r, p)
    for i in 1:n
        F = svd(Ahat[i, :, :])              # F.U (m,r), F.S (r,), F.Vt (r,p)
        Uhat[i, :, :]  .= F.U
        shat[i, :]     .= F.S
        Vthat[i, :, :] .= F.Vt
    end

    # Pack singular values into a dense f-diagonal tensor Shat (n, r, r).
    Shat = zeros(Float64, n, r, r)
    for i in 1:n, j in 1:r
        Shat[i, j, j] = shat[i, j]
    end

    Vhat = permutedims(Vthat, (1, 3, 2))    # (n, r, p) -> (n, p, r)

    # Inverse mode-3 transform via M^T (M is orthogonal).
    Mt = transpose(M)
    invmode3(B) = (let s = size(B); reshape(Mt * reshape(B, s[1], s[2]*s[3]), s) end)
    return invmode3(Uhat), invmode3(Shat), invmode3(Vhat)
end

function reconstruct(U::Array{Float64,3}, S::Array{Float64,3}, V::Array{Float64,3},
                     M::Matrix{Float64})
    fwd(B) = (let s = size(B); reshape(M * reshape(B, s[1], s[2]*s[3]), s) end)
    Uhat, Shat, Vhat = fwd(U), fwd(S), fwd(V)
    n, m, _ = size(Uhat)
    p = size(Vhat, 2)
    Ahat = Array{Float64,3}(undef, n, m, p)
    for i in 1:n
        Ahat[i, :, :] .= Uhat[i, :, :] * Shat[i, :, :] * transpose(Vhat[i, :, :])
    end
    Mt = transpose(M)
    return reshape(Mt * reshape(Ahat, n, m*p), n, m, p)
end

# --- Loading / generation ----------------------------------------------------

function load_fixture(path::AbstractString)
    open(path, "r") do f
        magic = read(f, 4)
        @assert String(magic) == "TSVD" "bad fixture magic"
        m = Int(read(f, Int32))
        p = Int(read(f, Int32))
        n = Int(read(f, Int32))
        # Disk: column-major (m, p, n). Read directly into a Julia (m, p, n) array.
        A_disk = Array{Float64,3}(undef, m, p, n)
        read!(f, A_disk)
        # Convert to (n, m, p) layout (matches Python tsvdm_core.py convention).
        A = permutedims(A_disk, (3, 1, 2))
        M = Array{Float64,2}(undef, n, n)
        read!(f, M)
        return A, M, (m, p, n)
    end
end

function generate(m::Int, p::Int, n::Int; seed::Int=0)
    Random.seed!(seed)
    A = randn(n, m, p)
    G = randn(n, n)
    F = qr(G)
    return A, Matrix(F.Q), (m, p, n)
end

# --- Driver ------------------------------------------------------------------

function parse_args(args::Vector{String})
    fixture = nothing
    gen = nothing
    reps = 1
    dump = nothing
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--gen"
            gen = (parse(Int, args[i+1]), parse(Int, args[i+2]), parse(Int, args[i+3]))
            i += 4
        elseif a == "--reps"
            reps = parse(Int, args[i+1]); i += 2
        elseif a == "--dump"
            dump = args[i+1]; i += 2
        elseif !startswith(a, "-")
            fixture = a; i += 1
        else
            error("unknown arg: $a")
        end
    end
    return (fixture, gen, reps, dump)
end

function main(args::Vector{String})
    fixture, gen, reps, dump = parse_args(args)
    A, M, (m, p, n) = if fixture !== nothing
        load_fixture(fixture)
    elseif gen !== nothing
        generate(gen...)
    else
        error("usage: tsvdm.jl <fixture.bin> [--reps N] [--dump out.bin]\n" *
              "       tsvdm.jl --gen M P N [--reps N] [--dump out.bin]")
    end

    # Warmup — covers Julia's first-run JIT compile cost (not timed).
    tsvdm(A, M)

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
        A_dump = permutedims(A_approx, (2, 3, 1))     # (n,m,p) -> (m,p,n)
        open(dump, "w") do f
            write(f, A_dump)                           # Julia writes column-major
        end
    end

    return rel_err < 1e-10 ? 0 : 1
end

exit(main(ARGS))
