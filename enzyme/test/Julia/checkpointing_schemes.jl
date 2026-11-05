# Numerical comparison of four checkpointing strategies for the function
#   f(x, n) = (cos ∘ (s ↦ s²))ⁿ(x)   — i.e. cos(s²) applied n times
#
# The four schemes tested here mirror the four MLIR schemes in
#   test/MLIR/ReverseMode/checkpointing_all_schemes.mlir
# and exercise the same Revolve / Online_r2 algorithms via Checkpointing.jl:
#
#   PLAIN   — for-loop , no checkpointing  → Enzyme.jl full-tape autodiff
#   BINOM   — for-loop , binomial Revolve  → Checkpointing.Revolve
#   WTAPE   — while-loop, no checkpointing → Enzyme.jl full-tape autodiff
#   WONLINE — while-loop, online Online_r2 → Checkpointing.Online_r2
#
# All four must produce the same df/dx for the same (x₀, n).
#
# Dependencies: Checkpointing, Enzyme (install via `julia --project -e 'using
# Pkg; Pkg.add(["Checkpointing","Enzyme","Test"])'`).

using Test
using Checkpointing
using Enzyme

# ── Shared parameters ─────────────────────────────────────────────────────────
const x₀      = 1.2    # active initial state
const n_steps = 5      # number of loop iterations
const snaps   = 3      # checkpoint budget (binomial and online schemes)

# ── Reference gradient via central finite difference ─────────────────────────
# Gold standard used to cross-check all four AD results.
function f_ref(x, n)
    s = x
    for _ in 1:n
        s = cos(s * s)
    end
    return s
end
const h_fd   = 1e-7
const dx_ref = (f_ref(x₀ + h_fd, n_steps) - f_ref(x₀ - h_fd, n_steps)) / (2h_fd)

# ── Mutable-struct types for Checkpointing.jl ─────────────────────────────────
# Checkpointing.jl snapshots a mutable struct at each checkpoint.  The shadow
# struct (with zero-initialised gradient fields) accumulates the adjoint.

# For for-loop schemes: the loop index is managed externally by @ad_checkpoint.
mutable struct ForLoopState
    x::Float64
end

# For while-loop schemes: the step counter is part of the loop condition.
mutable struct WhileLoopState
    x::Float64
    k::Int
end

# ── One-step advance functions ─────────────────────────────────────────────────
function advance_for!(s::ForLoopState)
    s.x = cos(s.x * s.x)
    return nothing
end

function advance_while!(s::WhileLoopState)
    s.x = cos(s.x * s.x)
    s.k += 1
    return nothing
end

# ── Differentiable objective wrappers ─────────────────────────────────────────
# @ad_checkpoint inserts the checkpointing logic.  The Enzyme custom rules for
# checkpoint_for / checkpoint_while differentiate through the scheme correctly.

# BINOM: for-loop with Revolve binomial checkpointing.
# scheme::Union{Revolve,Periodic} restricts to schemes that support for-loops.
function obj_for(s::ForLoopState, scheme::Union{Revolve,Periodic}, n::Int)
    @ad_checkpoint scheme for i in 1:n
        advance_for!(s)
    end
    return s.x
end

# WONLINE: while-loop with Online_r2 online checkpointing.
# Online_r2 is the only scheme that supports while-loops with an unknown
# trip count; it discovers the count during the forward pass.
function obj_while(s::WhileLoopState, scheme::Online_r2, n::Int)
    @ad_checkpoint scheme while s.k < n
        advance_while!(s)
    end
    return s.x
end

# ── Gradient drivers ──────────────────────────────────────────────────────────

# PLAIN — scf.for, no checkpointing.
# Enzyme.jl stores the full forward tape at the LLVM IR level (equivalent to
# the memref<?xf32> dynamic-tape path in the MLIR full-tape test).
function grad_plain(x₀, n)
    function f_for(x, n)
        s = x
        for _ in 1:n
            s = cos(s * s)
        end
        return s
    end
    return autodiff(Reverse, f_for, Active, Active(x₀), Const(n))[1][1]
end

# BINOM — scf.for, binomial (offline Revolve) checkpointing.
# Revolve(snaps) creates a lazy scheme; @ad_checkpoint instantiates it with
# the correct element type and step count before running the loop.
function grad_binom(x₀, n, snaps)
    s  = ForLoopState(x₀)
    ds = ForLoopState(0.0)
    scheme = Revolve(snaps)
    autodiff(
        Enzyme.ReverseWithPrimal, obj_for,
        Duplicated(s, ds), Const(scheme), Const(n),
    )
    return ds.x
end

# WTAPE — scf.while, no checkpointing (full tape).
# Enzyme.jl differentiates the while loop at the LLVM IR level without any
# a-priori knowledge of the trip count — the trip count is discovered when the
# loop terminates, equivalent to the enzyme.push/pop full-tape path in MLIR.
function grad_while_plain(x₀, n)
    function f_while(x, n)
        s = x
        k = 0
        while k < n
            s = cos(s * s)
            k += 1
        end
        return s
    end
    return autodiff(Reverse, f_while, Active, Active(x₀), Const(n))[1][1]
end

# WONLINE — scf.while, Online_r2 (Stumm-Walther 2010) checkpointing.
# Online_r2(snaps) creates a lazy scheme; @ad_checkpoint discovers the trip
# count during the forward pass and places at most snaps checkpoints adaptively.
function grad_while_online(x₀, n, snaps)
    s  = WhileLoopState(x₀, 0)
    ds = WhileLoopState(0.0, 0)
    scheme = Online_r2(snaps)
    autodiff(
        Enzyme.ReverseWithPrimal, obj_while,
        Duplicated(s, ds), Const(scheme), Const(n),
    )
    return ds.x
end

# ── Tests ─────────────────────────────────────────────────────────────────────
@testset "All four checkpointing schemes produce the same gradient" begin

    dx_plain  = grad_plain(x₀, n_steps)
    dx_binom  = grad_binom(x₀, n_steps, snaps)
    dx_wtape  = grad_while_plain(x₀, n_steps)
    dx_online = grad_while_online(x₀, n_steps, snaps)

    @info "Computed gradients:" PLAIN=dx_plain BINOM=dx_binom WTAPE=dx_wtape ONLINE=dx_online FD_ref=dx_ref

    # Each scheme must match the finite-difference reference within FD accuracy.
    # rtol=1e-5 accounts for FD truncation; exact AD results have no additional
    # rounding error beyond normal floating-point.
    @testset "All schemes match the finite-difference reference (rtol=1e-5)" begin
        @test isapprox(dx_plain,  dx_ref, rtol=1e-5)
        @test isapprox(dx_binom,  dx_ref, rtol=1e-5)
        @test isapprox(dx_wtape,  dx_ref, rtol=1e-5)
        @test isapprox(dx_online, dx_ref, rtol=1e-5)
    end

    # The four AD results execute the same IEEE-754 arithmetic and must agree
    # exactly (up to the tiny round-off from different control-flow orderings).
    @testset "Checkpointed schemes agree with the full-tape reference (rtol=1e-12)" begin
        @test isapprox(dx_binom,  dx_plain, rtol=1e-12)
        @test isapprox(dx_wtape,  dx_plain, rtol=1e-12)
        @test isapprox(dx_online, dx_plain, rtol=1e-12)
    end
end
