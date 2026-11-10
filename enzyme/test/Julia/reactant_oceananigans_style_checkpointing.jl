#!/usr/bin/env julia
# Reactant-backend counterpart of oceananigans_style_checkpointing.jl.
#
# Simplified Oceananigans-style simulation differentiated via Reactant.jl
# with Online_r2 checkpointing implemented in Enzyme's MLIR backend.
#
# The trip count is NOT known before the loop — it is determined by
# the physics (`time >= stop_time`) inside each time step.  This mirrors
# Oceananigans' `while sim.running; time_step!(sim); end`.
#
# Cross-validation: both this file and oceananigans_style_checkpointing.jl
# (the Checkpointing.jl version) compare against the same plain Enzyme
# full-tape reference gradient.  If both test suites pass, the two backends
# agree on every (s₀, stop_time, budget) combination.
#
# Usage: julia reactant_oceananigans_style_checkpointing.jl

using Test

import Pkg
Pkg.activate(; temp=true)
Pkg.develop(path="/Users/snarayan/soft/Reactant.jl/lib/ReactantCore")
Pkg.develop(path="/Users/snarayan/soft/Reactant.jl")
Pkg.add("Enzyme")

using Reactant
using ReactantCore: Online
using Enzyme

# ── Body function (Oceananigans-style) ────────────────────────────────────────
# The stopping criterion (`time < stop_time`) is evaluated INSIDE the loop
# body — the caller does not know the trip count before the loop.
# With dt = 1, the loop runs ceil(stop_time) times for non-integer stop_time,
# and exactly stop_time times for integer stop_time.

function ocean_body(state, stop_time, ckpt)
    time = zero(stop_time)
    dt   = one(stop_time)
    @trace checkpointing=ckpt track_numbers=false while time < stop_time
        state = cos.(state .* state)
        time = time + dt
    end
    return sum(state)
end

function ocean_grad(state_ra, stop_time_ra, budget)
    dx = Enzyme.make_zero(state_ra)
    Enzyme.autodiff(
        Reverse, ocean_body, Active,
        Duplicated(state_ra, dx), Const(stop_time_ra), Const(Online(budget))
    )
    return dx
end

# ── Reference: scalar f(x, n) for finite differences ────────────────────────
function f_scalar(s0, n)
    s = s0
    for _ in 1:n
        s = cos(s * s)
    end
    return s
end

function grad_finitediff(s0, n; h=1e-4)
    (f_scalar(s0 + h, n) - f_scalar(s0 - h, n)) / (2h)
end

# ── Reference: plain Enzyme full-tape (LLVM backend) ─────────────────────────
# This is the SAME reference used by oceananigans_style_checkpointing.jl
# (the Checkpointing.jl version), so if both test suites pass, the two
# backends agree on every (s₀, stop_time, budget) combination.
function grad_fulltape(s0, n)
    Enzyme.gradient(
        Reverse,
        x -> sum(foldl((v, _) -> cos.(v .* v), 1:n; init=x)),
        copy(Float32[s0])
    )[1]
end

# ── Tests ─────────────────────────────────────────────────────────────────────
@testset "Oceananigans-style Online_r2 checkpointing (Reactant)" begin

    s0 = 1.2f0    # initial state (e.g. sea-surface temperature)

    # Include large trip counts with small budgets so Online_r2 must overwrite
    # old checkpoints and replay segments — not just store everything.
    # Online_r2 with s checkpoints supports at most (s+2)(s+1)/2 iterations.
    online_r2_max(s) = (s + 2) * (s + 1) ÷ 2

    @testset "stop_time=$stop_time" for stop_time in
            Float32[3.0, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0]

        n_steps = Int(stop_time)
        gfd = grad_finitediff(Float64(s0), n_steps)
        gft = grad_fulltape(s0, n_steps)

        # Only test budgets that are (a) strictly less than the trip count and
        # (b) large enough to support the trip count under Online_r2.
        budgets = filter(b -> b < n_steps && n_steps <= online_r2_max(b), [2, 3, 5, 10, 15])

        @testset "Full-tape Enzyme matches finite differences" begin
            @test isapprox(gft[1], Float32(gfd); rtol=1e-3)
        end

        @testset "Online(budget=$budget) matches full-tape" for budget in budgets
            stop_ra  = Reactant.ConcreteRNumber(stop_time)
            state_ra = Reactant.to_rarray(Float32[s0])
            dx = @jit ocean_grad(state_ra, stop_ra, budget)
            @test Array(dx) ≈ gft
        end
    end

    # Non-integer stop_time: trip count = ceil(stop_time / dt), determined
    # dynamically by the `time < stop_time` condition inside the loop.
    @testset "Trip count unknown before loop (stop_time=$stop_time)" for stop_time in
            Float32[2.5, 4.7, 9.1]

        n_expected = Int(ceil(stop_time))
        gfd = grad_finitediff(Float64(s0), n_expected)

        @testset "Online(budget=3) correct for non-integer stop_time" begin
            stop_ra  = Reactant.ConcreteRNumber(stop_time)
            state_ra = Reactant.to_rarray(Float32[s0])
            dx = @jit ocean_grad(state_ra, stop_ra, 3)
            @test isapprox(Array(dx)[1], Float32(gfd); rtol=1e-3)
        end
    end
end
