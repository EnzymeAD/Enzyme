#!/usr/bin/env julia
# Oceananigans-style simulation differentiated via Reactant.jl with Online_r2
# checkpointing implemented in Enzyme's MLIR backend.
#
# This test exercises the full stack:
#   @trace checkpointing=Online(budget)
#     → ReactantCore.Online struct  (lib/ReactantCore/src/ReactantCore.jl)
#     → enzyme.online_checkpointing MLIR attribute  (src/Ops.jl)
#     → WhileOpInterfaceReverse::cacheWhileOnline  (SCFAutoDiffOpInterfaceImpl.cpp)
#     → enzyme.online_progress op  (EnzymeOps.td)
#     → LowerOnlineProgressPass  (LowerOnlineProgressPass.cpp)
#     → numerically correct gradient
#
# The "sim.running" stopping criterion is set INSIDE the loop body (time_step!),
# so the trip count is unknown before the loop — the canonical Online_r2 use case.
#
# Usage: julia reactant_online_checkpointing.jl

using Test

import Pkg
Pkg.activate(; temp=true)

# Use the Reactant fork that has Online(budget) support.
# To install from GitHub: Pkg.add(url="https://github.com/sriharikrishna/Reactant.jl")
# Locally we develop from the checkout so edits are picked up immediately.
Pkg.develop(path="/Users/snarayan/soft/Reactant.jl/lib/ReactantCore")
Pkg.develop(path="/Users/snarayan/soft/Reactant.jl")
Pkg.add("Enzyme")

using Reactant
using ReactantCore: Online
using Enzyme

# ── Body function: cos(s²) iterated n times ────────────────────────────────────
# Written with @trace so that Reactant compiles it to MLIR and attaches the
# enzyme.online_checkpointing attribute to the stablehlo.while op.
# The stopping criterion (step < n) is set INSIDE the body, mirroring
# Oceananigans' time_step! setting sim.running = (time < stop_time).

function online_ckpt_body(x, n, ckpt)
    step = zero(n)
    @trace checkpointing=ckpt track_numbers=false while step < n
        x    = cos.(x .* x)
        step = step + one(step)
    end
    return sum(x)
end

function online_ckpt_grad(x, n, ckpt)
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff(
        Reverse, online_ckpt_body, Active, Duplicated(x, dx), Const(n), Const(ckpt)
    )
    return dx
end

# ── Tests ──────────────────────────────────────────────────────────────────────
@testset "Reactant Online_r2 checkpointing (MLIR backend)" begin

    x  = Float32[1.2]

    # ── Small trip count (sanity check) ──────────────────────────────────────
    @testset "Small loop: n=5, budget=$budget" for budget in [2, 3]
        n = 5
        dx_ref = Enzyme.gradient(
            Reverse,
            x -> sum(foldl((v, _) -> cos.(v .* v), 1:n; init=x)),
            copy(x)
        )[1]
        n_ra = Reactant.ConcreteRNumber(n)
        x_fresh = Reactant.to_rarray(x)
        dx_online = @jit online_ckpt_grad(x_fresh, n_ra, Online(budget))
        @test Array(dx_online) ≈ dx_ref
    end

    # ── Stress tests: budget << trip count ────────────────────────────────────
    # These force Online_r2 to overwrite old checkpoints and replay segments.
    # Online_r2 with s checkpoints supports at most (s+2)(s+1)/2 iterations.
    @testset "Stress: n=$n, budget=$budget" for (n, budget) in
            [(8, 3), (10, 3), (20, 5), (60, 10), (130, 15)]
        dx_ref = Enzyme.gradient(
            Reverse,
            x -> sum(foldl((v, _) -> cos.(v .* v), 1:n; init=x)),
            copy(x)
        )[1]
        n_ra = Reactant.ConcreteRNumber(n)
        x_fresh = Reactant.to_rarray(x)
        dx_online = @jit online_ckpt_grad(x_fresh, n_ra, Online(budget))
        @test Array(dx_online) ≈ dx_ref
    end

    @testset "Online IR attributes present" begin
        x_ra = Reactant.to_rarray(x)
        n_ra = Reactant.ConcreteRNumber(5)
        ir = sprint(show, @code_hlo optimize="enzyme-batch" online_ckpt_body(x_ra, n_ra, Online(5)))
        @test occursin("enzyme.enable_checkpointing", ir)
        @test occursin("enzyme.online_checkpointing", ir)
        @test occursin("enzyme.checkpoint_period = 5", ir)
    end
end
