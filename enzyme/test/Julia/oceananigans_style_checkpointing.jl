#!/usr/bin/env julia
# Simplified Oceananigans-style simulation with Online_r2 checkpointing.
#
# Mirrors the structure of:
#   - Simulations/run.jl     : run!(sim) contains `while sim.running; time_step!(sim); end`
#   - TimeSteppers/rk3.jl   : time_step!(sim) advances state AND sets sim.running = false
#                               when a stopping criterion is met
#
# The trip count is NOT known before `run!` is called — it is determined by
# the physics (here: `sim.time >= sim.stop_time`) inside each time_step!.
# This is the canonical use-case for Online_r2 online checkpointing.
#
# Three gradient estimates are compared:
#   1. Finite differences (gold standard)
#   2. Enzyme full-tape (no checkpointing)
#   3. Online_r2 checkpointing (Checkpointing.jl)
#
# Usage: julia oceananigans_style_checkpointing.jl

using Test

import Pkg
Pkg.activate(; temp=true)
Pkg.add(["Checkpointing", "Enzyme"])

using Checkpointing
using Enzyme

# ── Simulation struct ─────────────────────────────────────────────────────────
# Mirrors Oceananigans' Simulation: carries state, clock, and a running flag
# that time_step! may flip to false to terminate the loop.

mutable struct OceanSim
    state     :: Float64   # prognostic variable (e.g. temperature / sea-ice fraction)
    time      :: Float64   # current simulation time
    dt        :: Float64   # fixed timestep size
    stop_time :: Float64   # stop when time >= stop_time (set by callbacks in real code)
    running   :: Bool      # flipped to false by time_step! when stop criterion is met
end

# ── time_step! ────────────────────────────────────────────────────────────────
# Advances the simulation by one timestep.
# Mirrors Oceananigans' time_step! (RK3): updates the model state,
# increments the clock, then evaluates stopping callbacks.
# Here the "physics" is cos(s²) iterated — the same function we use
# throughout to keep expected gradients analytically tractable.

function time_step!(sim::OceanSim)
    # ── physics: one explicit Euler step ─────────────────────────────────────
    # In real code this would be a Runge-Kutta substep over PDE tendencies.
    sim.state = cos(sim.state * sim.state)

    # ── clock ─────────────────────────────────────────────────────────────────
    sim.time += sim.dt

    # ── stopping criterion (Oceananigans: stop-time callback) ─────────────────
    # The decision is made INSIDE the timestep, not before the loop —
    # this is why the trip count is unknown a-priori and Online_r2 is needed.
    sim.running = sim.time < sim.stop_time

    return nothing
end

# ── run! (Online_r2 checkpointing variant) ────────────────────────────────────
# Mirrors the `while sim.running; time_step!(sim); end` loop in
# Oceananigans' run!.  The Online_r2 scheme intercepts the loop via
# @ad_checkpoint and places checkpoints adaptively.

function run_online!(sim::OceanSim, scheme::Online_r2)
    sim.running = true   # mirror: sim.running = true  at the top of run!
    @ad_checkpoint scheme while sim.running
        time_step!(sim)
    end
    return sim.state     # objective: final state value (e.g. a loss function)
end

# ── run! (plain, no checkpointing) ────────────────────────────────────────────
# Used for full-tape Enzyme comparison.

function run_plain!(sim::OceanSim)
    sim.running = true
    while sim.running
        time_step!(sim)
    end
    return sim.state
end

# ── Gradient helpers ──────────────────────────────────────────────────────────

function grad_finitediff(s0, dt, stop_time; h=1e-6)
    function fwd(s)
        sim = OceanSim(s, 0.0, dt, stop_time, true)
        run_plain!(sim)
    end
    (fwd(s0 + h) - fwd(s0 - h)) / (2h)
end

function grad_fulltape(s0, dt, stop_time)
    sim  = OceanSim(s0,  0.0, dt, stop_time, true)
    dsim = OceanSim(0.0, 0.0, 0.0, 0.0, false)
    autodiff(Enzyme.ReverseWithPrimal, run_plain!, Duplicated(sim, dsim))
    return dsim.state
end

function grad_online(s0, dt, stop_time, snaps)
    sim  = OceanSim(s0,  0.0, dt, stop_time, true)
    dsim = OceanSim(0.0, 0.0, 0.0, 0.0, false)
    scheme = Online_r2(snaps)
    autodiff(Enzyme.ReverseWithPrimal, run_online!, Duplicated(sim, dsim), Const(scheme))
    return dsim.state
end

# ── Tests ─────────────────────────────────────────────────────────────────────

@testset "Oceananigans-style Online_r2 checkpointing" begin

    s0        = 1.2          # initial state (e.g. sea-surface temperature)
    dt        = 1.0          # timestep = 1 "day"
    rtol      = 1e-5         # relative tolerance for gradient comparison

    # Test multiple stop_times (= different unknown trip counts)
    for stop_time in [1.0, 3.0, 5.0, 8.0, 10.0]
        n_steps = Int(stop_time / dt)   # actual number of steps (known analytically)

        gfd  = grad_finitediff(s0, dt, stop_time)
        gft  = grad_fulltape(s0, dt, stop_time)

        @testset "stop_time=$stop_time (n=$n_steps steps)" begin

            # Full-tape Enzyme matches finite differences
            @test isapprox(gft, gfd; rtol=rtol)

            # Online_r2 with several checkpoint budgets matches full-tape
            for snaps in [2, 3, 5, 10]
                g = grad_online(s0, dt, stop_time, snaps)
                @test isapprox(g, gft; rtol=rtol)
            end
        end
    end

    # Verify the key property: stop criterion evaluated INSIDE time_step!
    # — the caller (run!) does not know the trip count before the loop.
    @testset "Trip count unknown before loop" begin
        # stop_time is a non-integer multiple of dt.  The stopping criterion
        # is evaluated INSIDE time_step! after advancing the clock:
        #   sim.running = sim.time < sim.stop_time
        # so the loop runs until sim.time first reaches or exceeds stop_time.
        # That means n_steps = ceil(stop_time / dt), not floor.
        # The caller (run!) never computes this — it is determined dynamically.
        for stop_time in [2.5, 4.7, 9.1]
            n_expected = Int(ceil(stop_time / dt))   # actual steps (determined at runtime)
            sim = OceanSim(s0, 0.0, dt, stop_time, true)
            run_plain!(sim)
            # Confirm the loop ran exactly ceil(stop_time/dt) steps
            @test isapprox(sim.time, n_expected * dt; atol=1e-10)

            # Online_r2 gradient still correct despite unknown trip count
            gfd = grad_finitediff(s0, dt, stop_time)
            g   = grad_online(s0, dt, stop_time, 3)
            @test isapprox(g, gfd; rtol=1e-4)
        end
    end
end
