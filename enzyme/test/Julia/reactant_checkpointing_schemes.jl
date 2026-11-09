# Reactant-backend counterpart of checkpointing_schemes.jl.
#
# Exercises the same cos(s²)ⁿ function differentiated via:
#   PLAIN  — Enzyme full-tape (LLVM backend, no checkpointing)
#   ONLINE — while-loop, Online_r2 via Reactant MLIR backend
#
# Both must produce the same df/dx as the Checkpointing.jl version in
# checkpointing_schemes.jl (cross-validated via the shared plain Enzyme
# reference gradient).
#
# Usage: julia reactant_checkpointing_schemes.jl

using Test

import Pkg
Pkg.activate(; temp=true)
Pkg.develop(path="/Users/snarayan/soft/Reactant.jl/lib/ReactantCore")
Pkg.develop(path="/Users/snarayan/soft/Reactant.jl")
Pkg.add("Enzyme")

using Reactant
using ReactantCore: Online
using Enzyme

# ── Shared parameters ─────────────────────────────────────────────────────────
const x₀      = 1.2f0  # active initial state (Float32)
const n_steps  = 5      # number of loop iterations

# ── Reference gradient via central finite difference ─────────────────────────
function f_ref(x, n)
    s = x
    for _ in 1:n
        s = cos(s * s)
    end
    return s
end
const h_fd   = 1e-7
const dx_fd  = Float32((f_ref(Float64(x₀) + h_fd, n_steps) - f_ref(Float64(x₀) - h_fd, n_steps)) / (2h_fd))

# ── PLAIN — Enzyme full-tape (LLVM backend, no checkpointing) ────────────────
# This is the SAME reference used by checkpointing_schemes.jl, so if both
# the Checkpointing.jl and Reactant tests pass, the two backends agree.
function grad_plain(x₀, n)
    return Enzyme.gradient(
        Reverse,
        x -> sum(foldl((v, _) -> cos.(v .* v), 1:n; init=x)),
        copy([x₀])
    )[1]
end

# ── ONLINE — Reactant @trace with Online(budget) ─────────────────────────────
function online_body(x, n, ckpt)
    step = zero(n)
    @trace checkpointing=ckpt track_numbers=false while step < n
        x    = cos.(x .* x)
        step = step + one(step)
    end
    return sum(x)
end

function grad_online(x_ra, n_ra, budget)
    dx = Enzyme.make_zero(x_ra)
    Enzyme.autodiff(
        Reverse, online_body, Active,
        Duplicated(x_ra, dx), Const(n_ra), Const(Online(budget))
    )
    return dx
end

# ── Tests ─────────────────────────────────────────────────────────────────────
@testset "Checkpointing schemes (Reactant backend)" begin

    dx_plain = grad_plain(x₀, n_steps)
    n_ra     = Reactant.ConcreteRNumber(n_steps)

    @info "Reference gradients:" PLAIN=dx_plain FD_ref=dx_fd

    @testset "Plain Enzyme matches finite differences (rtol=1e-3)" begin
        @test isapprox(dx_plain[1], dx_fd, rtol=1e-3)
    end

    @testset "Online(budget=$budget) matches plain Enzyme" for budget in [2, 3, 5, 10]
        x_ra = Reactant.to_rarray(Float32[x₀])
        dx_online = @jit grad_online(x_ra, n_ra, budget)
        @test Array(dx_online) ≈ dx_plain
    end

    @testset "Online(budget=3) correct for n=$n steps" for n in [1, 3, 5, 8, 10]
        x_ref = copy(Float32[x₀])
        dx_ref = Enzyme.gradient(
            Reverse,
            x -> sum(foldl((v, _) -> cos.(v .* v), 1:n; init=x)),
            x_ref
        )[1]

        n_local = Reactant.ConcreteRNumber(n)
        x_ra = Reactant.to_rarray(Float32[x₀])
        dx_online = @jit grad_online(x_ra, n_local, 3)
        @test Array(dx_online) ≈ dx_ref
    end
end
