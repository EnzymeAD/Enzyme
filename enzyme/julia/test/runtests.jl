using Enzyme
using Test
using FiniteDifferences
using ForwardDiff
using Zygote

# Test against FiniteDifferences
function test_scalar(f, x; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)
    ∂x = autodiff(f, Active(x))
    @test isapprox(∂x, fdm(f, x); rtol=rtol, atol=atol, kwargs...)
end


@testset "Internal tests" begin
    f(x) = 1.0 + x
    thunk = Enzyme.Thunk(f, Float64, (Active{Float64},))
    thunk = Enzyme.Thunk(f, Float64, (Const{Float64},))
end

@testset "Simple tests" begin
    f1(x) = 1.0 + x
    f2(x) = x*x
    @test autodiff(f1, Active(1.0)) ≈ 1.0
    @test autodiff(f2, Active(1.0)) ≈ 2.0
    test_scalar(f1, 1.0)
    test_scalar(f2, 1.0)
end

@testset "Taylor series tests" begin

# Taylor series for `-log(1-x)`
# eval at -log(1-1/2) = -log(1/2)
function euroad(f::T) where T
    g = zero(T)
    for i in 1:10^7
        g += f^i / i
    end
    return g
end

euroad′(x) = autodiff(euroad, Active(x))

@test euroad(0.5) ≈ -log(0.5) # -log(1-x)
@show euroad′(0.5)
@test euroad′(0.5) ≈ 2.0 # d/dx -log(1-x) = 1/(1-x)
test_scalar(euroad, 0.5)
end

@testset "Array tests" begin

    function arsum(f::Array{T}) where T
        g = zero(T)
        for elem in f
            g += elem
        end
        return g
    end

    inp = Float64[1.0, 2.0]
    dinp = Float64[0.0, 0.0]
    autodiff(arsum, Duplicated(inp, dinp))
    @test inp ≈ Float64[1.0, 2.0]
    @test dinp ≈ Float64[1.0, 1.0]
end

@testset "Advanced array tests" begin

    function arsum2(f::Array{T}) where T
        return sum(f)
    end
    inp = Float64[1.0, 2.0]
    dinp = Float64[0.0, 0.0]
    autodiff(arsum2, Duplicated(inp, dinp))
    @test inp ≈ Float64[1.0, 2.0]
    @test dinp ≈ Float64[1.0, 1.0]
end

@testset "Compare against" begin
    x = 3.0
    fd = central_fdm(5, 1)(sin, x)

    @test fd ≈ ForwardDiff.derivative(sin, x)
    @test fd ≈ autodiff(sin, Active(x)) 

    x = 0.2 + sin(3.0)
    fd = central_fdm(5, 1)(asin, x)

    @test fd ≈ ForwardDiff.derivative(asin, x)
    # @test fd ≈ autodiff(asin, Active(x))

    function foo(x)
        a = sin(x)
        b = 0.2 + a
        c = asin(b)
        return c
    end

    x = 3.0
    fd = central_fdm(5, 1)(foo, x)

    @test fd ≈ ForwardDiff.derivative(foo, x)
    @test fd ≈ Zygote.gradient(foo, x)[1]
    # @test fd ≈ autodiff(foo, Active(x))
    # test_scalar(foo, x)

    # Input type shouldn't matter
    x = 3
    @test fd ≈ ForwardDiff.derivative(foo, x)
    @test fd ≈ Zygote.gradient(foo, x)[1]
    # @test fd ≈ autodiff(foo, Active(x))
end

@testset "Bessel" begin
    """
        J(ν, z) := ∑ (−1)^k / Γ(k+1) / Γ(k+ν+1) * (z/2)^(ν+2k)
    """
    function besselj(ν, z, atol=1e-8)
        k = 0
        s = (z/2)^ν / factorial(ν)
        out = s
        while abs(s) > atol
            k += 1
            s *= (-1) / k / (k+ν) * (z/2)^2
            out += s
        end
        out
    end
    besselj0(z) = besselj(0, z)
    besselj1(z) = besselj(1, z)
    # autodiff(besselj, Const(0), Active(1.0))
    # autodiff(besselj, 0, Active(1.0))
    # @testset "besselj0/besselj1" for x in (1.0, -1.0, 0.0, 0.5, 10, -17.1,) # 1.5 + 0.7im)
    #     test_scalar(besselj0, x)
    #     test_scalar(besselj1, x)
    # end

end

## https://github.com/JuliaDiff/ChainRules.jl/tree/master/test/rulesets
# @testset "Packages" begin
#     include("packages/specialfunctions.jl")
# end

@testset "DiffTest" begin
    include("DiffTests.jl")

    n = rand()
    x, y = rand(5, 5), rand(26)
    A, B = rand(5, 5), rand(5, 5)

    # f returns Number
    @testset "Number to Number" for f in DiffTests.NUMBER_TO_NUMBER_FUNCS
        test_scalar(f, n)
    end

    # for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
    #     @test isa(f(y), Number)
    # end

    # for f in DiffTests.MATRIX_TO_NUMBER_FUNCS
    #     @test isa(f(x), Number)
    # end

    # for f in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS
    #     @test isa(f(A, B, x), Number)
    # end

    # # f returns Array

    # for f in DiffTests.NUMBER_TO_ARRAY_FUNCS
    #     @test isa(f(n), Array)
    # end

    # for f in DiffTests.ARRAY_TO_ARRAY_FUNCS
    #     @test isa(f(A), Array)
    #     @test isa(f(y), Array)
    # end

    # for f in DiffTests.MATRIX_TO_MATRIX_FUNCS
    #     @test isa(f(A), Array)
    # end

    # for f in DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS
    #     @test isa(f(A, B), Array)
    # end

    # # f! returns Nothing

    # for f! in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS
    #     @test isa(f!(y, x), Nothing)
    # end

    # for f! in DiffTests.INPLACE_NUMBER_TO_ARRAY_FUNCS
    #     @test isa(f!(y, n), Nothing)
    # end

end
