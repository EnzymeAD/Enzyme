module Compiler

using GPUCompiler
using LLVM
using LLVM.Interop

import GPUCompiler: CompilerJob, FunctionSpec, codegen

import Libdl
llvmver = LLVM.version().major
if haskey(ENV, "ENZYME_PATH")
    enzyme_path = ENV["ENZYME_PATH"]
else
    error("Please set the environment variable ENZYME_PATH")
end
const libenzyme = abspath(joinpath(enzyme_path, "LLVMEnzyme-$(llvmver).$(Libdl.dlext)"))

if !isfile(libenzyme)
    error("$(libenzyme) does not exist, Please specify a correct path in ENZYME_PATH, and restart Julia.")
end

if Libdl.dlopen_e(libenzyme) in (C_NULL, nothing)
    error("$(libenzyme) cannot be opened, Please specify a correct path in ENZYME_PATH, and restart Julia.")
end

function __init__()
    Libdl.dlopen(libenzyme, Libdl.RTLD_GLOBAL)
    LLVM.clopts("-enzyme_preopt=0")
end

# Define EnzymeTarget
using LLVM: Target, TargetMachine

Base.@kwdef struct EnzymeTarget <: AbstractCompilerTarget
end
GPUCompiler.llvm_triple(::EnzymeTarget) = Sys.MACHINE

# GPUCompiler.llvm_datalayout(::EnzymeTarget) =  nothing

function GPUCompiler.llvm_machine(target::EnzymeTarget)
    triple = GPUCompiler.llvm_triple(target)
    t = Target(; triple = triple)
    tm = TargetMachine(t, triple)
    LLVM.asm_verbosity!(tm, true)

    return tm
end

module Runtime
    # the runtime library
    signal_exception() = return
    malloc(sz) =  return
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end


struct EnzymeCompilerParams <: AbstractCompilerParams end

## job

# TODO: We shouldn't blancket opt-out
GPUCompiler.check_invocation(job::CompilerJob{EnzymeTarget}, entry::LLVM.Function) = nothing

GPUCompiler.runtime_module(target::CompilerJob{EnzymeTarget}) = Runtime
GPUCompiler.isintrinsic(::CompilerJob{EnzymeTarget}, fn::String) = true
GPUCompiler.can_throw(::CompilerJob{EnzymeTarget}) = true

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
GPUCompiler.runtime_slug(job::CompilerJob{EnzymeTarget}) = "enzyme" 

include("compiler/optimize.jl")
include("compiler/cassette.jl")
include("compiler/validation.jl")



end