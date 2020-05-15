module Compiler

using GPUCompiler
using LLVM
using LLVM.Interop

import GPUCompiler: FunctionSpec, codegen

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

# Define EnzymeTarget & EnzymeJob
using LLVM: triple, Target, TargetMachine
import GPUCompiler: llvm_triple

Base.@kwdef struct EnzymeTarget <: AbstractCompilerTarget
end

GPUCompiler.isintrinsic(::EnzymeTarget, fn::String) = true
GPUCompiler.can_throw(::EnzymeTarget) = true

llvm_triple(::EnzymeTarget) = triple()

# GPUCompiler.llvm_datalayout(::EnzymeTarget) =  nothing

function GPUCompiler.llvm_machine(target::EnzymeTarget)
    t = Target(llvm_triple(target))
    tm = TargetMachine(t, llvm_triple(target))
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

GPUCompiler.runtime_module(target::EnzymeTarget) = Runtime

## job

export EnzymeJob

Base.@kwdef struct EnzymeJob <: AbstractCompilerJob
    target::EnzymeTarget
    source::FunctionSpec
end

# TODO: We shouldn't blancket opt-out
GPUCompiler.check_invocation(job::EnzymeJob, entry::LLVM.Function) = nothing

import GPUCompiler: target, source
target(job::EnzymeJob) = job.target
source(job::EnzymeJob) = job.source

Base.similar(job::EnzymeJob, source::FunctionSpec) =
    EnzymeJob(target=job.target, source=source)

function Base.show(io::IO, job::EnzymeJob)
    print(io, "Enzyme CompilerJob of ", GPUCompiler.source(job))
end

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
GPUCompiler.runtime_slug(job::EnzymeJob) = "enzyme" 

include("compiler/optimize.jl")
include("compiler/cassette.jl")

const libjulia = Ref{Ptr{Cvoid}}(C_NULL)

# GPUCompiler.check_ir(job::EnzymeJob, args...) = nothing
import GPUCompiler: IRError, DYNAMIC_CALL, DELAYED_BINDING, RUNTIME_FUNCTION, UNKNOWN_FUNCTION, POINTER_FUNCTION
function check_ir!(job::EnzymeJob, errors::Vector{IRError}, inst::LLVM.CallInst)
    bt = GPUCompiler.backtrace(inst)
    dest = called_value(inst)
    if isa(dest, LLVM.Function)
        fn = LLVM.name(dest)

        # some special handling for runtime functions that we don't implement
        if fn == "jl_get_binding_or_error"
            try
                m, sym, _ = operands(inst)
                sym = first(operands(sym::ConstantExpr))::ConstantInt
                sym = convert(Int, sym)
                sym = Ptr{Cvoid}(sym)
                sym = Base.unsafe_pointer_to_objref(sym)
                push!(errors, (DELAYED_BINDING, bt, sym))
            catch e
                isa(e,TypeError) || rethrow()
                @debug "Decoding arguments to jl_get_binding_or_error failed" inst bb=LLVM.parent(inst)
                push!(errors, (DELAYED_BINDING, bt, nothing))
            end
        elseif fn == "jl_invoke"
            try
                if VERSION < v"1.3.0-DEV.244"
                    meth, args, nargs, _ = operands(inst)
                else
                    f, args, nargs, meth = operands(inst)
                end
                meth = first(operands(meth::ConstantExpr))::ConstantExpr
                meth = first(operands(meth))::ConstantInt
                meth = convert(Int, meth)
                meth = Ptr{Cvoid}(meth)
                meth = Base.unsafe_pointer_to_objref(meth)::Core.MethodInstance
                push!(errors, (DYNAMIC_CALL, bt, meth.def))
            catch e
                isa(e,TypeError) || rethrow()
                @debug "Decoding arguments to jl_invoke failed" inst bb=LLVM.parent(inst)
                push!(errors, (DYNAMIC_CALL, bt, nothing))
            end
        elseif fn == "jl_apply_generic"
            try
                if VERSION < v"1.3.0-DEV.244"
                    args, nargs, _ = operands(inst)
                    ## args is a buffer where arguments are stored in
                    f, args = user.(uses(args))
                    ## first store into the args buffer is a direct store
                    f = first(operands(f::LLVM.StoreInst))::ConstantExpr
                else
                    f, args, nargs, _ = operands(inst)
                end

                f = first(operands(f))::ConstantExpr # get rid of addrspacecast
                f = first(operands(f))::ConstantInt # get rid of inttoptr
                f = convert(Int, f)
                f = Ptr{Cvoid}(f)
                f = Base.unsafe_pointer_to_objref(f)
                push!(errors, (DYNAMIC_CALL, bt, f))
            catch e
                isa(e,TypeError) || rethrow()
                @debug "Decoding arguments to jl_apply_generic failed" inst bb=LLVM.parent(inst)
                push!(errors, (DYNAMIC_CALL, bt, nothing))
            end

        # detect calls to undefined functions
        elseif isdeclaration(dest) && intrinsic_id(dest) == 0 && !isintrinsic(target(job), fn)
            # figure out if the function lives in the Julia runtime library
            if libjulia[] == C_NULL
                paths = filter(Libdl.dllist()) do path
                    name = splitdir(path)[2]
                    startswith(name, "libjulia")
                end
                libjulia[] = Libdl.dlopen(first(paths))
            end

            if Libdl.dlsym_e(libjulia[], fn) != C_NULL
                push!(errors, (RUNTIME_FUNCTION, bt, LLVM.name(dest)))
            else
                push!(errors, (UNKNOWN_FUNCTION, bt, LLVM.name(dest)))
            end
        end

    elseif isa(dest, InlineAsm)
        # let's assume it's valid ASM

    elseif isa(dest, ConstantExpr)
        # Enzyme should be able to handle these
    #     # detect calls to literal pointers
    #     if occursin("inttoptr", string(dest))
    #         # extract the literal pointer
    #         ptr_arg = first(operands(dest))
    #         GPUCompiler.@compiler_assert isa(ptr_arg, ConstantInt) job
    #         ptr_val = convert(Int, ptr_arg)
    #         ptr = Ptr{Cvoid}(ptr_val)

    #         # look it up in the Julia JIT cache
    #         frames = ccall(:jl_lookup_code_address, Any, (Ptr{Cvoid}, Cint,), ptr, 0)
    #         if length(frames) >= 1
    #             GPUCompiler.@compiler_assert length(frames) == 1 job frames=frames
    #             if VERSION >= v"1.4.0-DEV.123"
    #                 fn, file, line, linfo, fromC, inlined = last(frames)
    #             else
    #                 fn, file, line, linfo, fromC, inlined, ip = last(frames)
    #             end
    #             push!(errors, (POINTER_FUNCTION, bt, fn))
    #         else
    #             push!(errors, (POINTER_FUNCTION, bt, nothing))
    #         end
    #     end
    end

    return errors
end


end