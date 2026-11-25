#!/usr/bin/env python3
"""Measure the host CPU cost model Poseidon prices CPU candidates from.

Each benchmark is a JIT-compiled loop whose every trip runs 128 instances of
one operation of float or double, the formats a precision change offers a host
function. A trial run sizes the trip count so that a timed run lasts TARGET_S
seconds, and the best of REPS timed runs counts. A row is the time of one
operation, less the loop's own time per trip, in the unit of the GPU rows:
clock cycles times two, a cycle being the latency of a dependent integer add
as timed just before the row.
Rows go to stdout as they are measured, progress to stderr."""

import argparse
import ctypes
import os
import random
import struct
import sys
import tempfile
import time

import llvmlite.binding as llvm


random.seed(42)

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

FAST_MATH_FLAG = "fast"

unrolled = 128
TARGET_S = 0.25
TRIAL_S = 0.01
REPS = 3

precisions = ["float", "double"]

intrinsic_suffix = {"float": "f32", "double": "f64"}


def float_to_llvm_hex(f, precision):
    # LLVM IR spells a float constant as the double of the same value.
    if precision == "float":
        f = struct.unpack("<f", struct.pack("<f", f))[0]
    [i] = struct.unpack(">Q", struct.pack(">d", f))
    return f"0x{i:016X}"


def generate_random_fp(precision):
    if precision == "double":
        return random.uniform(-1e10, 1e10)
    return random.uniform(-1e5, 1e5)


OP_INFO = {
    "fneg": {"llvm_instr": "fneg", "num_operands": 1, "kind": "arithmetic"},
    "fadd": {"llvm_instr": "fadd", "num_operands": 2, "kind": "arithmetic"},
    "fsub": {"llvm_instr": "fsub", "num_operands": 2, "kind": "arithmetic"},
    "fmul": {"llvm_instr": "fmul", "num_operands": 2, "kind": "arithmetic"},
    "fdiv": {"llvm_instr": "fdiv", "num_operands": 2, "kind": "arithmetic"},
    "fcmp": {"llvm_instr": "fcmp", "num_operands": 2, "kind": "compare"},
}

CASTS = [("fpext", "float", "double"), ("fptrunc", "double", "float")]

FUNC_INFO = {
    "fmuladd": {"intrinsic": "llvm.fmuladd", "num_operands": 3},
    "sin": {"intrinsic": "llvm.sin", "num_operands": 1},
    "cos": {"intrinsic": "llvm.cos", "num_operands": 1},
    "tan": {"intrinsic": None, "num_operands": 1},
    "exp": {"intrinsic": "llvm.exp", "num_operands": 1},
    "log": {"intrinsic": "llvm.log", "num_operands": 1},
    "sqrt": {"intrinsic": "llvm.sqrt", "num_operands": 1},
    "expm1": {"intrinsic": None, "num_operands": 1},
    "log1p": {"intrinsic": None, "num_operands": 1},
    "cbrt": {"intrinsic": None, "num_operands": 1},
    "pow": {"intrinsic": "llvm.pow", "num_operands": 2},
    "fabs": {"intrinsic": "llvm.fabs", "num_operands": 1},
    "fma": {"intrinsic": "llvm.fma", "num_operands": 3},
    "maxnum": {"intrinsic": "llvm.maxnum", "num_operands": 2},
    "minnum": {"intrinsic": "llvm.minnum", "num_operands": 2},
    "ceil": {"intrinsic": "llvm.ceil", "num_operands": 1},
    "floor": {"intrinsic": "llvm.floor", "num_operands": 1},
    "exp2": {"intrinsic": "llvm.exp2", "num_operands": 1},
    "log10": {"intrinsic": "llvm.log10", "num_operands": 1},
    "log2": {"intrinsic": "llvm.log2", "num_operands": 1},
    "rint": {"intrinsic": "llvm.rint", "num_operands": 1},
    "round": {"intrinsic": "llvm.round", "num_operands": 1},
    "trunc": {"intrinsic": "llvm.trunc", "num_operands": 1},
    "copysign": {"intrinsic": "llvm.copysign", "num_operands": 2},
    "fdim": {"intrinsic": None, "num_operands": 2},
    "fmod": {"intrinsic": None, "num_operands": 2},
    "asin": {"intrinsic": None, "num_operands": 1},
    "acos": {"intrinsic": None, "num_operands": 1},
    "atan": {"intrinsic": None, "num_operands": 1},
    "atan2": {"intrinsic": None, "num_operands": 2},
    "sinh": {"intrinsic": None, "num_operands": 1},
    "cosh": {"intrinsic": None, "num_operands": 1},
    "tanh": {"intrinsic": None, "num_operands": 1},
    "asinh": {"intrinsic": None, "num_operands": 1},
    "acosh": {"intrinsic": None, "num_operands": 1},
    "atanh": {"intrinsic": None, "num_operands": 1},
    "hypot": {"intrinsic": None, "num_operands": 2},
    "erf": {"intrinsic": None, "num_operands": 1},
    "lgamma": {"intrinsic": None, "num_operands": 1},
    "tgamma": {"intrinsic": None, "num_operands": 1},
    "remainder": {"intrinsic": None, "num_operands": 2},
    "powi": {"intrinsic": "llvm.powi", "num_operands": 2},
}


def generate_loop_code(llvm_type, body_instructions, final_acc_reg):
    code = f"""
define i32 @main(i32 %n) optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca {llvm_type}
  store i32 0, i32* %i
  store {llvm_type} 0.0, {llvm_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, %n
  br i1 %cond, label %body, label %exit

body:
  %acc_val0 = load {llvm_type}, {llvm_type}* %acc
{body_instructions}
  store {llvm_type} {final_acc_reg}, {llvm_type}* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %final_acc = load {llvm_type}, {llvm_type}* %acc
  call void @use({llvm_type} %final_acc)
  ret i32 0
}}

define void @use({llvm_type} %val) {{
  ret void
}}
"""
    return code


def generate_arithmetic_op_code(op_key, precision):
    """Generate LLVM IR for a basic arithmetic operator (or fneg) based on OP_INFO."""
    op_info = OP_INFO[op_key]
    llvm_type = precision
    body_lines = ""
    for idx in range(unrolled):
        operands = []
        for _ in range(op_info["num_operands"]):
            f_val = generate_random_fp(precision)
            operands.append(float_to_llvm_hex(f_val, precision))

        line = f"  %result{idx} = {op_info['llvm_instr']} {FAST_MATH_FLAG} {llvm_type} {', '.join(operands)}"
        body_lines += line + "\n"
        body_lines += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {llvm_type} %acc_val{idx}, %result{idx}\n"

    final_acc = f"%acc_val{unrolled}"
    return generate_loop_code(llvm_type, body_lines, final_acc)


def generate_compare_op_code(precision):
    """Generate LLVM IR for an fcmp (comparison) operation."""
    llvm_type = precision

    body_lines = ""
    for idx in range(unrolled):
        f_a = generate_random_fp(precision)
        f_b = generate_random_fp(precision)
        a_hex = float_to_llvm_hex(f_a, precision)
        b_hex = float_to_llvm_hex(f_b, precision)
        line = f"  %cmp{idx} = fcmp {FAST_MATH_FLAG} olt {llvm_type} {a_hex}, {b_hex}"
        body_lines += line + "\n"
        body_lines += f"  %cmp_int{idx} = zext i1 %cmp{idx} to i32\n"

    code = f"""
define i32 @main(i32 %n) optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca i32
  store i32 0, i32* %i
  store i32 0, i32* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, %n
  br i1 %cond, label %body, label %exit

body:
  %acc_val0 = load i32, i32* %acc
{body_lines}
"""

    for idx in range(unrolled):
        code += f"  %acc_val{idx+1} = add i32 %acc_val{idx}, %cmp_int{idx}\n"

    final_acc = f"%acc_val{unrolled}"
    code += f"""
  store i32 {final_acc}, i32* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %final_acc = load i32, i32* %acc
  call void @use_i32(i32 %final_acc)
  ret i32 0
}}

define void @use_i32(i32 %val) {{
  ret void
}}
"""
    return code


def generate_cast_op_code(op_key, src_precision, dst_precision):
    """Generate LLVM IR for a cast operation (fptrunc or fpext)."""
    body_lines = ""
    for idx in range(unrolled):
        f_val = generate_random_fp(src_precision)
        hex_val = float_to_llvm_hex(f_val, src_precision)
        line = f"  %result{idx} = {op_key} {src_precision} {hex_val} to {dst_precision}"
        body_lines += line + "\n"
        body_lines += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {dst_precision} %acc_val{idx}, %result{idx}\n"

    final_acc = f"%acc_val{unrolled}"
    return generate_loop_code(dst_precision, body_lines, final_acc)


def generate_function_call_code(func_name, precision):
    """Generate LLVM IR for a function call based on FUNC_INFO."""

    func_info = FUNC_INFO[func_name]
    llvm_type = precision

    if func_info["intrinsic"]:
        fn = f"{func_info['intrinsic']}.{intrinsic_suffix[precision]}"
    else:
        fn = func_name

    num_operands = func_info["num_operands"]

    body_lines = "  %acc_val0 = load " + llvm_type + ", " + llvm_type + "* %acc\n"

    for idx in range(unrolled):
        if func_name == "powi":
            f_val = generate_random_fp(precision)
            i_val = random.randint(-10, 10)
            f_hex = float_to_llvm_hex(f_val, precision)
            args = f"{llvm_type} {f_hex}, i32 {i_val}"
        else:
            args = ", ".join(
                f"{llvm_type} {float_to_llvm_hex(generate_random_fp(precision), precision)}"
                for _ in range(num_operands)
            )
        body_lines += f"  %result{idx} = call {FAST_MATH_FLAG} {llvm_type} @{fn}({args})\n"
        body_lines += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {llvm_type} %acc_val{idx}, %result{idx}\n"

    if func_name == "powi":
        decl = f"declare {llvm_type} @{fn}({llvm_type}, i32)"
    else:
        arg_types = ", ".join([llvm_type] * num_operands)
        decl = f"declare {llvm_type} @{fn}({arg_types})"

    code = f"""
{decl}
define i32 @main(i32 %n) optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca {llvm_type}
  store i32 0, i32* %i
  store {llvm_type} 0.0, {llvm_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, %n
  br i1 %cond, label %body, label %exit

body:
{body_lines}
  store {llvm_type} %acc_val{unrolled}, {llvm_type}* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %final_acc = load {llvm_type}, {llvm_type}* %acc
  call void @use({llvm_type} %final_acc)
  ret i32 0
}}

define void @use({llvm_type} %val) {{
  ret void
}}
"""
    return code


def generate_baseline_code():
    return f"""
define i32 @main(i32 %n) optnone noinline {{
entry:
  %i = alloca i32
  store i32 0, i32* %i
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, %n
  br i1 %cond, label %body, label %exit

body:
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  ret i32 0
}}
"""


def generate_int_chain_code():
    """A dependent chain of integer adds, one cycle each: the clock."""
    body = "".join(f"  %a{i + 1} = add i64 %a{i}, %step\n" for i in range(unrolled))
    return f"""
define i32 @main(i32 %n) optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca i64
  %step = sext i32 %n to i64
  store i32 0, i32* %i
  store i64 0, i64* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, %n
  br i1 %cond, label %body, label %exit

body:
  %a0 = load i64, i64* %acc
{body}
  store i64 %a{unrolled}, i64* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %final = load i64, i64* %acc
  %r = trunc i64 %final to i32
  ret i32 %r
}}
"""


def create_target_machine(cpu, features):
    return llvm.Target.from_default_triple().create_target_machine(cpu=cpu, features=features)


def target_complaints(cpu, features):
    """What LLVM prints about the CPU and features it does not know."""
    with tempfile.TemporaryFile() as err:
        saved = os.dup(2)
        os.dup2(err.fileno(), 2)
        try:
            create_target_machine(cpu, features)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        err.seek(0)
        return list(dict.fromkeys(line for line in err.read().decode().splitlines() if line.strip()))


def checked_features(cpu, features):
    """The features to compile with. A disabled feature this llvmlite's LLVM
    does not know is one it cannot emit either, so it is dropped; an unknown
    CPU or enabled feature stops the measurement, which would otherwise time
    code for another processor."""
    unknown = {line.split("'")[1] for line in target_complaints(cpu, features) if "recognized feature" in line}
    features = ",".join(f for f in features.split(",") if f and not (f.startswith("-") and f in unknown))
    complaints = target_complaints(cpu, features)
    if complaints:
        v = ".".join(map(str, llvm.llvm_version_info))
        sys.exit(f"microbm_cpu.py: the LLVM {v} of llvmlite cannot compile for target-cpu {cpu}:\n" + "\n".join(complaints))
    return features


def compile_jit(llvm_ir, cpu, features):
    """The engine, which the caller keeps alive, and its main(n)."""
    engine = llvm.create_mcjit_compiler(llvm.parse_assembly(""), create_target_machine(cpu, features))
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    return engine, ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(engine.get_function_address("main"))


class Bench:
    """One JIT-compiled loop, its trip count fixed at `iterations` if given,
    else by trial runs to last `target` seconds."""

    def __init__(self, llvm_ir, args, target=TARGET_S):
        self.engine, self.main = compile_jit(llvm_ir, args.cpu, args.features)
        self.n = args.iterations or 1000
        t = self.run()
        while not args.iterations and t < TRIAL_S:
            self.n *= 10
            t = self.run()
        if not args.iterations:
            self.n = min(2**31 - 1, max(1, int(self.n * target / t)))

    def run(self):
        start = time.perf_counter()
        self.main(self.n)
        return time.perf_counter() - start

    def per_trip(self):
        """Seconds per trip, the best of REPS runs."""
        return min(self.run() for _ in range(REPS)) / self.n


def benchmarks():
    """(row, type, IR) of every row, in the order they print."""
    for precision in precisions:
        for instr, info in OP_INFO.items():
            if info["kind"] == "arithmetic":
                yield instr, precision, generate_arithmetic_op_code(instr, precision)
            else:
                yield instr, precision, generate_compare_op_code(precision)
        for op, src, dst in CASTS:
            if src == precision:
                yield f"{op}_{src}_to_{dst}", src, generate_cast_op_code(op, src, dst)
        for func in FUNC_INFO:
            yield func, precision, generate_function_call_code(func, precision)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu", required=True, help="target-cpu the microbenchmarks are compiled for")
    ap.add_argument("--features", default="", help="target features, as +name,-name")
    ap.add_argument("--device", required=True, help="the CPU's model name, for the header")
    ap.add_argument("--iterations", type=int, default=0, help="loop trips of each run, in place of sizing it by time")
    args = ap.parse_args()
    args.features = checked_features(args.cpu, args.features)

    # The clock is read again before every row, so that a row is counted in
    # the cycles of the frequency it ran at.
    loop = Bench(generate_baseline_code(), args, TARGET_S / 5)
    chain = Bench(generate_int_chain_code(), args, TARGET_S / 5)

    def clock():
        """The loop's own time per trip, and the length of a cycle."""
        base = loop.per_trip()
        cycle = (chain.per_trip() - base) / unrolled
        if cycle <= 0:
            sys.exit("microbm_cpu.py: the integer-add chain measured no time, so there is no cycle to count in")
        return base, cycle

    rows = list(benchmarks())
    print(f"# Generated by microbm_cpu.py on {args.device}")
    print(f"# native_arch={args.cpu}")
    print(f"# scalar_types={','.join(sorted(precisions, reverse=True))}", flush=True)
    for k, (name, precision, code) in enumerate(rows, 1):
        print(f"[microbm_cpu] {k}/{len(rows)} {name},{precision}", file=sys.stderr, flush=True)
        bench = Bench(code, args)
        base, cycle = clock()
        op = (bench.per_trip() - base) / unrolled
        # A positive floor, as the GPU rows keep: an operation the loop hides
        # entirely still has a row.
        print(f"{name},{precision},{max(op / cycle * 2.0, 1e-12):.9g}", flush=True)


if __name__ == "__main__":
    main()
