#!/usr/bin/env python3
"""Measure the host CPU cost model Poseidon prices CPU candidates from.

The rows are of float and double, the formats a precision change offers a
host function, timed in JIT-compiled loops of 128 operations a trip, compiled
as the target machine compiles a program. A row is the loop's time per
operation less its control's, the same loop without the operation, in the
unit of the GPU rows: clock cycles times two, a cycle being the latency of a
dependent integer add timed just before the row. Rows of two kinds:

- An arithmetic operation, comparison, conversion, fused multiply-add, square
  root, minimum, maximum, rounding or sign operation, and powi: its latency.
  It runs on its own result, a = op(a, c), as microbm.cu chains it, where the
  compiler keeps that chain, and else behind an empty asm, as microbm.cu does
  on AMD; a conversion, powi, and an operation neither chain keeps, run on a
  fixed input that a restoring step makes depend on the previous result, the
  restoring steps forming the control.
- A math-library function: its reciprocal throughput, calls on fixed inputs
  of their own whose results are stored, as a GPU row is a reciprocal
  throughput. A latency does not time a call whose result follows a
  predicted branch rather than its input.

Before a loop is timed, its machine code must hold, once for every operation,
an instruction that operation compiles to on its own, or the measurement
stops.

A trial run sizes the trip count so that a timed run lasts TARGET_S seconds,
and the best of REPS timed runs counts. Rows go to stdout as they are
measured, progress to stderr."""

import argparse
import ctypes
import os
import random
import re
import struct
import sys
import tempfile
import time

import llvmlite.binding as llvm


random.seed(42)

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

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


OPS = ["fneg", "fadd", "fsub", "fmul", "fdiv", "fcmp"]

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


def const(f, precision):
    return float_to_llvm_hex(f, precision)


def call(fn, precision, args):
    return f"call {precision} @{fn}.{intrinsic_suffix[precision]}({args})"


# The dependent chain through the operation itself, a = op(a, c), as
# microbm.cu times it: (row, type) -> the step from %x to its result, the i-th
# step of the chain. The constants keep the value finite and normal however
# long the chain runs: a multiplier alternates with its inverse, and the
# fused multiply-add converges to 2.
def chain_step(name, precision):
    T = precision
    eps = 2.0**-23 if T == "float" else 2.0**-52
    up, down = const(1 + eps, T), const(1 - eps, T)
    one, half = const(1.0, T), const(0.5, T)
    steps = {
        "fneg": lambda x, i: f"fneg {T} {x}",
        "fadd": lambda x, i: f"fadd {T} {x}, {one}",
        "fsub": lambda x, i: f"fsub {T} {x}, {half}",
        "fmul": lambda x, i: f"fmul {T} {x}, {up if i % 2 else down}",
        "fdiv": lambda x, i: f"fdiv {T} {x}, {up if i % 2 else down}",
        "fma": lambda x, i: call("llvm.fma", T, f"{T} {x}, {T} {half}, {T} {one}"),
        "fmuladd": lambda x, i: call("llvm.fmuladd", T, f"{T} {x}, {T} {half}, {T} {one}"),
        "sqrt": lambda x, i: call("llvm.sqrt", T, f"{T} {x}"),
        "fabs": lambda x, i: call("llvm.fabs", T, f"{T} {x}"),
        "copysign": lambda x, i: call("llvm.copysign", T, f"{T} {x}, {T} {const(-1.0, T)}"),
        "maxnum": lambda x, i: call("llvm.maxnum", T, f"{T} {x}, {T} {one}"),
        "minnum": lambda x, i: call("llvm.minnum", T, f"{T} {x}, {T} {one}"),
    }
    for fn in ("ceil", "floor", "trunc", "round", "rint"):
        steps[fn] = lambda x, i, fn=fn: call(f"llvm.{fn}", T, f"{T} {x}")
    if name == "fcmp":
        # A regime split's condition and its select, as in microbm.cu.
        return lambda x, i: (
            [f"%c{i} = fcmp une {T} {x}, {one}", f"%s{i} = select i1 %c{i}, {T} {x}, {T} {one}"],
            f"%s{i}",
        )
    if name not in steps:
        return None
    return lambda x, i: ([f"%s{i} = {steps[name](x, i)}"], f"%s{i}")


# The operation on an input of its own, (row, type) -> (input type, result
# type, the operation on %y as the i-th step). Every input is the fixed value
# today's inputs come from; the restoring step puts it in front of each step.
def fixed_input_step(name, precision):
    T = precision
    if name.startswith("fpext_") or name.startswith("fptrunc_"):
        op, rest = name.split("_", 1)
        src, dst = rest.split("_to_")
        return src, dst, lambda y, i: ([f"%s{i} = {op} {src} {y} to {dst}"], f"%s{i}")
    if name == "fneg":
        return T, T, lambda y, i: ([f"%s{i} = fneg {T} {y}"], f"%s{i}")
    if name == "fcmp":
        return T, T, chain_step(name, T)
    if name in OPS:
        return T, T, lambda y, i: ([f"%s{i} = {name} {T} {y}, {const(generate_random_fp(T), T)}"], f"%s{i}")
    info = FUNC_INFO[name]
    # The C library names the float form of a function with an f suffix.
    fn = f"{info['intrinsic']}.{intrinsic_suffix[T]}" if info["intrinsic"] else name + ("f" if T == "float" else "")
    if name == "powi":
        # An exponent of 0 or 1 is no operation at all.
        k = lambda i: random.choice([e for e in range(-10, 11) if e not in (0, 1)])
        return T, T, lambda y, i: ([f"%s{i} = call {T} @{fn}({T} {y}, i32 {k(i)})"], f"%s{i}")

    def rest():
        return "".join(f", {T} {const(generate_random_fp(T), T)}" for _ in range(info["num_operands"] - 1))

    return T, T, lambda y, i: ([f"%s{i} = call {T} @{fn}({T} {y}{rest()})"], f"%s{i}")


def bits(T):
    return {"float": "i32", "double": "i64"}[T]


def restore(x, R, S, i):
    """The step that gives the next operation its fixed input of type S, as a
    function of the previous result %x of type R, which it keeps the input
    dependent on while discarding its value, whatever it is (NaN and infinity
    included): the bits of %x masked by a zero the compiler cannot see, or-ed
    into the bits of the input."""
    lines = [f"%rb{i} = bitcast {R} {x} to {bits(R)}"]
    zero = "%zero32" if R == "float" else "%zero64"
    lines.append(f"%rm{i} = and {bits(R)} %rb{i}, {zero}")
    m = f"%rm{i}"
    if bits(R) != bits(S):
        lines.append(f"%rw{i} = {'trunc' if bits(R) == 'i64' else 'zext'} {bits(R)} {m} to {bits(S)}")
        m = f"%rw{i}"
    [c] = struct.unpack("<I" if S == "float" else "<Q", struct.pack("<f" if S == "float" else "<d", generate_random_fp(S)))
    # The input leaves the integer registers through an addition of +0, which
    # the compiler keeps, so that no integer operation can stand in for the
    # floating-point one it feeds (fneg of a bitcast is an integer xor).
    lines += [f"%ro{i} = or {bits(S)} {m}, {c}", f"%rc{i} = bitcast {bits(S)} %ro{i} to {S}",
              f"%ry{i} = fadd {S} %rc{i}, {const(0.0, S)}"]
    return lines, f"%ry{i}"


class Shape:
    """The steps of one benchmark, of its control (the same loop without the
    operation) and the operation alone, each as step(x, i) -> (IR lines,
    result), with the type the loop carries and the operation's input type."""

    def __init__(self, kind, carried, step, control, op, op_input):
        self.kind, self.carried, self.step, self.control = kind, carried, step, control
        self.op, self.op_input = op, op_input


# The register class of a floating-point value in an inline-asm constraint, by
# the architecture of the target triple.
FP_REGISTER = {"x86_64": "x", "i686": "x", "aarch64": "w"}


def opaque(x, T, i):
    """%x passed through an empty asm: no instruction, and nothing known about
    the value afterwards, as microbm.cu does on AMD."""
    arch = llvm.get_process_triple().split("-")[0]
    if arch not in FP_REGISTER:
        return None
    c = FP_REGISTER[arch]
    return [f'%o{i} = call {T} asm sideeffect "", "={c},0"({T} {x})'], f"%o{i}"


def chain_shape(name, precision, behind_asm=False):
    op = chain_step(name, precision)
    if op is None:
        return None
    if not behind_asm:
        return Shape("chain", precision, op, lambda x, i: ([], x), op, precision)
    if opaque("%x", precision, 0) is None:
        return None

    def step(x, i):
        lines, y = opaque(x, precision, i)
        more, r = op(y, i)
        return lines + more, r

    return Shape("chain behind an empty asm", precision, step, lambda x, i: opaque(x, precision, i), op, precision)


def fixed_input_shape(name, precision):
    S, R, op = fixed_input_step(name, precision)

    def step(x, i):
        lines, y = restore(x, R, S, i)
        more, r = op(y, i)
        return lines + more, r

    # The control restores from the same type it restores to; a move between
    # the vector and the integer registers takes as long at either width.
    return Shape("fixed input", R, step, lambda x, i: restore(x, R, R, i), op, S)


def independent_input(T, i):
    """A fixed input of type T the compiler cannot see, on no previous step:
    the bits of the input or-ed with a zero it cannot see."""
    zero = "%zero32" if T == "float" else "%zero64"
    [c] = struct.unpack("<I" if T == "float" else "<Q", struct.pack("<f" if T == "float" else "<d", generate_random_fp(T)))
    return [f"%io{i} = or {bits(T)} {zero}, {c}", f"%ic{i} = bitcast {bits(T)} %io{i} to {T}",
            f"%iy{i} = fadd {T} %ic{i}, {const(0.0, T)}"], f"%iy{i}"


def independent_shape(name, precision):
    """The calls of a math-library function on inputs of their own, each result
    stored, so that the calls overlap as far as the processor lets them: the
    loop's time is their throughput. A dependent chain does not time a call
    whose result follows from a predicted branch rather than from its input
    (an overflowing exp, a saturated tanh, fdim's zero), which a chain would
    price at nothing."""
    S, R, op = fixed_input_step(name, precision)

    def run(i, call):
        lines, y = independent_input(S, i)
        more, r = (op(y, i) if call else ([], y))
        return lines + more + [f"store volatile {R} {r}, {R}* @sink.{R}"]

    return Shape("independent calls", R, lambda x, i: (run(i, True), x), lambda x, i: (run(i, False), x), op, S)


def body(step, k, x):
    lines = []
    for i in range(k):
        more, x = step(x, i)
        lines += more
    return lines, x


def declarations(text):
    """The declarations of the functions `text` calls."""
    out, seen = ["@zero = global i64 0", "@sink.float = global float 0.0", "@sink.double = global double 0.0"], set()
    for m in re.finditer(r"call (float|double) @([\w.]+)\(([^)]*)\)", text):
        T, fn, args = m.groups()
        if fn not in seen:
            seen.add(fn)
            out.append(f"declare {T} @{fn}({', '.join(a.split()[0] for a in args.split(', '))})")
    return "\n".join(out) + "\n"


def zero_loads():
    return ["%zero64 = load volatile i64, i64* @zero", "%zero32 = trunc i64 %zero64 to i32"]


def loop_code(shape, step):
    """The timed loop: `unrolled` steps a trip, the value carried in memory."""
    T = shape.carried
    lines, x = body(step, unrolled, "%x0")
    text = "\n  ".join(zero_loads() + lines)
    code = f"""
define i32 @main(i32 %n) noinline {{
entry:
  %i = alloca i32
  %acc = alloca {T}
  store i32 0, i32* %i
  store {T} {const(1.5, T)}, {T}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, %n
  br i1 %cond, label %body, label %exit

body:
  %x0 = load {T}, {T}* %acc
  {text}
  store {T} {x}, {T}* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  ret i32 0
}}
"""
    return declarations(code) + code


def probe_code(T, R, steps):
    """The IR lines of each of `steps` on an argument of type T of its own,
    their results of type R stored through %p, each store on its own so that
    no two are merged into one vector store."""
    args = ", ".join(f"{T} %a{j}" for j in range(len(steps)))
    lines = []
    for j, (more, r) in enumerate(steps):
        lines += more + [f"%p{j} = getelementptr {R}, {R}* %p, i64 {j}", f"store volatile {R} {r}, {R}* %p{j}"]
    text = "\n  ".join(lines)
    code = f"define void @probe({args}{', ' if args else ''}{R}* %p) noinline {{\n  {text}\n  ret void\n}}\n"
    return declarations(code) + code


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


def instructions(llvm_ir, tm):
    """How often each mnemonic occurs in the machine code `tm` emits for
    `llvm_ir`, the code the engine runs."""
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    count = {}
    for line in tm.emit_assembly(mod).splitlines():
        line = line.strip()
        if line and not line.startswith((".", "#", "//", ";")) and not line.endswith(":"):
            # Without digits, the operand-order variants of one instruction
            # (vfmadd132ss, vfmadd213ss) count as one.
            m = re.sub(r"\d", "", line.split()[0])
            count[m] = count.get(m, 0) + 1
    return count


def minus(a, b):
    return {m: n - b.get(m, 0) for m, n in a.items() if n > b.get(m, 0)}


def guard(shape, tm, bench, control):
    """None if the loop of `shape` executes every operation it times, else why
    not. The code of an operation is what a second copy of it, on an argument
    of its own, adds to a function holding the first, beyond what a second
    argument stored alike adds. The loop, beyond its control, must hold some
    instruction of that code at least once for every operation; the others may
    be constants the loop sets up once. An operation that folds away, alone or
    in its chain, leaves fewer."""
    T, R = shape.op_input, shape.carried
    control = instructions(control, tm)
    stores = minus(instructions(probe_code(R, R, [([], "%a0"), ([], "%a1")]), tm),
                   instructions(probe_code(R, R, [([], "%a0")]), tm))
    # Operations that differ in no more than their floating-point constants
    # compile alike. A copy takes the index two on, the same parity.
    code, kinds = {}, []
    for i in range(unrolled):
        lines, _ = shape.op("%x", i)
        k = re.sub(r"0x[0-9A-F]+", "C", re.sub(r"%(\w+?)\d+\b", r"%\1", "\n".join(lines)))
        if k not in code:
            one = [shape.op("%a0", i)]
            two = one + [shape.op("%a1", i + 2)]
            code[k] = minus(minus(instructions(probe_code(T, R, two), tm), instructions(probe_code(T, R, one), tm)), stores)
            if not code[k]:
                return f"{lines[-1].split('=', 1)[1].strip()} compiles to no instruction of its own"
        kinds.append(k)
    got = minus(instructions(bench, tm), control)
    for k in dict.fromkeys(kinds):
        n = kinds.count(k)
        if not any(got.get(m, 0) >= n for m in code[k]):
            return f"the loop holds none of {', '.join(sorted(code[k]))} once per operation: " + ", ".join(
                f"{m} {got.get(m, 0)} of {n}" for m in sorted(code[k]))
    return None


def shape_of(name, precision, tm, only_chain):
    """The shape a row is timed in, and the IR of its loop and of its control.
    A math-library function is timed in independent calls; any other
    operation in the chain through itself if the compiler keeps it, else that
    chain behind an empty asm, else on fixed inputs."""
    tried = []
    if name in FUNC_INFO and chain_step(name, precision) is None and name != "powi":
        shapes = [lambda: independent_shape(name, precision)]
    else:
        shapes = [lambda: chain_shape(name, precision)]
        if not only_chain:
            shapes += [lambda: chain_shape(name, precision, True), lambda: fixed_input_shape(name, precision)]
    for make in shapes:
        shape = make()
        if shape is None:
            continue
        bench, control = loop_code(shape, shape.step), loop_code(shape, shape.control)
        why = guard(shape, tm, bench, control)
        if why is None:
            return shape, bench, control
        tried.append(f"{shape.kind}: {why}")
    sys.exit(f"microbm_cpu.py: {name},{precision} cannot be measured on this target, its operation is not in the code: "
             + ("; ".join(tried) or f"no shape for {llvm.get_process_triple()}"))


def benchmarks():
    """(row, type) of every row, in the order they print."""
    for precision in precisions:
        for op in OPS:
            yield op, precision
        for op, src, dst in CASTS:
            if src == precision:
                yield f"{op}_{src}_to_{dst}", src
        for func in FUNC_INFO:
            yield func, precision


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", required=True, help="target-cpu the microbenchmarks are compiled for")
    ap.add_argument("--features", default="", help="target features, as +name,-name")
    ap.add_argument("--device", required=True, help="the CPU's model name, for the header")
    ap.add_argument("--iterations", type=int, default=0, help="loop trips of each run, in place of sizing it by time")
    ap.add_argument("--chain-only", action="store_true", help="time every operation in the chain through itself, refusing one that folds")
    args = ap.parse_args()
    args.features = checked_features(args.cpu, args.features)
    tm = create_target_machine(args.cpu, args.features)

    # The clock is read again before every row, so that a row is counted in
    # the cycles of the frequency it ran at.
    loop = Bench(generate_baseline_code(), args, TARGET_S / 5)
    chain = Bench(generate_int_chain_code(), args, TARGET_S / 5)

    def cycle():
        c = (chain.per_trip() - loop.per_trip()) / unrolled
        if c <= 0:
            sys.exit("microbm_cpu.py: the integer-add chain measured no time, so there is no cycle to count in")
        return c

    rows = list(benchmarks())
    print(f"# Generated by microbm_cpu.py on {args.device}")
    print(f"# native_arch={args.cpu}")
    print(f"# scalar_types={','.join(sorted(precisions, reverse=True))}", flush=True)
    for k, (name, precision) in enumerate(rows, 1):
        shape, bench_ir, control_ir = shape_of(name, precision, tm, args.chain_only)
        print(f"[microbm_cpu] {k}/{len(rows)} {name},{precision} ({shape.kind})", file=sys.stderr, flush=True)
        bench = Bench(bench_ir, args)
        control = Bench(control_ir, args, TARGET_S / 5)
        op = (bench.per_trip() - control.per_trip()) / unrolled
        # A positive floor, as the GPU rows keep.
        print(f"{name},{precision},{max(op / cycle() * 2.0, 1e-12):.9g}", flush=True)


if __name__ == "__main__":
    main()
