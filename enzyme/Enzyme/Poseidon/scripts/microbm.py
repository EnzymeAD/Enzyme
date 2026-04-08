import time
import csv
import struct
import random
import numpy as np
import llvmlite.binding as llvm
import ctypes


random.seed(42)

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

FAST_MATH_FLAG = "fast"

unrolled = 128
iterations = 100000000
AMPLIFIER = 10

precision_to_llvm_type = {
    "double": "double",
    "float": "float",
    "half": "half",
    "fp80": "x86_fp80",
    "fp128": "fp128",
    "bf16": "bfloat",
}

precision_to_intrinsic_suffix = {
    "double": "f64",
    "float": "f32",
    "half": "f16",
    "fp80": "f80",
    "fp128": "f128",
    "bf16": "bf16",
}

precision_ranks = {"bf16": 0, "half": 1, "float": 2, "double": 3, "fp80": 4, "fp128": 5}
precisions_ordered = ["bf16", "half", "float", "double", "fp80", "fp128"]
precisions = ["float", "double"]


def get_zero_literal(precision):
    if precision in ("double", "float", "half"):
        return "0.0"
    elif precision == "bf16":
        return "0xR0000"
    elif precision == "fp80":
        return "0xK00000000000000000000"
    elif precision == "fp128":
        return "0xL00000000000000000000000000000000"
    return "0.0"


def float64_to_fp80_bytes(value: np.float64) -> bytes:
    packed = struct.pack(">d", value)
    (bits,) = struct.unpack(">Q", packed)
    sign = (bits >> 63) & 0x1
    exponent = (bits >> 52) & 0x7FF
    mantissa = bits & 0xFFFFFFFFFFFFF

    if exponent == 0:
        if mantissa == 0:
            fp80_exponent = 0
            fp80_mantissa = 0
        else:
            shift = 0
            while (mantissa & (1 << 52)) == 0:
                mantissa <<= 1
                shift += 1
            exponent = 1 - shift
            exponent_bias_64 = 1023
            exponent_bias_80 = 16383
            fp80_exponent = exponent - exponent_bias_64 + exponent_bias_80
            fp80_mantissa = mantissa << (63 - 52)
    elif exponent == 0x7FF:
        fp80_exponent = 0x7FFF
        if mantissa == 0:
            fp80_mantissa = 0x8000000000000000
        else:
            fp80_mantissa = 0xC000000000000000 | (mantissa << (63 - 52))
    else:
        exponent_bias_64 = 1023
        exponent_bias_80 = 16383
        fp80_exponent = exponent - exponent_bias_64 + exponent_bias_80
        fp80_mantissa = (0x8000000000000000) | (mantissa << (63 - 52))

    exponent_sign = (sign << 15) | fp80_exponent
    fp80_bits = (exponent_sign << 64) | fp80_mantissa
    fp80_bytes = fp80_bits.to_bytes(10, byteorder="big")
    return fp80_bytes


def float64_to_fp128_bytes(value: np.float64) -> bytes:
    packed = struct.pack(">d", value)
    (bits,) = struct.unpack(">Q", packed)
    sign = (bits >> 63) & 0x1
    exponent = (bits >> 52) & 0x7FF
    mantissa = bits & 0xFFFFFFFFFFFFF

    if exponent == 0:
        fp128_exponent = 0
    elif exponent == 0x7FF:
        fp128_exponent = 0x7FFF
    else:
        exponent_bias_64 = 1023
        exponent_bias_128 = 16383
        fp128_exponent = exponent - exponent_bias_64 + exponent_bias_128

    fp128_mantissa = mantissa << 60
    fp128_bits = (sign << 127) | (fp128_exponent << 112) | fp128_mantissa
    fp128_bytes = fp128_bits.to_bytes(16, byteorder="big")
    return fp128_bytes


def float_to_llvm_hex(f, precision):
    if precision == "double":
        f_cast = np.float64(f)
        packed = struct.pack(">d", f_cast)
        [i] = struct.unpack(">Q", packed)
        return f"0x{i:016X}"
    elif precision == "float":
        f_cast = np.float32(f)
        packed = struct.pack(">d", f_cast)
        [i] = struct.unpack(">Q", packed)
        return f"0x{i:016X}"
    elif precision == "half":
        f_cast = np.float16(f)
        packed = f_cast.tobytes()
        [i] = struct.unpack(">H", packed)
        return f"0xH{i:04X}"
    elif precision == "bf16":
        f_cast = np.float32(f)
        [bits] = struct.unpack(">I", struct.pack(">f", f_cast))
        bf16_bits = bits >> 16
        return f"0xR{bf16_bits:04X}"
    elif precision == "fp80":
        f_cast = np.float64(f)
        fp80_bytes = float64_to_fp80_bytes(f_cast)
        return f"0xK{fp80_bytes.hex().upper()}"
    elif precision == "fp128":
        f_cast = np.float64(f)
        fp128_bytes = float64_to_fp128_bytes(f_cast)
        swapped = fp128_bytes[8:] + fp128_bytes[:8]
        return f"0xL{swapped.hex().upper()}"
    else:
        return str(f)


def generate_random_fp(precision):
    if precision == "double":
        f = random.uniform(-1e10, 1e10)
        dtype = np.float64
    elif precision == "float":
        f = random.uniform(-1e5, 1e5)
        dtype = np.float32
    elif precision == "half":
        f = random.uniform(-1e3, 1e3)
        dtype = np.float16
    elif precision == "bf16":
        f = random.uniform(-1e3, 1e3)
        dtype = np.float32
    elif precision == "fp80":
        f = random.uniform(-1e10, 1e10)
        dtype = np.float64
    elif precision == "fp128":
        f = random.uniform(-1e10, 1e10)
        dtype = np.float64
    else:
        f = random.uniform(-1e3, 1e3)
        dtype = np.float64
    return dtype(f).item()


OP_INFO = {
    "fneg": {"llvm_instr": "fneg", "num_operands": 1, "kind": "arithmetic"},
    "fadd": {"llvm_instr": "fadd", "num_operands": 2, "kind": "arithmetic"},
    "fsub": {"llvm_instr": "fsub", "num_operands": 2, "kind": "arithmetic"},
    "fmul": {"llvm_instr": "fmul", "num_operands": 2, "kind": "arithmetic"},
    "fdiv": {"llvm_instr": "fdiv", "num_operands": 2, "kind": "arithmetic"},
    "fcmp": {"llvm_instr": "fcmp", "num_operands": 2, "kind": "compare"},
    "fptrunc": {"llvm_instr": "fptrunc", "num_operands": 1, "kind": "cast"},
    "fpext": {"llvm_instr": "fpext", "num_operands": 1, "kind": "cast"},
}

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


def generate_loop_code(llvm_type, iterations, body_instructions, final_acc_reg):
    zero_literal = get_zero_literal(llvm_type)
    code = f"""
define i32 @main() optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca {llvm_type}
  store i32 0, i32* %i
  store {llvm_type} {zero_literal}, {llvm_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
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


def generate_arithmetic_op_code(op_key, precision, iterations):
    """Generate LLVM IR for a basic arithmetic operator (or fneg) based on OP_INFO."""
    op_info = OP_INFO[op_key]
    llvm_type = precision_to_llvm_type[precision]
    body_lines = ""
    for idx in range(unrolled):
        operands = []
        for _ in range(op_info["num_operands"]):
            f_val = generate_random_fp(precision)
            operands.append(float_to_llvm_hex(f_val, precision))

        if op_info["num_operands"] == 1:
            line = f"  %result{idx} = {op_info['llvm_instr']} {FAST_MATH_FLAG} {llvm_type} {operands[0]}"
            body_lines += line + "\n"
            body_lines += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {llvm_type} %acc_val{idx}, %result{idx}\n"
        elif op_info["num_operands"] == 2:
            line = f"  %result{idx} = {op_info['llvm_instr']} {FAST_MATH_FLAG} {llvm_type} {operands[0]}, {operands[1]}"
            body_lines += line + "\n"
            body_lines += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {llvm_type} %acc_val{idx}, %result{idx}\n"

    final_acc = f"%acc_val{unrolled}"
    return generate_loop_code(llvm_type, iterations, body_lines, final_acc)


def generate_compare_op_code(precision, iterations):
    """Generate LLVM IR for an fcmp (comparison) operation."""
    llvm_type = precision_to_llvm_type[precision]

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
define i32 @main() optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca i32
  store i32 0, i32* %i
  store i32 0, i32* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
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


def generate_cast_op_code(op_key, src_precision, dst_precision, iterations):
    """Generate LLVM IR for a cast operation (fptrunc or fpext)."""
    op_info = OP_INFO[op_key]
    src_type = precision_to_llvm_type[src_precision]
    dst_type = precision_to_llvm_type[dst_precision]
    zero_literal = get_zero_literal(dst_precision)

    body_lines = ""
    for idx in range(unrolled):
        f_val = generate_random_fp(src_precision)
        hex_val = float_to_llvm_hex(f_val, src_precision)
        line = f"  %result{idx} = {op_info['llvm_instr']} {src_type} {hex_val} to {dst_type}"
        body_lines += line + "\n"
        body_lines += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {dst_type} %acc_val{idx}, %result{idx}\n"

    final_acc = f"%acc_val{unrolled}"
    code = f"""
define i32 @main() optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca {dst_type}
  store i32 0, i32* %i
  store {dst_type} {zero_literal}, {dst_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  %acc_val0 = load {dst_type}, {dst_type}* %acc
{body_lines}
  store {dst_type} {final_acc}, {dst_type}* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %final_acc = load {dst_type}, {dst_type}* %acc
  call void @use({dst_type} %final_acc)
  ret i32 0
}}

define void @use({dst_type} %val) {{
  ret void
}}
"""
    return code


def generate_function_call_code(func_name, precision, iterations):
    """Generate LLVM IR for a function call based on FUNC_INFO."""

    func_info = FUNC_INFO[func_name]
    llvm_type = precision_to_llvm_type[precision]
    intrinsic_suffix = precision_to_intrinsic_suffix.get(precision, "")

    if func_info["intrinsic"]:
        fn = f"{func_info['intrinsic']}.{intrinsic_suffix}"
    else:
        fn = func_name

    num_operands = func_info["num_operands"]

    body_lines = "  %acc_val0 = load " + llvm_type + ", " + llvm_type + "* %acc\n"

    for idx in range(unrolled):
        if func_name == "powi":
            f_val = generate_random_fp(precision)
            i_val = random.randint(-10, 10)
            f_hex = float_to_llvm_hex(f_val, precision)
            line = f"  %result{idx} = call {FAST_MATH_FLAG} {llvm_type} @{fn}(" f"{llvm_type} {f_hex}, i32 {i_val})"
            body_lines += line + "\n"
            body_lines += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {llvm_type} %acc_val{idx}, %result{idx}\n"
        else:
            operands = []
            for _ in range(num_operands):
                f_val = generate_random_fp(precision)
                operands.append(float_to_llvm_hex(f_val, precision))

            if num_operands == 1:
                call_str = f"call {FAST_MATH_FLAG} {llvm_type} @{fn}({llvm_type} {operands[0]})"
            elif num_operands == 2:
                call_str = (
                    f"call {FAST_MATH_FLAG} {llvm_type} @{fn}({llvm_type} {operands[0]}, {llvm_type} {operands[1]})"
                )
            elif num_operands == 3:
                call_str = f"call {FAST_MATH_FLAG} {llvm_type} @{fn}({llvm_type} {operands[0]}, {llvm_type} {operands[1]}, {llvm_type} {operands[2]})"
            else:
                call_str = ""

            body_lines += f"  %result{idx} = {call_str}\n"
            body_lines += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {llvm_type} %acc_val{idx}, %result{idx}\n"

    if func_name == "powi":
        decl = f"declare {llvm_type} @{fn}({llvm_type}, i32)"
    else:
        arg_types = ", ".join([llvm_type] * num_operands)
        decl = f"declare {llvm_type} @{fn}({arg_types})"

    code = f"""
{decl}
define i32 @main() optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca {llvm_type}
  store i32 0, i32* %i
  store {llvm_type} {get_zero_literal(precision)}, {llvm_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
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


def generate_baseline_code(iterations):
    return f"""
define i32 @main() optnone noinline {{
entry:
  %i = alloca i32
  store i32 0, i32* %i
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  ret i32 0
}}
"""


def create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(mod, target_machine)
    return engine


def run_llvm_ir_jit(llvm_ir):
    engine = create_execution_engine()
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()

    func_ptr = engine.get_function_address("main")
    cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(func_ptr)

    start = time.perf_counter()
    retval = cfunc()
    end = time.perf_counter()

    return (end - start), retval


if __name__ == "__main__":
    csv_file = "results.csv"
    with open(csv_file, "w", newline="") as csvfile:
        fieldnames = ["instruction", "precision", "cost"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        llvm_code = generate_baseline_code(iterations)
        baseline_time, _ = run_llvm_ir_jit(llvm_code)

        for precision in precisions:
            for instr in OP_INFO:
                op_kind = OP_INFO[instr]["kind"]
                if op_kind == "cast":
                    src_precision = precision
                    src_rank = precision_ranks.get(src_precision)
                    if src_rank is None:
                        continue

                    if instr == "fptrunc":
                        dst_precisions = [
                            p for p in precisions_ordered if p in precisions and precision_ranks[p] < src_rank
                        ]
                    else:
                        dst_precisions = [
                            p for p in precisions_ordered if p in precisions and precision_ranks[p] > src_rank
                        ]

                    for dst_precision in dst_precisions:
                        if (src_precision, dst_precision) in [
                            ("half", "bf16"),
                            ("bf16", "half"),
                        ]:
                            continue
                        code = generate_cast_op_code(instr, src_precision, dst_precision, iterations)
                        name = f"{instr}_{src_precision}_to_{dst_precision}"
                        elapsed, _ = run_llvm_ir_jit(code)
                        adjusted = (elapsed - baseline_time) * AMPLIFIER
                        writer.writerow(
                            {
                                "instruction": name,
                                "precision": src_precision,
                                "cost": int(adjusted),
                            }
                        )
                else:
                    if op_kind == "arithmetic":
                        code = generate_arithmetic_op_code(instr, precision, iterations)
                    elif op_kind == "compare":
                        code = generate_compare_op_code(precision, iterations)
                    else:
                        code = ""
                    if code.strip():
                        elapsed, _ = run_llvm_ir_jit(code)
                        adjusted = (elapsed - baseline_time) * AMPLIFIER
                        writer.writerow(
                            {
                                "instruction": instr,
                                "precision": precision,
                                "cost": int(adjusted),
                            }
                        )

            for func in FUNC_INFO:
                code = generate_function_call_code(func, precision, iterations)
                if code.strip():
                    elapsed, _ = run_llvm_ir_jit(code)
                    adjusted = (elapsed - baseline_time) * AMPLIFIER
                    writer.writerow(
                        {
                            "instruction": func,
                            "precision": precision,
                            "cost": int(adjusted),
                        }
                    )

    print(f"Results in '{csv_file}'. Baseline: {baseline_time:.6f}s")
