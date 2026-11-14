#!/usr/bin/env python3
"""make_cost_models.py <out dir> [<gpu arch>]

Writes the cost models the Poseidon tests price from. Every price is made up:
none is a measurement of any device, and none predicts one. The prices are
chosen only so that the orderings the tests exercise hold.

  cm_cpu_native.csv       x86-64: 1 per float operation, 2 per float math
                          library call, 4 times that in double, 1 per cast
  cm_cpu_foreign.csv      the same, filed under sm_120
  cm_cpu_noarch.csv       the same, with no native_arch header
  cm_cpu_missing.csv      the same, without the fmul,double row
  cm_gfx942_synthetic.csv 1 per float and 64 per double operation
  cm_sm_120_synthetic.csv the GPU prices of gpu_rows(), with three in-kernel
                          tensor-core classes
  cm_sm_120_synthetic_dispatch.csv  and the host-dispatched GEMMs but the
                                    error-corrected one, which a build
                                    without cuMpSGEMM does not offer
  cm_sm_120_synthetic_tcec.csv      and that one too
  cm_sm_120_synthetic_half.csv      and half at half the float price
  cm_sm_90_synthetic.csv  cm_sm_120_synthetic.csv filed under sm_90
  cm_gpu_synthetic.csv    with <gpu arch>: every family a calibrated NVIDIA
                          model has, filed under that architecture
"""
import os
import sys

CPU_OPS = [
    ("fadd", 1), ("fsub", 1), ("fmul", 1), ("fdiv", 1), ("fneg", 1),
    ("fabs", 1), ("fcmp", 1), ("fma", 1), ("fmuladd", 1), ("sqrt", 1),
    ("maxnum", 1), ("minnum", 1), ("ceil", 1), ("floor", 1), ("exp2", 1),
    ("log10", 1), ("log2", 1), ("rint", 1), ("round", 1), ("trunc", 1),
    ("copysign", 1), ("sin", 2), ("cos", 2), ("tan", 2), ("asin", 2),
    ("acos", 2), ("atan", 2), ("atan2", 2), ("sinh", 2), ("cosh", 2),
    ("tanh", 2), ("asinh", 2), ("acosh", 2), ("atanh", 2), ("exp", 2),
    ("log", 2), ("cbrt", 2), ("pow", 2), ("powi", 2), ("hypot", 2),
    ("expm1", 2), ("log1p", 2), ("erf", 2), ("lgamma", 2), ("tgamma", 2),
    ("fdim", 2), ("fmod", 2), ("remainder", 2),
]

# GPU float prices in float adds. Double costs 512 times as much on the FP64
# pipe, so that a double-single expansion is cheaper than FP64 even on a
# test-sized body; 2 times for what only touches the sign or rounds to an
# integer; 128 times otherwise.
GPU_OPS = [
    ("fadd", 1), ("fsub", 1), ("fmul", 1), ("fdiv", 8), ("fneg", 1),
    ("fabs", 1), ("fcmp", 1), ("fma", 1), ("fmuladd", 1), ("sqrt", 8),
    ("rsqrt", 8), ("cbrt", 16), ("exp", 8), ("expm1", 16), ("exp2", 8),
    ("log", 16), ("log1p", 16), ("log2", 16), ("log10", 16), ("sin", 16),
    ("cos", 16), ("tan", 16), ("asin", 16), ("acos", 16), ("atan", 16),
    ("sinh", 16), ("cosh", 16), ("tanh", 16), ("asinh", 16), ("acosh", 16),
    ("atanh", 16), ("erf", 16), ("lgamma", 128), ("tgamma", 64), ("ceil", 1),
    ("floor", 1), ("trunc", 1), ("round", 8), ("rint", 8), ("atan2", 32),
    ("pow", 64), ("powi", 64), ("hypot", 16), ("fmod", 8), ("remainder", 16),
    ("fdim", 2), ("maxnum", 1), ("minnum", 1), ("copysign", 1),
]
FP64_PIPE = {"fadd", "fsub", "fmul", "fdiv", "fma", "fmuladd", "sqrt"}
SIGN_OR_INTEGRAL = {"fneg", "fabs", "copysign", "maxnum", "minnum", "ceil",
                    "floor", "trunc"}

# Tensor-core classes, as k/256 of the scalar FP64 GEMM's wall clock, in the
# order the classes take on an NVIDIA GPU with tensor cores. A tile row only
# says the device has the tile.
TILES = ["wmma_mma_m16n16k16,f16_f32", "wmma_mma_m16n16k16,s8_s32",
         "wmma_mma_m16n16k16,f16_f16", "wmma_mma_m16n16k16,bf16_f32",
         "wmma_mma_m16n16k8,tf32_f32"]
INKERNEL = [("f16_f16", 15), ("f16_f32", 16), ("bf16_f32", 17),
            ("tcec_n2_bf16_f32", 19), ("tcec_n2_f16_f32", 20),
            ("tcec_n3_f16_f32", 24), ("tf32_f32", 27),
            ("tcec_n2_tf32_f32", 44), ("tcec_n5_f16_f32", 66),
            ("tcec_n3_tf32_f32", 80), ("tcec_n5_tf32_f32", 182)]
INKERNEL_LIT = {"f16_f32", "tcec_n2_f16_f32", "tcec_n3_f16_f32"}
DISPATCH = [("direct_dispatch_rel,f16_f32", 2),
            ("direct_dispatch_rel,bf16_f32", 3),
            ("direct_dispatch_rel,tf32_f32", 4)] + \
           [("ozaki_dispatch_rel,nm%d" % k, k) for k in range(8, 15)] + \
           [("ozaki_dispatch_rel,dgemm", 128)]
TCEC_DISPATCH = [("tcec_dispatch_rel,fp16tcec", 7)]


def num(v):
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def cpu_rows(missing=()):
    rows = []
    for op, price in CPU_OPS:
        for ty, scale in (("double", 4), ("float", 1)):
            if (op, ty) not in missing:
                rows.append("%s,%s,%d" % (op, ty, price * scale))
    return rows + ["fptrunc_double_to_float,double,1",
                   "fpext_float_to_double,float,1"]


def gfx_rows():
    ops = [op for op, _ in GPU_OPS]
    return (["%s,float,1" % op for op in ops] +
            ["%s,double,64" % op for op in ops] +
            ["fpext_float_to_double,float,1",
             "fptrunc_double_to_float,double,1"])


def gpu_rows(inkernel=INKERNEL_LIT, dispatch=False, tcec=False, half=False):
    rows = ["%s,float,%d" % (op, p) for op, p in GPU_OPS]
    for op, p in GPU_OPS:
        scale = 512 if op in FP64_PIPE else 2 if op in SIGN_OR_INTEGRAL else 128
        rows.append("%s,double,%d" % (op, p * scale))
    rows += ["fpext_float_to_double,float,64",
             "fptrunc_double_to_float,double,64"]
    if half:
        rows += ["%s,half,%s" % (op, num(p / 2)) for op, p in GPU_OPS]
        rows += ["fptrunc_double_to_half,double,64",
                 "fptrunc_float_to_half,float,64",
                 "fpext_half_to_double,half,64", "fpext_half_to_float,half,64"]
    rows += ["%s,64" % t for t in TILES]
    rows += ["wmma_inkernel_rel,%s,%s" % (c, num(k / 256))
             for c, k in INKERNEL if c in inkernel]
    if dispatch:
        rows += ["%s,%s" % (r, num(k / 256)) for r, k in DISPATCH]
    if tcec:
        rows += ["%s,%s" % (r, num(k / 256)) for r, k in TCEC_DISPATCH]
    return rows


def write(out, name, headers, rows):
    with open(os.path.join(out, name), "w") as f:
        f.write("".join(h + "\n" for h in headers))
        f.write("".join(r + "\n" for r in rows))


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    cpu = "# Lit fixture with unit prices"
    types = "# scalar_types=double,float"
    write(out, "cm_cpu_native.csv",
          [cpu + ", not a device measurement; native_arch matches the test "
           "function target-cpu.", "# native_arch=x86-64", types], cpu_rows())
    write(out, "cm_cpu_foreign.csv",
          [cpu + ", not a device measurement; native_arch names a different "
           "device than the test function target-cpu.", "# native_arch=sm_120",
           types], cpu_rows())
    write(out, "cm_cpu_noarch.csv",
          [cpu + "; the native_arch header is deliberately absent.", types],
          cpu_rows())
    write(out, "cm_cpu_missing.csv",
          [cpu + "; the fmul,double row is deliberately absent.",
           "# native_arch=x86-64", types],
          cpu_rows(missing={("fmul", "double")}))
    write(out, "cm_gfx942_synthetic.csv",
          ["# Lit fixture, synthetic: prices measured on no device, filed "
           "under gfx942 so",
           "# that a gfx942 compile has a cost model to price from; every "
           "operation costs 1",
           "# in float and 64 in double. Never a cost model for a real run.",
           "# native_arch=gfx942", types], gfx_rows())
    gpu = "# Lit fixture, synthetic: prices measured on no device."
    write(out, "cm_sm_120_synthetic.csv",
          [gpu, "# native_arch=sm_120", types], gpu_rows())
    write(out, "cm_sm_120_synthetic_dispatch.csv",
          [gpu, "# native_arch=sm_120", types], gpu_rows(dispatch=True))
    write(out, "cm_sm_120_synthetic_tcec.csv",
          [gpu, "# native_arch=sm_120", types],
          gpu_rows(dispatch=True, tcec=True))
    write(out, "cm_sm_120_synthetic_half.csv",
          [gpu, "# native_arch=sm_120", "# scalar_types=double,float,half"],
          gpu_rows(half=True))
    write(out, "cm_sm_90_synthetic.csv",
          [gpu, "# native_arch=sm_90", types], gpu_rows())
    if len(sys.argv) > 2:
        write(out, "cm_gpu_synthetic.csv",
              [gpu, "# native_arch=" + sys.argv[2], types],
              gpu_rows(inkernel={c for c, _ in INKERNEL}, dispatch=True,
                       tcec=True))


if __name__ == "__main__":
    main()
