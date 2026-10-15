#!/usr/bin/env python3
"""
Execution test: verify that BINOM (binomial/Revolve) and WONLINE
(online/Stumm-Walther) checkpointing schemes produce correct numerical
gradients for f(x, n) = cos(x*x) iterated n times.

The test pipeline for each scheme:
  1. enzymemlir-opt  — differentiate + lower Enzyme-specific passes
  2. mlir-opt        — lower standard MLIR dialects to LLVM dialect
  3. mlir-runner     — JIT-execute and print the gradient

Expected gradient (chain rule: ∏ -sin(sᵢ²)·2sᵢ, s₀=1.2):
  n=1: -2.3795    n=2: 0.010558   n=3: -0.017762   n=4: 0.005531

Usage:
  python3 exec_gradient_schemes.py [enzymemlir-opt] [mlir-opt] [mlir-runner] \\
                                   [libmlir_runner_utils] [libmlir_c_runner_utils]

All five arguments are optional; they default to the paths used during
development.  The test exits 0 on success, non-zero on any failure.
"""

import math
import os
import subprocess
import sys
import tempfile

# ── Tool paths ─────────────────────────────────────────────────────────────────
ENZYME_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# ENZYME_ROOT = .../soft/Enzyme/enzyme; LLVM lives at .../soft/llvm-project/
LLVM_BUILD = os.path.join(os.path.dirname(os.path.dirname(ENZYME_ROOT)),
                          "llvm-project", "mybuild1")

defaults = {
    "eopt":         os.path.join(ENZYME_ROOT, "mybuild1/Enzyme/MLIR/enzymemlir-opt"),
    "mlir_opt":     os.path.join(LLVM_BUILD, "bin/mlir-opt"),
    "mlir_runner":  os.path.join(LLVM_BUILD, "bin/mlir-runner"),
    "runner_utils": os.path.join(LLVM_BUILD,
        "lib/libmlir_runner_utils.24.0git.dylib"),
    "c_utils":      os.path.join(LLVM_BUILD,
        "lib/libmlir_c_runner_utils.dylib"),
}

args = sys.argv[1:]
EOPT        = args[0] if len(args) > 0 else defaults["eopt"]
MLIR_OPT    = args[1] if len(args) > 1 else defaults["mlir_opt"]
MLIR_RUNNER = args[2] if len(args) > 2 else defaults["mlir_runner"]
RUNNER_UTILS= args[3] if len(args) > 3 else defaults["runner_utils"]
C_UTILS     = args[4] if len(args) > 4 else defaults["c_utils"]

# ── MLIR source ────────────────────────────────────────────────────────────────
def make_mlir_src(scheme, body, ret_expr="%r"):
    return (
        "module {\n"
        f"  func.func @f_{scheme}(%arg0: f32, %n: index) -> f32 {{\n"
        "    %lb = arith.constant 0 : index\n"
        "    %st = arith.constant 1 : index\n"
        + body + "\n"
        f"    return {ret_expr} : f32\n"
        "  }\n"
        "}\n"
    )

BINOM_BODY = """\
    %r  = scf.for %iv = %lb to %n step %st iter_args(%s = %arg0) -> f32 {
      %sq  = arith.mulf %s, %s : f32
      %cos = math.cos  %sq     : f32
      scf.yield %cos : f32
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing,
       enzyme.checkpoint_period = 3 : i64}"""

WONLINE_BODY = """\
    %r:2 = "scf.while"(%arg0, %lb) ({
    ^bb0(%s: f32, %k: index):
      %cond = arith.cmpi slt, %k, %n : index
      scf.condition(%cond) %s, %k : f32, index
    }, {
    ^bb0(%s_in: f32, %k_in: index):
      %sq  = arith.mulf %s_in, %s_in : f32
      %cos = math.cos  %sq           : f32
      %k1  = arith.addi %k_in, %st   : index
      scf.yield %cos, %k1 : f32, index
    }) {enzyme.enable_checkpointing = true,
        enzyme.online_checkpointing,
        enzyme.checkpoint_period = 10 : i64} : (f32, index) -> (f32, index)"""

WONLINE_RET = "%r#0"


def expected_grad(x0, n):
    """Analytically compute df/dx0 via chain rule."""
    s, g = x0, 1.0
    for _ in range(n):
        sq = s * s
        g = g * (-math.sin(sq)) * 2 * s
        s = math.cos(sq)
    return g


def run_gradient(scheme, n, grad_fn, enzyme_passes):
    """Differentiate, lower, execute; return the printed float."""
    if scheme == "binom":
        body, ret = BINOM_BODY, "%r"
    else:
        body, ret = WONLINE_BODY, WONLINE_RET
    src = make_mlir_src(scheme, body, ret)

    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(src)
        src_path = f.name

    main_suffix = f"""\
  func.func @main() {{
    %x    = arith.constant 1.200000e+00 : f32
    %n    = arith.constant {n} : index
    %dret = arith.constant 1.0 : f32
    %grad = call @{grad_fn}(%x, %n, %dret) : (f32, index, f32) -> f32
    %v    = vector.broadcast %grad : f32 to vector<1xf32>
    vector.print %v : vector<1xf32>
    func.return
  }}
}}"""

    try:
        # Step 1: enzyme differentiation
        p1 = subprocess.run(
            [EOPT, src_path,
             f"--enzyme-wrap=infn=f_{scheme} outfn={grad_fn}"
             f" argTys=enzyme_active,enzyme_const retTys=enzyme_active"
             f" mode=ReverseModeCombined"]
            + enzyme_passes,
            capture_output=True, text=True,
        )
        if p1.returncode != 0:
            raise RuntimeError(f"enzymemlir-opt failed:\n{p1.stderr}")

        # Inject @main inside the module
        lines = p1.stdout.splitlines()
        body_lines = lines[1:-2]   # strip 'module {' and '}' + trailing blank
        mlir_with_main = (
            "module {\n"
            + "\n".join(body_lines)
            + "\n"
            + main_suffix
        )

        # Step 2: lower to LLVM dialect
        p2 = subprocess.run(
            [MLIR_OPT, "--test-lower-to-llvm"],
            input=mlir_with_main, capture_output=True, text=True,
        )
        if p2.returncode != 0:
            raise RuntimeError(f"mlir-opt failed:\n{p2.stderr}")

        # Step 3: JIT execute
        p3 = subprocess.run(
            [MLIR_RUNNER, "--entry-point-result=void",
             f"--shared-libs={RUNNER_UTILS},{C_UTILS}"],
            input=p2.stdout, capture_output=True, text=True,
        )
        if p3.returncode != 0:
            raise RuntimeError(f"mlir-runner failed:\n{p3.stderr}")

        # Parse "( value )" output from vector.print
        output = p3.stdout.strip()
        val = float(output.strip("( )"))
        return val

    finally:
        os.unlink(src_path)


# ── Test cases ─────────────────────────────────────────────────────────────────
BINOM_PASSES = [
    "--canonicalize", "--remove-unnecessary-enzyme-ops",
    "--lower-enzyme-binomial-progress",
    "--canonicalize", "--enzyme-simplify-math", "--canonicalize",
]
WONLINE_PASSES = [
    "--canonicalize", "--remove-unnecessary-enzyme-ops",
    "--lower-enzyme-online-progress", "--lower-enzyme-binomial-progress",
    "--canonicalize", "--enzyme-simplify-math", "--canonicalize",
]

# n=5 with budget=3 triggers a JIT crash in the mlir-runner's AArch64 code
# generator for complex Revolve schedules; use budget=10 for n>=5.
TEST_CASES = [
    ("binom",    "f_binom_grad",   BINOM_PASSES,   1),
    ("binom",    "f_binom_grad",   BINOM_PASSES,   2),
    ("binom",    "f_binom_grad",   BINOM_PASSES,   3),
    ("binom",    "f_binom_grad",   BINOM_PASSES,   4),
    ("binom",    "f_binom_grad",   BINOM_PASSES,   5),
    ("wonline",  "f_wonline_grad", WONLINE_PASSES, 1),
    ("wonline",  "f_wonline_grad", WONLINE_PASSES, 2),
    ("wonline",  "f_wonline_grad", WONLINE_PASSES, 3),
    ("wonline",  "f_wonline_grad", WONLINE_PASSES, 4),
    # n=5 with the default budget=10 in WONLINE_BODY works correctly
    ("wonline",  "f_wonline_grad", WONLINE_PASSES, 5),
]

failures = []
for scheme, grad_fn, passes, n in TEST_CASES:
    exp = expected_grad(1.2, n)
    try:
        got = run_gradient(scheme, n, grad_fn, passes)
        ok = abs(got - exp) / max(abs(exp), 1e-12) < 1e-4
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures.append((scheme, n, exp, got))
        print(f"{status}  {scheme} n={n}  expected={exp:.6f}  got={got:.6f}")
    except Exception as e:
        failures.append((scheme, n, exp, f"ERROR: {e}"))
        print(f"ERROR {scheme} n={n}: {e}")

print()
if failures:
    print(f"FAILED: {len(failures)} test(s)")
    sys.exit(1)
else:
    print(f"All {len(TEST_CASES)} tests passed.")
