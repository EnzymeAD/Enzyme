#!/usr/bin/env python3
"""
Execution test: Oceananigans-style simulation differentiated with four schemes.

The simulation mirrors Oceananigans' run!/time_step! structure:
  - run!      : while sim.running; time_step!(sim); end
  - time_step!: advance state, increment clock, set sim.running = (time < stop_time)

In MLIR we model this as an scf.while where:
  - iter_args: (state: f32, time: index, running: i1)
  - condition block: scf.condition(%running) ...
  - after block: updates state, increments time, sets running = (time < stop_time)

The stopping criterion is evaluated INSIDE the loop body, so the trip count is
not known before the loop starts — this is the canonical Online_r2 use case.

Four differentiation schemes are compared:
  PLAIN   — scf.for,  no checkpointing  (known trip count for reference)
  BINOM   — scf.for,  binomial (Revolve)
  WTAPE   — scf.while, full-tape (enzyme.push/pop)
  WONLINE — scf.while, online (Stumm-Walther 2010 / Online_r2)

All four must agree to within 1e-4 relative tolerance.

Usage:
  python3 exec_oceananigans_style.py [enzymemlir-opt] [mlir-opt] [mlir-runner] \\
                                     [libmlir_runner_utils] [libmlir_c_runner_utils]
"""

import math
import os
import subprocess
import sys
import tempfile

# ── Tool paths ─────────────────────────────────────────────────────────────────
ENZYME_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
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

cli = sys.argv[1:]
EOPT        = cli[0] if len(cli) > 0 else defaults["eopt"]
MLIR_OPT    = cli[1] if len(cli) > 1 else defaults["mlir_opt"]
MLIR_RUNNER = cli[2] if len(cli) > 2 else defaults["mlir_runner"]
RUNNER_UTILS= cli[3] if len(cli) > 3 else defaults["runner_utils"]
C_UTILS     = cli[4] if len(cli) > 4 else defaults["c_utils"]

# ── Expected gradient (analytical) ────────────────────────────────────────────
def expected_grad(x0, n_steps):
    """df/dx0 where f = cos(x*x) iterated n_steps times (chain rule)."""
    s, g = x0, 1.0
    for _ in range(n_steps):
        sq = s * s
        g  = g * (-math.sin(sq)) * 2 * s
        s  = math.cos(sq)
    return g

# ── MLIR sources ───────────────────────────────────────────────────────────────
# f_plain / f_binom: scf.for with known trip count %n (used as reference).
# These are equivalent to a while loop that runs exactly n times but with
# the trip count given explicitly — simpler to write, same math.

FOR_SRC = """\
module {{
  func.func @f_plain(%arg0: f32, %n: index) -> f32 {{
    %lb = arith.constant 0 : index
    %st = arith.constant 1 : index
    %r  = scf.for %iv = %lb to %n step %st iter_args(%s = %arg0) -> f32 {{
      %sq  = arith.mulf %s, %s : f32
      %cos = math.cos  %sq     : f32
      scf.yield %cos : f32
    }}
    return %r : f32
  }}
  func.func @f_binom(%arg0: f32, %n: index) -> f32 {{
    %lb = arith.constant 0 : index
    %st = arith.constant 1 : index
    %r  = scf.for %iv = %lb to %n step %st iter_args(%s = %arg0) -> f32 {{
      %sq  = arith.mulf %s, %s : f32
      %cos = math.cos  %sq     : f32
      scf.yield %cos : f32
    }} {{enzyme.enable_checkpointing = true,
        enzyme.binomial_checkpointing,
        enzyme.checkpoint_period = {budget} : i64}}
    return %r : f32
  }}
}}
"""

# f_while_tape / f_while_online: scf.while where the stopping criterion
# (time < stop_time) is evaluated INSIDE the loop body, exactly as in
# Oceananigans' time_step!.
#
# iter_args:  (state: f32,  step: index,  running: i1)
# condition:  scf.condition(%running) state, step, running
# after-block:
#   new_state   = cos(state * state)          -- physics
#   new_step    = step + 1                    -- clock increment
#   new_running = new_step < stop_time        -- stopping criterion set inside body
#   scf.yield new_state, new_step, new_running
#
# The primal loop starts with running=true and exits when new_running=false,
# i.e. after exactly stop_time steps (since step starts at 0 and stop_time
# is an integer here).

WHILE_SRC = """\
module {{
  func.func @f_while_tape(%arg0: f32, %stop: index) -> f32 {{
    %true    = arith.constant true
    %lb      = arith.constant 0 : index
    %st      = arith.constant 1 : index
    %r:3 = scf.while (%s = %arg0, %step = %lb, %run = %true)
           : (f32, index, i1) -> (f32, index, i1) {{
      scf.condition(%run) %s, %step, %run : f32, index, i1
    }} do {{
    ^bb0(%s_in: f32, %step_in: index, %run_in: i1):
      %sq      = arith.mulf %s_in, %s_in   : f32
      %cos     = math.cos  %sq             : f32
      %step1   = arith.addi %step_in, %st  : index
      %new_run = arith.cmpi slt, %step1, %stop : index
      scf.yield %cos, %step1, %new_run : f32, index, i1
    }}
    return %r#0 : f32
  }}
  func.func @f_while_online(%arg0: f32, %stop: index) -> f32 {{
    %true    = arith.constant true
    %lb      = arith.constant 0 : index
    %st      = arith.constant 1 : index
    %r:3 = "scf.while"(%arg0, %lb, %true) ({{
    ^bb0(%s: f32, %step: index, %run: i1):
      scf.condition(%run) %s, %step, %run : f32, index, i1
    }}, {{
    ^bb0(%s_in: f32, %step_in: index, %run_in: i1):
      %sq      = arith.mulf %s_in, %s_in   : f32
      %cos     = math.cos  %sq             : f32
      %step1   = arith.addi %step_in, %st  : index
      %new_run = arith.cmpi slt, %step1, %stop : index
      scf.yield %cos, %step1, %new_run : f32, index, i1
    }}) {{enzyme.enable_checkpointing = true,
          enzyme.online_checkpointing,
          enzyme.checkpoint_period = {budget} : i64}}
         : (f32, index, i1) -> (f32, index, i1)
    return %r#0 : f32
  }}
}}
"""

MAIN_TEMPLATE = """\
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

# Passes applied by enzymemlir-opt after differentiation.
# --convert-enzyme-to-memref lowers enzyme.store/load/push/pop/init to memref.
# --reconcile-unrealized-casts cleans up the cast pairs it introduces.
# These must run BEFORE --test-lower-to-llvm (which uses plain mlir-opt and
# does not know the enzyme dialect).
_TAIL = ["--convert-enzyme-to-memref", "--reconcile-unrealized-casts"]

PLAIN_PASSES = [
    "--canonicalize", "--remove-unnecessary-enzyme-ops",
    "--canonicalize", "--enzyme-simplify-math", "--canonicalize",
] + _TAIL
BINOM_PASSES = [
    "--canonicalize", "--remove-unnecessary-enzyme-ops",
    "--lower-enzyme-binomial-progress",
    "--canonicalize", "--enzyme-simplify-math", "--canonicalize",
] + _TAIL
WTAPE_PASSES = [
    "--canonicalize", "--remove-unnecessary-enzyme-ops",
    "--canonicalize", "--enzyme-simplify-math", "--canonicalize",
] + _TAIL
WONLINE_PASSES = [
    "--canonicalize", "--remove-unnecessary-enzyme-ops",
    "--lower-enzyme-online-progress", "--lower-enzyme-binomial-progress",
    "--canonicalize", "--enzyme-simplify-math", "--canonicalize",
] + _TAIL

# ── Execution helper ───────────────────────────────────────────────────────────
def run_gradient(src_text, fn_name, grad_fn, n, enzyme_passes):
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(src_text)
        src_path = f.name

    main_block = MAIN_TEMPLATE.format(n=n, grad_fn=grad_fn)

    try:
        p1 = subprocess.run(
            [EOPT, src_path,
             f"--enzyme-wrap=infn={fn_name} outfn={grad_fn}"
             f" argTys=enzyme_active,enzyme_const retTys=enzyme_active"
             f" mode=ReverseModeCombined"]
            + enzyme_passes,
            capture_output=True, text=True,
        )
        if p1.returncode != 0:
            raise RuntimeError(f"enzymemlir-opt failed:\n{p1.stderr}")

        # Inject @main inside the module.
        # enzymemlir-opt output: "module {", ..., "}", ""  (trailing newline).
        # splitlines() gives [..., "}", ""].  Strip "module {" (first) and
        # "}" + trailing "" (last two) to get the inner function body.
        lines = p1.stdout.splitlines()
        body = "\n".join(lines[1:-2])  # strip 'module {' and '}' + trailing blank
        mlir_with_main = "module {\n" + body + "\n" + main_block

        p2 = subprocess.run(
            [MLIR_OPT, "--test-lower-to-llvm"],
            input=mlir_with_main, capture_output=True, text=True,
        )
        if p2.returncode != 0:
            raise RuntimeError(f"mlir-opt failed:\n{p2.stderr}")

        p3 = subprocess.run(
            [MLIR_RUNNER, "--entry-point-result=void",
             f"--shared-libs={RUNNER_UTILS},{C_UTILS}"],
            input=p2.stdout, capture_output=True, text=True,
        )
        if p3.returncode != 0:
            raise RuntimeError(f"mlir-runner failed:\n{p3.stderr}")

        return float(p3.stdout.strip().strip("( )"))
    finally:
        os.unlink(src_path)

# ── Test cases ─────────────────────────────────────────────────────────────────
X0     = 1.2
BUDGET = 10          # checkpoint budget (enough for all n up to 20)
RTOL   = 1e-4

for_src    = FOR_SRC.format(budget=BUDGET)
while_src  = WHILE_SRC.format(budget=BUDGET)

# (label, src, fn_name, grad_fn, passes)
SCHEMES = [
    ("BINOM",   for_src,   "f_binom",        "f_binom_grad",        BINOM_PASSES),
    ("WTAPE",   while_src, "f_while_tape",   "f_while_tape_grad",   WTAPE_PASSES),
    ("WONLINE", while_src, "f_while_online", "f_while_online_grad", WONLINE_PASSES),
]

N_VALUES = [1, 2, 3, 4, 5, 8, 10]

failures = []
print(f"{'N':>4}  {'BINOM':>12}  {'WTAPE':>12}  {'WONLINE':>12}  {'EXPECTED':>12}")
print("-" * 68)

for n in N_VALUES:
    exp = expected_grad(X0, n)
    row = f"{n:>4}"
    for label, src, fn_name, grad_fn, passes in SCHEMES:
        try:
            got = run_gradient(src, fn_name, grad_fn, n, passes)
            ok  = abs(got - exp) / max(abs(exp), 1e-12) < RTOL
            mark = "" if ok else "!"
            row += f"  {got:>12.6f}{mark}"
            if not ok:
                failures.append((label, n, exp, got))
        except Exception as e:
            row += f"  {'ERROR':>12}"
            failures.append((label, n, exp, str(e)))
    row += f"  {exp:>12.6f}"
    print(row)

print()
if failures:
    print(f"FAILED: {len(failures)} case(s):")
    for label, n, exp, got in failures:
        print(f"  {label} n={n}: expected={exp:.6f} got={got}")
    sys.exit(1)
else:
    print(f"All {len(N_VALUES) * len(SCHEMES)} cases passed.")
