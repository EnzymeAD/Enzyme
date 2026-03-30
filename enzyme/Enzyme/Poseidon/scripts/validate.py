#!/usr/bin/env python3
"""
Poseidon validation reference script.

Compiles a uniformly sampled subset of Pareto-optimal variants with
Enzyme/Poseidon, runs each, and reports geomean relative error against
a gold reference and runtime.

This is a REFERENCE SCRIPT meant to be copied and adapted for your
application.  Search for "CUSTOMIZE" to find the places that likely
need modification for a new benchmark.

Usage:
    python validate.py <source.cpp> --enzyme-plugin <ClangEnzyme.so> \
                       [--cxx <clang++>] [--extra-flags "..."] \
                       [--num-samples 10] [--num-runs 5] \
                       [--gold-path gold.txt]
"""

import argparse, glob, json, math, os, re, subprocess, sys, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = json.load(open(os.path.join(SCRIPT_DIR, "validate_config.json")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_artifact(pattern):
    hits = glob.glob(pattern)
    return hits[0] if hits else None


def run(cmd, desc="", timeout=600, capture=False):
    """Run a command and optionally capture stdout."""
    label = " ".join(cmd) if len(cmd) < 8 else f"{cmd[0]} ... ({desc})"
    print(f"  $ {label}")
    try:
        r = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
        if r.returncode != 0:
            print(f"  FAILED ({desc}): exit {r.returncode}")
            if capture and r.stderr:
                print(r.stderr[:500])
            return None
        return r.stdout if capture else ""
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({desc})")
        return None


# ---- CUSTOMIZE: output parsing -------------------------------------------
# Replace this with a parser suited to your application's output format.
# The default extracts every floating-point number from stdout.

def parse_output(text):
    """Extract comparable numeric values from program output.

    Returns a list of floats in the order they appear.  Adapt this for
    your application -- e.g. skip header lines, parse specific columns,
    or read a binary file instead.
    """
    return [float(m) for m in
            re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", text)]


# ---- CUSTOMIZE: error metric ---------------------------------------------
# The default is geomean relative error.  Replace with ULP, L2 norm, or
# whatever is appropriate for your domain.

def compute_error(gold_vals, test_vals):
    """Compute geomean and max relative error between two value lists."""
    n = min(len(gold_vals), len(test_vals))
    if n == 0:
        return float("nan"), float("nan")
    log_sum = 0.0
    count = 0
    max_err = 0.0
    for i in range(n):
        g, t = gold_vals[i], test_vals[i]
        if g == 0 and t == 0:
            continue
        denom = abs(g) if g != 0 else abs(t)
        rel = abs(g - t) / denom
        max_err = max(max_err, rel)
        if rel > 0:
            log_sum += math.log(rel)
            count += 1
    if count == 0:
        return 0.0, 0.0
    return math.exp(log_sum / count), max_err


# ---- CUSTOMIZE: runtime measurement --------------------------------------
# The default uses wall-clock time.  If your program reports its own
# timing (e.g. "Elapsed: 1.23s"), parse it here for more stable results.

def measure_runtime(exe, num_runs, extra_run_args=None):
    """Measure median wall-clock runtime of an executable."""
    times = []
    args = [exe] + (extra_run_args or [])
    for _ in range(num_runs):
        t0 = time.perf_counter()
        r = subprocess.run(args, capture_output=True, text=True, timeout=300)
        t1 = time.perf_counter()
        if r.returncode == 0:
            times.append(t1 - t0)
    if not times:
        return float("nan")
    times.sort()
    return times[len(times) // 2]


# ---- CUSTOMIZE: how to run the program and capture output -----------------
# The default runs the executable with extra_run_args and captures stdout.
# If your program writes to a file, reads from stdin, or needs environment
# variables, modify this function.

def run_and_capture(exe, extra_run_args):
    """Run executable and return its stdout as a string, or None on failure."""
    return run([exe] + extra_run_args, f"run {os.path.basename(exe)}", capture=True)


def uniform_sample(lst, n):
    """Pick n uniformly spaced items from lst (always include first and last)."""
    if n >= len(lst):
        return list(range(len(lst)))
    if n <= 0:
        return []
    if n == 1:
        return [0]
    step = (len(lst) - 1) / (n - 1)
    return sorted(set(int(round(i * step)) for i in range(n)))


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_gold(source, cxx, raptor_dir, extra_flags, gold_flags, out_dir):
    """Compile with RAPTOR for MPFR gold reference.

    The source must contain RAPTOR truncation calls (e.g. guarded by
    -DPOSEIDON_GOLD).  See the Poseidon README for the full pattern.
    """
    plugin = find_artifact(os.path.join(raptor_dir, "pass", "ClangRaptor-*.so"))
    rt = find_artifact(os.path.join(raptor_dir, "runtime", "libRaptor-RT-*.a"))
    if not plugin or not rt:
        print(f"Error: RAPTOR artifacts not found in {raptor_dir}")
        return None
    rt_dir = os.path.dirname(rt)
    rt_name = os.path.basename(rt).replace("lib", "", 1).replace(".a", "")

    exe = os.path.join(out_dir, "gold.exe")
    cmd = [cxx, source] + extra_flags.split() + gold_flags.split() + [
        f"-fpass-plugin={plugin}",
        "-Xclang", "-load", "-Xclang", plugin,
        f"-L{rt_dir}", f"-l{rt_name}",
        "-lmpfr", "-lm",
        "-o", exe,
    ]
    if run(cmd, "compile gold") is None:
        return None
    return exe


def compile_variant(source, cxx, enzyme_plugin, budget, config, extra_flags, out_dir):
    """Compile a Poseidon-optimized variant at a given budget."""
    exe = os.path.join(out_dir, f"opt_{budget}.exe")
    cmd = [cxx, source] + extra_flags.split() + [
        f"-fpass-plugin={enzyme_plugin}",
        "-Xclang", "-load", "-Xclang", enzyme_plugin,
        "-mllvm", f"--fpprofile-use={config['profile_path']}",
        "-mllvm", "--fpopt-enable-solver",
        "-mllvm", "--fpopt-enable-herbie=1",
        "-mllvm", "--fpopt-enable-pt",
        "-mllvm", f"--fpopt-comp-cost-budget={budget}",
        "-mllvm", f"--fpopt-cache-path={config['cache_path']}",
        "-lmpfr", "-lm",
        "-o", exe,
    ]
    if run(cmd, f"compile budget={budget}") is None:
        return None
    return exe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Poseidon accuracy/runtime validation (reference script)")
    p.add_argument("source", help="C/C++ source file")
    p.add_argument("--enzyme-plugin", required=True,
                   help="Path to ClangEnzyme-XX.so")
    p.add_argument("--cxx", default="clang++",
                   help="C++ compiler (default: clang++)")
    p.add_argument("--extra-flags", default="-O3 -ffast-math -lm",
                   help="Compile flags (must match profiling flags)")
    p.add_argument("--extra-run-args", default="",
                   help="Args passed to each executable at runtime")
    p.add_argument("--num-samples", type=int, default=10,
                   help="Number of Pareto points to validate")
    p.add_argument("--num-runs", type=int, default=5,
                   help="Runtime measurement repetitions per variant")
    p.add_argument("--gold-path", default="",
                   help="Pre-computed gold reference (skip RAPTOR compilation)")
    p.add_argument("--raptor-dir", default="",
                   help="RAPTOR build directory (for gold compilation)")
    p.add_argument("--gold-source", default="",
                   help="Source file for gold compilation (if different)")
    p.add_argument("--gold-flags", default="",
                   help="Extra flags for gold compilation (e.g. -DPOSEIDON_GOLD)")
    args = p.parse_args()

    budgets = CONFIG["budgets"]
    est_acc = CONFIG.get("estimated_accuracy_costs", [])
    run_args = args.extra_run_args.split() if args.extra_run_args else []

    out_dir = os.path.join(SCRIPT_DIR, "validation_output")
    os.makedirs(out_dir, exist_ok=True)

    # --- Gold reference ---
    if args.gold_path:
        print(f"=== Using pre-computed gold reference: {args.gold_path} ===")
        gold_text = open(args.gold_path).read()
    else:
        print("=== Step 1: Compiling MPFR gold reference with RAPTOR ===")
        if not args.raptor_dir:
            print("Error: --raptor-dir required for gold compilation.")
            print("Either provide --gold-path or --raptor-dir.")
            sys.exit(1)
        gold_src = args.gold_source if args.gold_source else args.source
        gold_exe = compile_gold(gold_src, args.cxx, args.raptor_dir,
                                args.extra_flags, args.gold_flags, out_dir)
        if not gold_exe:
            print("Gold compilation failed.")
            sys.exit(1)
        print("Running gold binary...")
        gold_text = run_and_capture(gold_exe, run_args)
        if gold_text is None:
            print("Error: gold run failed.")
            sys.exit(1)

    gold_vals = parse_output(gold_text)
    print(f"Gold reference: {len(gold_vals)} values extracted.\n")

    # --- Original (unoptimized) baseline ---
    print("=== Step 2: Measuring baseline (original) ===")
    orig_exe = os.path.join(out_dir, "original.exe")
    orig_cmd = [args.cxx, args.source] + args.extra_flags.split() + [
        f"-fpass-plugin={args.enzyme_plugin}",
        "-Xclang", "-load", "-Xclang", args.enzyme_plugin,
        "-lmpfr", "-lm", "-o", orig_exe,
    ]
    orig_geo_err = float("nan")
    orig_max_err = float("nan")
    orig_runtime = float("nan")
    if run(orig_cmd, "compile original") is not None:
        orig_runtime = measure_runtime(orig_exe, args.num_runs, run_args)
        orig_text = run_and_capture(orig_exe, run_args)
        if orig_text is not None:
            orig_vals = parse_output(orig_text)
            orig_geo_err, orig_max_err = compute_error(gold_vals, orig_vals)
        print(f"Baseline runtime: {orig_runtime:.6f}s, "
              f"geomean error: {orig_geo_err:.6e}, max error: {orig_max_err:.6e}\n")
    else:
        print("Warning: could not compile original.\n")

    # --- Sample and compile Pareto variants ---
    sample_indices = uniform_sample(budgets, args.num_samples)
    sampled_budgets = [budgets[i] for i in sample_indices]
    sampled_est = [est_acc[i] if i < len(est_acc) else float("nan")
                   for i in sample_indices]

    print(f"=== Step 3: Compiling {len(sampled_budgets)} Pareto variants ===")

    results = []
    for budget, est in zip(sampled_budgets, sampled_est):
        print(f"\n--- Budget: {budget} (estimated acc cost: {est:.6e}) ---")
        exe = compile_variant(args.source, args.cxx, args.enzyme_plugin,
                              budget, CONFIG, args.extra_flags, out_dir)
        if not exe:
            results.append({"budget": budget, "error": "compile_failed"})
            continue

        test_text = run_and_capture(exe, run_args)
        if test_text is None:
            results.append({"budget": budget, "error": "run_failed"})
            continue

        test_vals = parse_output(test_text)
        geo_err, max_err = compute_error(gold_vals, test_vals)
        rt = measure_runtime(exe, args.num_runs, run_args)
        speedup = orig_runtime / rt if rt > 0 and not math.isnan(orig_runtime) else float("nan")

        results.append({
            "budget": budget,
            "estimated_accuracy_cost": est,
            "geomean_relative_error": geo_err,
            "max_relative_error": max_err,
            "runtime": rt,
            "speedup": speedup,
        })
        print(f"  geomean error: {geo_err:.6e}, max error: {max_err:.6e}, "
              f"runtime: {rt:.6f}s, speedup: {speedup:.2f}x")

    # --- Summary ---
    print("\n" + "=" * 72)
    print(f"{'Budget':>10} {'Est.AccCost':>12} {'GeomErr':>12} {'MaxErr':>12} "
          f"{'Runtime':>10} {'Speedup':>8}")
    print("-" * 72)
    if not math.isnan(orig_runtime):
        print(f"{'ORIGINAL':>10} {'--':>12} "
              f"{orig_geo_err:>12.4e} {orig_max_err:>12.4e} "
              f"{orig_runtime:>10.6f} {'1.00x':>8}")
        print("-" * 72)
    for r in results:
        if "error" in r and isinstance(r.get("error"), str):
            print(f"{r['budget']:>10} {'':>12} {r['error']:>12}")
            continue
        print(f"{r['budget']:>10} {r.get('estimated_accuracy_cost',0):>12.4e} "
              f"{r['geomean_relative_error']:>12.4e} {r['max_relative_error']:>12.4e} "
              f"{r['runtime']:>10.6f} {r['speedup']:>7.2f}x")
    print("=" * 72)

    # Save results
    results_path = os.path.join(SCRIPT_DIR, f"{CONFIG['function']}_validation.json")
    with open(results_path, "w") as f:
        json.dump({"config": CONFIG,
                    "baseline": {"runtime": orig_runtime,
                                 "geomean_relative_error": orig_geo_err,
                                 "max_relative_error": orig_max_err},
                    "results": results}, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
