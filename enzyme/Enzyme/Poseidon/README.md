# Poseidon

Poseidon is a modular and extensible framework that fully automates advanced floating-point rewriting techniques for real-world applications within a production compiler. It operates as a PGO-like two-phase compiler that automatically extract numerical context (e.g., value ranges, sensitivities) from small surrogate profiling runs. It synthesizes algebraic rewrites via [Herbie](https://herbie.uwplse.org/), generates precision tuning candidates, and uses a dynamic programming solver to find a Pareto frontier of optimized programs.

For details, please read our paper [Thinking Fast and Correct: Automated Rewriting of Numerical Code through Compiler Augmentation](https://ece.is/assets/pdf/poseidon-cgo26.pdf) (CGO 2026).

If you use Poseidon in an academic setting, please kindly cite:

```bibtex
@inproceedings{poseidon,
  author={Qian, Siyuan Brant and Sathia, Vimarsh and Ivanov, Ivan R. and H\"{u}ckelheim, Jan and Hovland, Paul and Moses, William S.},
  booktitle={2026 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)},
  title={Thinking Fast and Correct: Automated Rewriting of Numerical Code through Compiler Augmentation},
  year={2026},
  pages={548-562},
  doi={10.1109/CGO68049.2026.11395228}
}
```

## Table of Contents

- [Build](#build)
- [Two-Phase Pipeline](#two-phase-pipeline)
- [How to Apply Rewrites?](#how-to-apply-rewrites)
- [Reporting](#reporting)
  - [Per-Rewrite Analysis](#per-rewrite-analysis)
  - [Applying User-Selected Rewrites](#applying-user-selected-rewrites)
- [Optimized Program Validation](#optimized-program-validation)
- [Generating MPFR References with RAPTOR](#optional-generating-mpfr-references-with-raptor)
- [Command-Line Reference](#command-line-reference)


## Build

We recommend building from source and generating a hardware-specific cost model for best results. The cost model estimates per-operation latencies on your machine and is used by the DP solver to estimate computation costs.

### Prerequisites

```bash
sudo apt install build-essential cmake ninja-build libmpfr-dev
pip install lit numpy matplotlib tqdm
```

Additionally, install [Racket](https://racket-lang.org/) and [Rust](https://www.rust-lang.org/tools/install).

### Build LLVM

```bash
cd llvm-project
mkdir build && cd build
cmake -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=Release \
  ../llvm
ninja
cd ../..
```

### Build Enzyme with Poseidon Enabled

```bash
cd Enzyme
mkdir build && cd build
cmake -G Ninja ../enzyme/ \
  -DLLVM_DIR=<...>/llvm-project/build/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$(which lit) \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_POSEIDON=ON \
  -DCMAKE_C_COMPILER=<...>/llvm-project/build/bin/clang \
  -DCMAKE_CXX_COMPILER=<...>/llvm-project/build/bin/clang++
ninja
cd ../..
```

Replace `<...>` with the appropriate path prefix.

A preconfigured Docker image is also available; see the [CGO artifact repository](https://github.com/PRONTOLab/Poseidon) for details.

### Generate the Cost Model

The cost model (`cm.csv`) is hardware-specific. A benchmarking script is provided at `Poseidon/scripts/microbm.py`. To generate the cost model for your machine:

```bash
python3 <enzyme-src>/Enzyme/Poseidon/scripts/microbm.py
cp results.csv cm.csv
```

Then pass `--fpopt-cost-model-path=cm.csv` during the optimization phase. An example cost model (generated on AMD Ryzen Threadripper PRO 7995WX) is provided at `Poseidon/scripts/cm_example.csv` for reference. Without a cost model, Poseidon falls back to LLVM's `TargetTransformInfo` estimates.

## Two-Phase Pipeline

See the [dquat benchmark](https://github.com/PRONTOLab/Poseidon/tree/main/dquat) for an end-to-end example.

:warning: Please Note: Both phases we introduce below must use identical compiler flags (e.g., `-O3`, `-ffast-math`, `-march=native`, etc.) to ensure profile indices match between compilations.

### User Code Modification

Annotate the function to optimize with `__enzyme_fp_optimize`:

```cpp
template <typename return_type, typename... T>
return_type __enzyme_fp_optimize(void *, T...);
int enzyme_dup;

__attribute__((noinline))
void my_kernel(double *out, const double *in);

void run() {
    double out[N], out_grad[N], in[M], in_grad[M];
    // Gradients seed the sensitivity analysis (set to 1.0 for uniform weighting)
    std::fill(out_grad, out_grad + N, 1.0);
    __enzyme_fp_optimize<void>((void *)my_kernel,
        enzyme_dup, out, out_grad,
        enzyme_dup, in, in_grad);
}
```

### Phase 1: Profiling

```bash
clang++ mycode.cpp $CXXFLAGS \
  -fpass-plugin=ClangEnzyme-XX.so -Xclang -load -Xclang ClangEnzyme-XX.so \
  -mllvm --fpprofile-generate \
  -L$ENZYME_BUILD/Enzyme -lEnzymeFPProfile -lm -o mycode-prof

ENZYME_FPPROFILE_DIR=./fpprofile ./mycode-prof <surrogate_inputs>
```

The instrumented binary records per-instruction value ranges, gradient sensitivities, and execution counts into a `.fpprofile` directory. A small surrogate input (e.g., 100 samples) is often sufficient.

### Phase 2: Optimization

```bash
clang++ mycode.cpp $CXXFLAGS \
  -fpass-plugin=ClangEnzyme-XX.so -Xclang -load -Xclang ClangEnzyme-XX.so \
  -mllvm --fpprofile-use=./fpprofile \
  -mllvm --fpopt-enable-herbie=1 \
  -mllvm --fpopt-enable-solver \
  -mllvm --fpopt-enable-pt \
  -mllvm --fpopt-comp-cost-budget=0 \
  -mllvm --fpopt-cache-path=./cache \
  -mllvm --fpopt-cost-model-path=cm.csv \
  -mllvm --fpopt-strict-mode \
  -lmpfr -lm -o mycode-opt
```

The first run invokes Herbie and the DP solver; results are cached in `--fpopt-cache-path`. Subsequent runs reuse the cache. The `--fpopt-comp-cost-budget` selects a point on the Pareto curve (0 = no-op baseline).

## How to Apply Rewrites?

Poseidon offers two ways to apply numerical rewrites:

1. **Select an optimized program from the Pareto frontier.** Pass `--fpopt-comp-cost-budget=N` to pick a point computed by the DP solver. Each budget value corresponds to a different combination of rewrites and precision changes that the solver determined to be Pareto-optimal. The full set of available budgets is listed in `validate_config.json` (when using `--fpopt-report-path`) and `cache/budgets.txt`.

2. **Obtain a custom optimized program.** Generate a [report](#reporting), review the individual rewrites in `_rewrites.json`, and pass the IDs of the ones you want via `--fpopt-apply-rewrites=R3_0,PT1_0,...`. This bypasses the DP solver and gives fine-grained control over exactly which rewrites are applied. See [Applying User-Selected Rewrites](#applying-user-selected-rewrites) for details.

## Reporting

By default, Poseidon silently applies the best solution within the given budget to the compiled binary. To understand *what* was changed and *why*, add `--fpopt-report-path=<dir>` (with `-g` for source locations):

```bash
clang++ mycode.cpp $CXXFLAGS -g ... \
  -mllvm --fpopt-report-path=./report
```

### Report Contents

| File | Description |
|------|-------------|
| `<func>.json` | Full Pareto table with details |
| `<func>.txt` | Plain-text version of the Pareto table |
| `<func>_rewrites.json` | Detailed per-rewrite information: all rewrites categorized and ranked |

Source locations (file, line, column) are populated when compiled with `-g`. Without `-g`, the symbolic expressions and affected IR are still available.

:bulb: For complex programs/rewrites, it may be beneficial to feed the `.txt` report and the program source code to an LLM and asking it to explain what each rewrite does and why it improves numerical accuracy.

### Understanding the Pareto Report

The `<func>.json` and `<func>.txt` reports describe the full Pareto frontier. Each Pareto point represents one **optimized program** — a combination of rewrites and precision changes selected by the DP solver at a given computation cost budget. For example:

```
--- Pareto Point #5: Cost=-6863264, Accuracy=3.964900e-01 ---
  [Rewrite] (* (sqrt (fma v6 v6 (fma v5 v5 (fma v4 v4 0)))) 0.25)
        --> (*.f64 (sqrt.f64 ...) #s(literal 1/4 binary64))
    Herbie accuracy: 0.935 -> 0.999 bits
    Source: dquat.cpp:87:18
    Source: dquat.cpp:56:11
    Affected IR:
      %mul = fmul fast double %sqrt, 2.500000e-01
```

### Per-Rewrite Analysis

The `<func>_rewrites.json` file lists every **individual** rewrite candidate, categorized by its estimated impact:

- **`free_win`** — improves both accuracy and speed. Always beneficial.
- **`accuracy_for_speed`** — improves accuracy at the cost of extra computation.
- **`speed_for_accuracy`** — improves speed at the cost of some accuracy.

Each entry includes an `efficiency` score that ranks tradeoffs: higher means more benefit per unit of cost. Entries are sorted by category (free wins first) then by efficiency descending.

Each rewrite has a stable `id` (e.g., `R3_0`, `PT1_2`) that can be used with `--fpopt-apply-rewrites` (see below).

Example:
```json
{
  "id": "R6_3",
  "category": "speed_for_accuracy",
  "efficiency": 4.602e+11,
  "computation_cost_delta": -22632,
  "accuracy_cost_delta": 4.918e-08,
  "original_expr": "(* (/ 1 (sqrt ...)) v5)",
  "rewritten_expr": "#s(approx ...)",
  "source_location": {"file": "dquat.cpp", "line": 116, "col": 18}
}
```

### Applying User-Selected Rewrites

A user can pick specific rewrites from `_rewrites.json` by their IDs (this bypasses the DP solver):

```bash
clang++ mycode.cpp $CXXFLAGS ... \
  -mllvm --fpopt-apply-rewrites=R5_5,R6_3,R7_3,PT1_0
```

With a few constraints:
- At most one rewrite per expression (e.g., `R5_0` and `R5_1` conflict)
- At most one precision change per subgraph (e.g., `PT1_0` and `PT1_2` conflict)

Duplicates are detected and skipped with a warning.

## Optimized Program Validation

To validate the actual accuracy and runtime of optimized programs, a reference validation script is provided at `Poseidon/scripts/validate.py`. Copy it to your report directory alongside `validate_config.json`, then run:

```bash
cp <enzyme-src>/Enzyme/Poseidon/scripts/validate.py ./report/
python3 report/validate.py mycode.cpp \
  --enzyme-plugin /path/to/ClangEnzyme-XX.so \
  --cxx /path/to/clang++ \
  --extra-flags "-O3 -ffast-math -march=native -fno-exceptions -lmpfr" \
  --extra-run-args "--num-tests 100 --seed 42" \
  --gold-path gold_mpfr.txt \
  --num-samples 10 \
  --num-runs 5
```

### What it does

1. **Loads a gold accuracy reference** from `--gold-path` (MPFR ground truth; see [RAPTOR section](#generating-mpfr-gold-references-with-raptor) below)
2. **Compiles the original** (unoptimized) binary, measures its runtime and accuracy against gold
3. **Uniformly samples** N budgets from the full Pareto table (87 points for dquat with `-ffast-math`)
4. **For each budget:** recompiles with Poseidon at that budget using the cached Herbie results + DP table, runs the resulting binary, captures output
5. **Computes** geomean and max relative error against gold for each variant
6. **Measures** median runtime over multiple runs
7. **Prints a summary table** showing the original program as baseline, then each Pareto variant with estimated vs. validated accuracy and speedup

### Example output

```
========================================================================
    Budget  Est.AccCost      GeomErr       MaxErr    Runtime  Speedup
------------------------------------------------------------------------
  ORIGINAL           --   6.3760e-11   1.1794e+03   0.000015    1.00x
------------------------------------------------------------------------
  -7064000   3.5985e+01   6.3990e-03   3.4521e+02   0.000013    1.10x
  -6863264   3.9649e-01   5.9403e-09   1.1794e+03   0.000014    1.08x
   1096856  -1.1591e-16   6.8253e-11   1.1794e+03   0.000013    1.13x
========================================================================
```

Here `ORIGINAL` is the unoptimized double-precision program. Negative budgets allow the solver to trade accuracy for speed; positive budgets allow extra computation to improve accuracy. The `MaxErr: 1179` in the original comes from catastrophic cancellation (`1 - cos(theta)` near zero), which Herbie's rewrites at positive budgets fix.

## (Optional) Generating MPFR References with RAPTOR

For accuracy validation we recommend computing reference results at high floating-point precision (e.g., MPFR-2048 bits). [RAPTOR](https://github.com/RIKEN-RCCS/RAPTOR) provides this capability by running floating-point operations through MPFR. For the latest build and usage instructions of RAPTOR, see the [RAPTOR repository](https://github.com/RIKEN-RCCS/RAPTOR).

### Building RAPTOR

```bash
git clone https://github.com/RIKEN-RCCS/RAPTOR
cd RAPTOR && mkdir build && cd build
cmake .. -DLLVM_DIR=/path/to/llvm -DCMAKE_BUILD_TYPE=Release
make -j
```

### Source preparation

Add a `#ifdef POSEIDON_GOLD` guard that wraps the target function call with RAPTOR's MPFR truncation:

```cpp
#ifdef POSEIDON_GOLD
template <typename fty>
__attribute__((nothrow)) fty *__raptor_truncate_mem_func(fty *, int, int, int, int);
__attribute__((nothrow)) extern double __raptor_truncate_mem_value(...);
__attribute__((nothrow)) extern double __raptor_expand_mem_value(...);
extern "C" double raptor_fprt_gc_mark_seen(double);
extern "C" void raptor_fprt_gc_doit();
// RAPTOR API: (func_ptr, from_bits, type_selector, exponent, mantissa)
// type_selector: 0=IEEE, 1=MPFR
#define RAPTOR_FROM 64
#define RAPTOR_TYPE 1
#define RAPTOR_TO_E 64
#define RAPTOR_TO_M 2048
#define TRUNC_SELF(X) X = __raptor_truncate_mem_value(X, RAPTOR_FROM, RAPTOR_TYPE, RAPTOR_TO_E, RAPTOR_TO_M)
#define EXPAND_SELF(X) X = __raptor_expand_mem_value(X, RAPTOR_FROM, RAPTOR_TYPE, RAPTOR_TO_E, RAPTOR_TO_M)
#else
int enzyme_dup;
template <typename return_type, typename... T>
return_type __enzyme_fp_optimize(void *, T...);
#endif
```

In the main loop:

```cpp
#ifdef POSEIDON_GOLD
    for (int i = 0; i < num_inputs; i++) TRUNC_SELF(inputs[i]);
    __raptor_truncate_mem_func(my_kernel, RAPTOR_FROM, RAPTOR_TYPE,
                               RAPTOR_TO_E, RAPTOR_TO_M)(outputs, inputs);
    for (int i = 0; i < num_outputs; i++) EXPAND_SELF(outputs[i]);
    for (int i = 0; i < num_inputs; i++) EXPAND_SELF(inputs[i]);
    raptor_fprt_gc_doit();
#else
    __enzyme_fp_optimize<void>((void *)my_kernel,
        enzyme_dup, outputs, outputs_grad, enzyme_dup, inputs, inputs_grad);
#endif
```

### Compiling and running

```bash
clang++ mycode.cpp -O0 -fno-exceptions -DPOSEIDON_GOLD \
  -fpass-plugin=/path/to/RAPTOR/build/pass/ClangRaptor-XX.so \
  -Xclang -load -Xclang /path/to/RAPTOR/build/pass/ClangRaptor-XX.so \
  -L/path/to/RAPTOR/build/runtime -lRaptor-RT-XX \
  -lmpfr -lm -o gold.exe

./gold.exe --output-path gold_mpfr.txt
```

Then pass `--gold-path gold_mpfr.txt` to `validate.py`.

## Command-Line Reference

### Profiling
| Flag | Default | Description |
|------|---------|-------------|
| `--fpprofile-generate` | false | Instrument for FP profiling |
| `--fpprofile-use=<dir>` | | Profile directory for optimization |

### Herbie
| Flag | Default | Description |
|------|---------|-------------|
| `--fpopt-enable-herbie` | true | Use Herbie for algebraic rewrites |
| `--fpopt-cache-path` | "cache" | Cache directory for Herbie results and DP table |
| `--herbie-num-threads` | 8 | Herbie worker threads |
| `--herbie-timeout` | 120 | Per-expression Herbie timeout (seconds) |

### Solver
| Flag | Default | Description |
|------|---------|-------------|
| `--fpopt-enable-solver` | true | Enable DP solver |
| `--fpopt-enable-pt` | true | Enable precision tuning candidates |
| `--fpopt-comp-cost-budget` | 0 | Computation cost budget (0 = no-op) |
| `--fpopt-cost-model-path` | | Hardware-specific cost model CSV |
| `--fpopt-strict-mode` | false | Discard candidates that produce NaN/inf |
| `--fpopt-num-samples` | 1024 | Number of MPFR evaluation samples |
| `--fpopt-early-prune` | true | Prune dominated candidates during DP |
| `--fpopt-loose-coverage` | false | Allow unexecuted instructions (suppress coverage errors) |

### Reporting
| Flag | Default | Description |
|------|---------|-------------|
| `--fpopt-report-path` | | Output directory for JSON/text reports + validate.py |
| `--fpopt-apply-rewrites` | | Comma-separated rewrite IDs from `_rewrites.json` (bypasses DP solver) |
| `--fpopt-print` | false | Print debug info to stderr |
| `--fpopt-show-table` | false | Print full DP table to stderr |
