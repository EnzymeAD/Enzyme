# Poseidon

Poseidon is a modular and extensible framework that fully automates floating-point optimizations for real-world applications within a production compiler. It operates as an LLVM pass inside [Enzyme](https://enzyme.mit.edu/) and uses a PGO-like two-phase compilation to automatically extract numerical context (value ranges, sensitivities) from small surrogate profiling runs. It then synthesizes algebraic rewrites via [Herbie](https://herbie.uwplse.org/), generates precision tuning candidates, and uses a dynamic programming solver to find Pareto-optimal combinations that trade off computation cost against numerical accuracy.

Unlike prior tools that require DSL inputs or manual code edits, Poseidon works directly on compiled LLVM IR and interoperates with standard compiler analyses and optimizations (mem2reg, inlining, loop unrolling, SimplifyCFG), enabling it to extract larger FP subgraphs than would be available in source code or binaries.

For details, please read our paper [Thinking Fast and Correct: Automated Rewriting of Numerical Code through Compiler Augmentation](https://ece.is/assets/pdf/poseidon-cgo26.pdf) (CGO 2026).

If you use Poseidon in an academic setting, please cite:

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

## Build

See the [artifact repository](https://github.com/PRONTOLab/Poseidon) for full build instructions including Docker. In short:

```bash
cd Enzyme && mkdir build && cd build
cmake -G Ninja ../enzyme/ \
  -DLLVM_DIR=/path/to/llvm/build/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$(which lit) \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_POSEIDON=ON
ninja
```

Both phases must use identical compiler flags (e.g., `-O3`, `-ffast-math`, `-march=native`, etc.) to ensure profile indices match between compilations.

## Two-Phase Pipeline

### User Code

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

The instrumented binary records per-instruction value ranges, gradient sensitivities, and execution counts into a `.fpprofile` directory. A small surrogate input (e.g., 1000 samples) is often sufficient.

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

## Reporting

Add `--fpopt-report-path=<dir>` to emit structured optimization reports:

```bash
clang++ mycode.cpp $CXXFLAGS ... \
  -mllvm --fpopt-report-path=./report \
```

### Report Contents

| File | Description |
|------|-------------|
| `<function>.json` | Full Pareto table: per-step source locations, symbolic expressions (original and rewritten), affected LLVM IR, Herbie accuracy bits, cost/accuracy deltas, gradient and execution count |
| `<function>.txt` | Human-readable version of the same (suitable for feeding to an LLM for explanation) |
| `validate_config.json` | Baked-in configuration: all Pareto budgets, profile path, cache path, RAPTOR dir |
| `validate.py` | Self-contained validation script |

Source locations (file, line, column) are populated when the source is compiled with `-g`. Without `-g`, the symbolic expressions and affected IR are still available.

### Understanding Rewrites

The text report lists each Pareto point with the rewrites applied. For example:

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

Each rewrite shows:
- **Original and rewritten symbolic expressions** in Herbie's S-expression format
- **Source locations** (when compiled with `-g`) pointing to the C/C++ lines affected
- **Affected LLVM IR instructions** that are replaced or erased
- **Herbie accuracy improvement** in bits

For complex rewrites, consider feeding the `.txt` report along with the source code to an LLM and asking it to explain what each rewrite does and why it improves accuracy. The symbolic expressions are self-contained and map directly to the source locations listed.

## Validation with validate.py

The `validate.py` script (emitted alongside the report) automates accuracy and performance measurement across Pareto-optimal variants:

```bash
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

1. **Loads a gold reference** from `--gold-path` (MPFR ground truth; see [RAPTOR section](#generating-mpfr-gold-references-with-raptor) below)
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

## Generating MPFR Gold References with RAPTOR

For meaningful accuracy validation, the gold reference should be computed at high precision (MPFR-2048 bits) rather than using the original double-precision output. [RAPTOR](https://github.com/RIKEN-RCCS/RAPTOR) provides this capability by running every floating-point operation through MPFR with garbage collection to avoid OOM on large applications.

### Building RAPTOR

```bash
git clone https://github.com/RIKEN-RCCS/RAPTOR
cd RAPTOR && mkdir build && cd build
cmake .. -DLLVM_DIR=/path/to/llvm -DCMAKE_BUILD_TYPE=Release
make -j
```

RAPTOR requires LLVM >= 15 and MPFR. For LLVM >= 23, two patches are needed in `pass/Raptor.cpp`:
- `#include "llvm/Passes/PassPlugin.h"` -> `#include "llvm/Plugins/PassPlugin.h"`
- Remove the third argument from `createFunctionToLoopPassAdaptor`

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
