# Poseidon

Accelerators concentrate their arithmetic throughput in reduced-precision
units. On an RTX 5090 an FP64 multiply-add runs at 1/64 of the FP32 rate and a
fraction of what the tensor cores deliver, so a scientific kernel written
CPU-first in FP64 and ported as a parallel loop realizes a small share of the
machine. Poseidon is an LLVM pass that takes such a program together with an
accuracy target and rewrites it to run as fast as that target allows: it
profiles the program once on the accelerator to learn each floating-point
instruction's value range, execution count and gradient-weighted sensitivity,
then searches per-instruction precision changes, algebraic rewrites from
Herbie, floating-point expansions, tensor-core raises of scalar reduction
loops, and emulated high-accuracy matrix products (error correction, Ozaki-II),
prices every candidate against a cost model measured on the target device, and
solves for the cheapest assignment whose modelled error stays inside the target.

## Layout

Poseidon is a subproject of the Enzyme repository, built from the same CMake
tree with `-DENZYME_POSEIDON=ON`:

```
enzyme/Poseidon/            the pass (Plugin.cpp and the sources beside it, matmul/)
  runtime/                  profiler and GEMM dispatch runtimes the driver links
  tools/                    poseidon-clang driver, poseidon-calibrate, Herbie platform
  cost_models/              the paper's RTX 5090 model and its Herbie platform
  artifacts/                reproduction packages (see Artifacts)
enzyme/include/poseidon/    the one public header
enzyme/test/Poseidon/       lit suite: check-poseidon, check-poseidon-integration
```

## Requirements

- **LLVM**, with its clang. The plugin is built against one LLVM and loaded by
  that LLVM's clang. This tree is built and tested against LLVM 24
  (`e56c2cefc3e7`, the commit Enzyme's CI pins); the paper's numbers were
  produced with LLVM 23 (`a70419505471`), and the apt packages of LLVM 21 and
  22 build the CPU path.
- **Enzyme**: the one in this tree. Enzyme differentiates the probes the
  profiler places, so it is needed for profile generation; a profile-use
  compile does not load it. The two fixes on the paper path (NVPTX barrier
  operands in the adjoint on LLVM 21+, a TypeAnalysis guard against zero-sized
  globals) are commits on this branch.
- **MPFR** (with GMP): the accuracy model evaluates candidates against an
  arbitrary-precision reference.
- **CUDA** 12.x or 13.x for the GPU path (cuBLAS, and cuSOLVER for the
  eigensolver benchmark). Poseidon builds and runs its CPU tests without CUDA.
- **Herbie**, for the algebraic rewrites. The build compiles upstream
  [herbie-fp/herbie](https://github.com/herbie-fp/herbie) at the commit
  `cmake/Herbie.cmake` pins (Racket 8.15 and Rust are then needed);
  `-DPOSEIDON_HERBIE_BINARY=<path>` reuses an existing one instead. Herbie
  result caches are stamped with the digest of the binary that produced them
  and are searched again under any other.

## Build

```sh
cmake -G Ninja -S enzyme -B build \
    -DCMAKE_C_COMPILER=<llvm-build>/bin/clang \
    -DCMAKE_CXX_COMPILER=<llvm-build>/bin/clang++ \
    -DLLVM_DIR=<llvm-build>/lib/cmake/llvm \
    -DENZYME_POSEIDON=ON
ninja -C build
```

This builds Enzyme's plugins, `build/Poseidon/lib/Poseidon-<ver>.so`, the
runtimes and the tools under `build/Poseidon/{lib,bin}`. The driver finds the
Enzyme plugin of the same build tree. `ninja -C build check-poseidon` runs the
regression tests and `check-poseidon-integration` the ones that compile and
run CUDA.

## Use

Three actions, modelled on clang's PGO. Mark the computation to optimize,
profile it once on a small problem, then compile with a target:

```c
#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE __global__ void step(double *out, const double *in, int n) { ... }

int main() { ...; poseidon_metric("energy drift", drift); }
```

```sh
poseidon-clang++ -poseidon-profile-generate app.cu -o app
./app small_problem                  # writes ./poseidon.profile
poseidon-clang++ -poseidon-profile-use -poseidon-tau=1e-7 app.cu -o app_opt
```

`POSEIDON_OPTIMIZE` makes a kernel an optimization site: the whole kernel body
is the annotated computation, and nothing else in the application changes.
`-poseidon-kernels=<regex>` (or `all`) annotates a library's kernels without
touching its source, and only sites above a share of the profiled cost are
optimized. `poseidon_metric` is optional and declares the one quantity the
accuracy target refers to; the profiling run measures how far it moves when
each site is perturbed, which is what lets the solver spend error where it does
not reach the answer. `poseidon-clang++ -poseidon-help` lists the user flags,
and any other `-poseidon-<flag>` reaches the pass.

**The target is a modelled bound, not a proof.** `-poseidon-tau` is compared
against an error the accuracy model estimates by sampling inputs inside the
profiled ranges and comparing the candidate against a high-precision reference,
weighted by the profiled sensitivity. It is an estimate over a sampled domain
on a profiled workload, not a certified bound over all inputs, and a deployment
whose inputs leave the profiled ranges is outside what was modelled.

## The cost model

Poseidon prices every candidate from a CSV measured on the device the code will
run on; a model measured elsewhere reproduces that device's decisions.
`poseidon-calibrate` measures one, the way `llvm-profdata` prepares a profile:

```sh
poseidon-calibrate --gpu 0            # writes ~/.cache/poseidon/cm_<arch>_<gpu>.csv
poseidon-calibrate --gpu 0 --check    # validate the resolved model against this device
```

A compile that is not given `-poseidon-cost-model=<csv>` looks one up by the
module's target-cpu, in `~/.cache/poseidon` and then in what the installation
ships; zero or several matches abort naming the tool. `cost_models/README.md`
describes the row families and what each calibration arm measures.

## Artifacts

- `artifacts/cgo2027/` reproduces the GPU paper end to end: `./run_artifact.sh`
  (`--quick` for a smoke test, `--check` for the preflight alone) calibrates
  this GPU, runs the benchmarks, redraws every figure and collects the results.
  Its `README.md` is the guide, and each benchmark carries an
  `expected_results.md`.
- `artifacts/cgo2026/` runs two case studies of the CPU paper (the quaternion
  differentiator and the 3x3 eigensolver) through this compiler.
- `artifacts/mfem-workshop-2026/` runs the MFEM elasticity operator of the
  workshop talk; it needs an MFEM build (its `README.md` gives the recipe).

## Citation

    @inproceedings{poseidon-gpu,
      title  = {Floating up the Peak: Automatic Numerical Rewriting Enables Fast
                and Tensorized Accelerator Code},
      note   = {To appear, CGO 2027},
    }

The framework this extends is the CPU Poseidon of CGO 2026, "Thinking Fast
and Correct: Automated Rewriting of Numerical Code through Compiler Augmentation".

## License

Apache License v2.0 with LLVM Exceptions, the repository's `LICENSE`.
