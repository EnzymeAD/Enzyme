# Poseidon

Poseidon is a profile-guided numerical optimizer, built as an LLVM pass plugin
next to Enzyme. Given an accuracy target, it rewrites the floating-point code a
program marks to run as fast as the target allows: algebraic rewrites found by
Herbie, precision changes, floating-point expansions, and on GPUs matrix
products raised to tensor cores or emulated with Ozaki-II. It judges every
candidate on the values a profiling run of the program recorded, with Enzyme
differentiating the probes. It optimizes host code in C and C++ and GPU code
in CUDA and HIP; what each target has is listed under
[How it works](#how-it-works).

## Build

Poseidon needs LLVM 21 or newer, built with clang, and MPFR and GMP. Its
algebraic rewrites run Herbie: `-DPOSEIDON_HERBIE_BINARY=<herbie>` names a
prebuilt one, and `-DPOSEIDON_BUILD_HERBIE=ON` builds it from source, which
takes Rust and Racket. A build without Herbie proposes no algebraic rewrites
and says so in a remark (`-Rpass-missed=poseidon`); asking for them with
`-poseidon-enable-algebraic` is then an error, unless the `-poseidon-cache`
directory holds Herbie's answers from another build.

The CUDA runtimes are built when CMake finds a CUDA toolkit
(`-DPOSEIDON_ENABLE_CUDA=On` requires one, `Off` builds none), for the GPUs of
the machine, or for every major architecture of the toolkit on a machine
without one (`-DCMAKE_CUDA_ARCHITECTURES` chooses others). The HIP runtimes are
built when `-DPOSEIDON_ROCM_PATH=<rocm>` names a ROCm installation, for the
processor `-DPOSEIDON_AMDGPU_ARCH` names (default `gfx942`); the
error-corrected GEMM runtime when `-DPOSEIDON_CUMPSGEMM_ROOT=<checkout>` names
a built cuMpSGEMM.

```sh
cmake -G Ninja -S enzyme -B build \
    -DLLVM_DIR=<llvm-build>/lib/cmake/llvm \
    -DENZYME_POSEIDON=ON
ninja -C build
ninja -C build check-poseidon
```

The build installs the drivers `poseidon-clang` and `poseidon-clang++`, the
tools `poseidon-calibrate`, `poseidon-kappa` and `poseidon-explain`, the header
`poseidon/poseidon.h`, the runtimes and a CMake package.

## Use

### 1. Mark the code to optimize

```c
#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE void step(const double *x, double *y, int n) { ... }
POSEIDON_OPTIMIZE_TAU(1e-10) __global__ void force(const double *p, double *f) { ... }

double r = __poseidon_optimize(energy, x, n); /* this call alone */
```

A site is a function marked `POSEIDON_OPTIMIZE` (a host or device function, a
kernel or a lambda), or one call made through `__poseidon_optimize`. The
`_TAU` forms give a site an accuracy target of its own.
`-poseidon-kernels=<regex>` makes every matching kernel a site without a
change to the source.

### 2. Measure the device

Candidates are priced from a cost model of the device the program runs on. No
model ships with Poseidon: each host CPU and GPU is measured once, and a
compile finds its model by the module's target CPU.

```sh
poseidon-calibrate --cpu native   # this host, for compiles with -march=native
poseidon-calibrate --cpu x86-64   # this host, for compiles without -march
poseidon-calibrate --gpu 0        # GPU 0, CUDA or HIP
```

The models go to `~/.cache/poseidon`. Measuring a host CPU takes the Python
package `llvmlite`. [tools/calibrate/README.md](tools/calibrate/README.md)
describes the model and every measurement.

### 3. Profile

```sh
poseidon-clang -O2 -poseidon-profile-generate app.c -o app
./app small-input
```

The run writes the profile of each site to `./poseidon.profile`, to the
directory `-poseidon-profile-generate=<dir>` names, or to the one
`POSEIDON_PROFILE_DIR` names when the program runs. A small problem is
enough when its values are of the kind the large ones compute. A program
that declares its result with `poseidon_metric(name, value)` runs under
`poseidon-kappa -- ./app small-input` instead, which measures each site's
condition number against that result in 1 + 3 x <sites> more runs. Profiling
host code needs glibc.

### 4. Optimize

```sh
poseidon-clang -O2 -poseidon-profile-use -poseidon-tau=1e-8 app.c -o app_opt
poseidon-clang++ -O2 -poseidon-profile-use -poseidon-tau=1e-8 app.cu -o app_opt
```

`-poseidon-tau` is the relative accuracy target of every site without its own.
It bounds a site's modelled relative error, a sum over the parts of the site:
a matrix product contributes its relative output error (none as written), an
FP subgraph its output error weighted by the profiled sensitivities and
divided by the site's summed sensitivity. The cheapest set of rewrites that
keeps the sum within the target is applied. The set is solved exactly;
`-poseidon-early-prune` solves an approximation instead, faster on sites with
many units, which drops a partial set for a cheaper, no less accurate one that
the later choices admit alike.
`-poseidon-cost-budget=<share>` instead bounds the share of a site's profiled
cost a rewrite may add, or with a negative share has to save. Herbie results
and solved tables are kept in `./.poseidon` (`-poseidon-cache=<dir>`), so a
repeated compile does not search again.

### 5. See what changed

`-poseidon-explain=<dir>` writes `<function>.explain.json` for every site: what
its solve changed, considered and rejected, at source locations.
`poseidon-explain <dir>` prints it as a report, and with `--format source` or
`diff` as the source with the applied rewrites in it, placed through the AST of
the clang `poseidon-clang++` runs (`--clang` names another). `-Rpass=poseidon`
reports the decisions as compiler remarks.

`poseidon-clang -poseidon-help` lists the common flags, and
`poseidon-clang -mllvm -help-hidden -c -x c /dev/null` every `-poseidon-<flag>`.

### CMake

A project that does not compile through `poseidon-clang` links the interface
target of an action after `find_package(Poseidon CONFIG)`:
`Poseidon::ProfileGenerate`, `Poseidon::ProfileUse` or `Poseidon::Plain`. Each
loads the plugins and links the runtimes of the action; a profile-use build
names its profile and target through `POSEIDON_MLLVM`:

```cmake
find_package(Poseidon CONFIG REQUIRED)
target_link_libraries(app PRIVATE Poseidon::ProfileUse)
target_compile_options(app PRIVATE
  "SHELL:${POSEIDON_MLLVM} -poseidon-profile-use=${PROFILE_DIR}"
  "SHELL:${POSEIDON_MLLVM} -poseidon-tau=1e-8")
```

## How it works

### Targets

| Technique | Host | NVPTX (CUDA) | AMDGPU (HIP) |
|---|---|---|---|
| Algebraic rewrites (Herbie) | yes | yes | yes |
| Precision changes, floating-point expansions | yes | yes | yes |
| Matrix product raised to tensor cores in the kernel | no | yes | no |
| Matrix product as a library call: native, Ozaki-II, direct | no | yes | yes, without TF32 |
| Matrix product as a library call: TCEC (cuMpSGEMM) | no | when built with it | no |

The host has no matrix-product technique. Profiling records host sites with
the host profiler and device sites with the device profiler, whose probes the
plugin links into each device module it compiles for profile generation.

### Source layout

| Folder | What it holds |
|---|---|
| `Core/` | What the rest is written against: the expression graph of a site and its FPCore and C forms (`FPNode`, `FPCore`), the number formats (`NumberFormat`), the compile of a module and the states it holds for the later layers (`Session.h`), the optimization unit the solver chooses among (`Unit.h`), and a site's solve and what a technique provides the pass (`Technique.h`) |
| `Model/` | The models every technique shares: the measured cost table and the cost of code over it, the sampling of a subgraph's inputs, the evaluation of an expression graph at a point, in floating point and exactly, and the unit over one FP subgraph priced on its samples (`SubgraphUnit.h`) |
| `Techniques/` | One folder per technique, each with its own pricing and accuracy model: `Algebraic/` (Herbie), `Precision/` (narrowing and floating-point expansions), `Matmul/` (matrix products on tensor cores or as library calls); `Techniques.cpp` lists them |
| `Solve/` | A site's FP subgraphs, the collection and pricing of its units, the DP over them (`Solve.h`), pins, and the explanation of the result |
| `Sites/`, `Profile/` | Finding and outlining sites, their canonical form and launch manifest; profile generation and the reading of profiles |
| `Pass/`, `Targets/`, `Support/` | The pass pipeline and its options; what the pass knows of each target; shared helpers |
| `runtime/`, `tools/` | The libraries a compiled program links (profilers, GEMM runtimes); `poseidon-clang`, `poseidon-calibrate`, `poseidon-explain`, `poseidon-kappa` |

The units are an algebraic unit, a precision unit or a matmul unit, one kind
per technique; the solver sees each as an optimization unit whose candidates
carry a cost change and an error change. A technique joins the pass as a
subclass of `Technique` in its own folder, which the list in
`Techniques/Techniques.cpp` names: the kind of its units, the option that turns
it on, and the steps of a compile it takes part in, from proposing units for a
site to rewriting the host half; a step it does not override does nothing.
What it keeps for a site or a compile is a state of its own type, which the
site or the session holds in a slot, made on first use. The session holds the
states of the other layers the same way: the site records, the launch manifest,
the cost model and the pins.
Each unit writes its own part of the explanation of a solve; the explanation
(`Solve/Explain.cpp`) words a remark by the kind the unit gives a candidate,
one of the kinds its schema lists.

### Passes

The plugin adds its passes at five extension points of the new pass manager.
At PipelineStart, before the inliner can fold a site into its callers, a
`POSEIDON_OPTIMIZE` host function is outlined as a site, a device function's
site is handed to the kernels that call it, and the launch stubs of the host
half of a CUDA or HIP compile are rewritten from the launch manifest (below).
At OptimizerEarly, in a module with a site, the canonical prefix runs first:
the start of Enzyme's own pipeline, so that the profile-generate and the
profile-use compile number a site's instructions alike. The site pass follows
(profile generation instruments each site; profile use builds its units,
prices and models them, solves, and rewrites), then the always-inliner, since
nothing later folds the outlined bodies and the rewritten clones back. At
OptimizerLast a deferred site, whose body is a derivative Enzyme produces, is
handled once Enzyme has lowered that request. Under profile generation Enzyme
then runs once more, by its pipeline name, to lower the derivative request the
instrumented site emits; in both compiles the function simplification pipeline
runs again over the kernels that carried the site's marker, which kept its
arguments in memory until then. The finalizing pass then checks that no
derivative request is left, links the device profiler, writes the launch
manifest, and builds the output from the module as written. An LTO link runs
the same passes at the full-LTO extension points. For `opt`, the pipeline names
are `poseidon`, `poseidon-deferred-sites` and `poseidon-finalize`.

The module as written is a copy of the module taken at PipelineStart in a
profile-use compile with a site. The canonical prefix, the outlined site bodies
and, on a GPU, math kept out of line change the code of every function of the
module, not only of those a rewrite reached. The copy is compiled with the
default pipeline as a compile without a profile, and only the functions a
rewrite reached are taken from the rewritten module into it, so every other
function is the code a compile without Poseidon emits. There is no such copy
under profile generation, in the host half of a GPU compile, in a joint solve
(`-poseidon-solve-jointly`), or at an LTO link.

### Device and host halves

clang compiles a CUDA or HIP source in two processes, the device half first.
Where the host half has to change a kernel launch (under profile generation, to
launch a profiled kernel with the shadows of its pointer arguments; under
profile use, to replace a matrix product's launch with a library call), the
device half hands the launches of its sites to the host half in a small
manifest. `poseidon-clang` keeps it in a temporary directory that lives as
long as the compile. A build that compiles the two halves in separate
invocations finds it in the `-poseidon-cache` directory, or in the one
`-poseidon-launch-dir` names; both invocations then run on one machine.

### The cache directory

`-poseidon-cache=<dir>`, by default `./.poseidon` in the working directory of
the compile, holds Herbie's answers, each site's solved table and the
matrix-accuracy estimates, each filed under a digest of what it was computed
from, so that a changed input misses it. A compile thus reads files that are
not on its command line, the profile among them: a compiler cache such as
ccache, which keys an object by its command line and source, can serve an
object compiled against another profile, and a distributed build runs the
compile where neither the profile nor the cache directory is. Poseidon's
compiles are therefore run without either.

Herbie's searches are bounded by counts, the `-poseidon-herbie-num-*` flags,
which are part of the key, and by wall-clock limits, its per-core `--timeout`
(`-poseidon-herbie-timeout`) and `-poseidon-herbie-subgraph-timeout`, which are
not. A compile on a loaded machine can reach a limit and get fewer rewrites for
itself. An answer a limit cut short is never cached, so the next compile asks
Herbie again; the table solved from it is filed under its candidates and
serves only a compile that gets the same answers.

### Environment

| Variable | Read by | Meaning |
|---|---|---|
| `POSEIDON_PROFILE_DIR` | profiler runtime, `poseidon-kappa` | Where a profiling run writes its profiles and `metric.txt`; default `./poseidon.profile` |
| `POSEIDON_PROFILE_HEAP_MB` | device profiler | Device heap for the profiled kernels' reverse passes; default 4096 |
| `POSEIDON_KAPPA` | profiler runtime | `0` declines the condition numbers of a declared metric; `child` marks a run of the measurement (set by `poseidon-kappa`) |
| `POSEIDON_KAPPA_LADDER` | profiler runtime, `poseidon-kappa` | The relative perturbations of the measurement; default `1e-8,1e-6,1e-4` |
| `POSEIDON_KAPPA_SITE`, `POSEIDON_KAPPA_EPS`, `POSEIDON_KAPPA_SITES` | profiler runtime | Set by `poseidon-kappa`: the site a run perturbs and by how much, and the file a profiling run lists its sites in |
| `POSEIDON_COST_MODEL` | plugin | The cost model when no `-poseidon-cost-model` names one |
| `XDG_CACHE_HOME`, `HOME` | plugin, `poseidon-calibrate` | The user cache of cost models, `$XDG_CACHE_HOME/poseidon` or `~/.cache/poseidon` |
| `POSEIDON_PLUGIN`, `POSEIDON_ENZYME_PLUGIN` | `poseidon-clang` | The plugins to load in place of those of the build |
| `POSEIDON_ECHO` | `poseidon-clang` | Prints the clang command it runs |
| `HIP_PATH`, `ROCM_PATH` | `poseidon-clang` | The ROCm installation of a HIP compile without `--hip-path` or `--rocm-path` |
| `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` | `poseidon-calibrate` | The GPU to measure without `--gpu` |
| `TMPDIR` | `poseidon-kappa`, plugin | Where the runs of the measurement keep their scratch directories, and a compile the input files and output directories of its Herbie runs; default `/tmp` |

### File formats

| File | Written by | Read by | Specified in |
|---|---|---|---|
| Profile, `<site>.fpprofile` and `metric.txt` | profiler runtime; `poseidon-kappa` adds the condition number | plugin | `runtime/profile/ProfileFormat.h` |
| Cost model, `cm_<arch>_<device>.csv` and its Herbie platform | `poseidon-calibrate` | plugin | `tools/calibrate/README.md` |
| Explanation, `<function>.explain.json` | plugin (`-poseidon-explain`) | `poseidon-explain`, `poseidon-calibrate` | `Solve/Explain.h` |
| Launch manifest | device half of a GPU compile | host half | `Sites/LaunchManifest.h` |
| Herbie answer, in the cache directory | plugin | plugin | `Techniques/Algebraic/HerbieRun.cpp` |
| Solved table (DP), in the cache directory | plugin | plugin | `Solve/SolveSites.cpp`, `Support/CacheKey.h` |

### Platform

Poseidon is built and tested on Linux. Profiling host code links a library
that replaces `malloc` and its family for the whole process, to know the
extent of every allocation a site's pointer reaches; it calls glibc's own
allocator underneath, so it takes glibc. A profiling run that declared a
metric names the command to measure it from `/proc/self/cmdline`, and
`poseidon-kappa` runs the program with POSIX `fork` and `exec`. The GPU
runtimes and profilers take CUDA or ROCm; a HIP build is for one processor.

## Third-party code and references

### Code under other licenses

These passages are a subset of another project. Each file says so in its
header, with the project's license text.

| Code | Location | Source | License | Relation |
| --- | --- | --- | --- | --- |
| Two-component expansion routines | `Techniques/Precision/ExpansionRoutines.h` | [MultiFloats.jl](https://github.com/dzhang314/MultiFloats.jl), commit 28041a3 | MIT | Translated from Julia, operation for operation |
| Three- and four-component expansion routines, two-component division | `Techniques/Precision/ExpansionRoutines.h` | [mX_real](https://github.com/RIKEN-RCCS/mX_real) (its QxW routines), commit 72a35d6 | BSD 2-Clause | Translated, operation for operation |
| Herbie platform declarations | `tools/calibrate/herbie_platform.py` | [Herbie](https://github.com/herbie-fp/herbie) `src/platforms/c.rkt`, commit 73ba1fe | MIT | The declarations' form, with the cost model's costs and operations |
| Reference routines of a test | `test/Poseidon/Inputs/expansion_match/multifloats_subset.h` | MultiFloats.jl, commit 28041a3 | MIT | Translated from Julia, statement for statement |
| Reference routines of a test | `test/Poseidon/Inputs/expansion_match/mx_real_subset.h` | mX_real, commit 72a35d6 | BSD 2-Clause | QxW routines verbatim; mX_real's operator dispatch condensed |

The expansion routines are compiled into the plugin, so a binary distribution
of it carries mX_real's notice, as its license's second clause asks.

### Published algorithms

These reimplement a published algorithm; no code is taken from the reference
implementation named.

| Code | Location | Algorithm | Reference implementation |
| --- | --- | --- | --- |
| Ozaki-II GEMM runtime | `runtime/gemm/OzakiII.cu`, `runtime/gemm/SchemeConstants.h` | K. Ozaki, Y. Uchino, T. Imamura, [arXiv:2504.08009](https://arxiv.org/abs/2504.08009); fast-mode scaling as in GEMMul8 v3.1.1, which attributes it to S. Kawakami, D. Takahashi, [arXiv:2606.29129](https://arxiv.org/abs/2606.29129) | [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) v3.1.1 (MIT): its constants and residue method, computed from their definitions, so that results agree bit for bit |
| Host simulation of the GEMM runtimes | `Techniques/Matmul/Simulation.cpp` | The Ozaki-II runtime above; TCEC, H. Ootomo, R. Yokota, IJHPCA 36(4), 2022 | GEMMul8 v3.1.1, cuMpSGEMM (MIT): their arithmetic, modeled |
| In-kernel split scheme | `Techniques/Matmul/RealizeInKernel.cpp` | The two-slice split of S. Markidis et al. (IPDPSW 2018), with the residual scaled as Ootomo and Yokota scale it, generalized to N slices | none |

### Dependencies

| Dependency | License | Used for | Needed |
| --- | --- | --- | --- |
| LLVM and Clang | Apache-2.0 WITH LLVM-exception | The pass plugin's IR and the drivers | Always |
| Enzyme | Apache-2.0 WITH LLVM-exception | Differentiating the profiling probes, as a co-loaded plugin | To build; to profile |
| MPFR | LGPL-3.0-or-later | The accuracy model's high-precision reference; linked into the plugin | Always |
| GMP | LGPL-3.0-or-later or GPL-2.0-or-later | MPFR's arithmetic; linked into the plugin | Always |
| Herbie | MIT | Algebraic rewrites; run as a separate process, with FPCore in and out | Optional: without it, no algebraic rewrites |
| CUDA toolkit: CUDA runtime, cuBLAS | NVIDIA CUDA Toolkit EULA | The CUDA runtimes and profiler, linked | For CUDA targets |
| ROCm: HIP runtime, hipBLAS | MIT | The HIP runtimes and profiler, linked | For HIP targets |
| cuMpSGEMM | MIT | The TCEC GEMM runtime, linked | Optional (`POSEIDON_CUMPSGEMM_ROOT`): without it, no TCEC candidate |
| llvmlite | BSD 2-Clause | `poseidon-calibrate`'s host CPU microbenchmark | For host CPU cost models |
| Python 3 | PSF-2.0 | `poseidon-calibrate`'s scripts, `poseidon-explain`, the tests | For those tools |

## License

Apache License v2.0 with LLVM Exceptions, the repository's `LICENSE`, except
the passages listed under
[Code under other licenses](#code-under-other-licenses).
