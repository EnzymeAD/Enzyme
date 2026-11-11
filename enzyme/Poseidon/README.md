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
enzyme/Poseidon/            the pass: the sources beside and under Pass/ (the
                            plugin entry Pass/Plugin.cpp), Core/, Sites/,
                            Profile/, Model/, Techniques/ and Targets/
  runtime/                  profiler and GEMM dispatch runtimes the driver links
  tools/                    poseidon-clang driver, poseidon-calibrate, poseidon-explain,
                            Herbie platform, Compiler Explorer
enzyme/include/poseidon/    the one public header
enzyme/test/Poseidon/       lit suite: check-poseidon, check-poseidon-integration
```

## Requirements

- **LLVM**, with its clang. The plugin is built against one LLVM and loaded by
  that LLVM's clang. This tree is built and tested against LLVM 24
  (`9d618fc02431`, the commit Enzyme's MLIR CI pins), and the apt packages of
  LLVM 21 and 22 build the CPU path.
- **Enzyme**: the one in this tree. Enzyme differentiates the probes the
  profiler places, so profile generation needs its plugin; profile use and
  plain compiles need it only for Enzyme requests of their own, such as the
  body of a deferred site. `poseidon-clang++` loads it for every action: a
  module that asks nothing of Enzyme compiles as it does without it.
- **MPFR** (with GMP): the accuracy model evaluates candidates against an
  arbitrary-precision reference.
- **CUDA** 12.x or 13.x for the GPU path (cuBLAS). Poseidon builds and runs
  its CPU tests without CUDA; without a CUDA compiler at configure time it has
  no device profiler for NVIDIA GPUs.
- **ROCm**, for HIP, optional: `-DPOSEIDON_ROCM_PATH=<rocm>` with an LLVM that
  builds the AMDGPU target adds the AMDGPU device profiler, the HIP runtimes
  and calibrators, for the one processor `-DPOSEIDON_AMDGPU_ARCH` names
  (default `gfx942`).
- **cuMpSGEMM**, optional: `-DPOSEIDON_TCEC_USE_CUMPSGEMM=ON
  -DPOSEIDON_CUMPSGEMM_ROOT=<checkout>` makes the error-corrected GEMM runtime
  delegate to that build of cuMpSGEMM instead of its built-in cuBLAS kernels.
- **Herbie**, for the algebraic rewrites. The build compiles upstream
  [herbie-fp/herbie](https://github.com/herbie-fp/herbie) at the commit
  `cmake/Herbie.cmake` pins (Racket 8.15 and Rust are then needed);
  `-DPOSEIDON_HERBIE_BINARY=<path>` reuses an existing one instead, and
  `-poseidon-herbie-binary=<path>` names one for a compile. Herbie results are
  cached under the digest of the binary that produced them and are searched
  again under any other, unless `-poseidon-import-herbie-answers=<dir>` names a
  cache directory whose answers are then replayed whichever binary gave them.
  `-poseidon-herbie-num-taylor-terms=<n,...>` searches every expression once
  per series length and takes the union of the answers; it needs a Herbie
  that takes `--num-taylor-terms`, which the pinned commit does not, and a
  compile given the flag with a Herbie that does not stops with an error.

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
profile it once on a small problem, then compile with a target. A function on
the host and a kernel on the GPU are marked the same way:

```c
#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE void kick(double *v, const double *a, double dt, int n) { ... }
POSEIDON_OPTIMIZE __global__ void force(double *a, const double *x, int n) { ... }
POSEIDON_OPTIMIZE_TAU(1e-12) __global__ void advance(double *x, const double *v, int n) { ... }

int main() { ...; poseidon_metric("energy drift", drift); }
```

```sh
poseidon-clang++ -poseidon-profile-generate app.cu -o app
./app small_problem                  # writes ./poseidon.profile
poseidon-clang++ -poseidon-profile-use -poseidon-tau=1e-7 app.cu -o app_opt
```

A CUDA program is profiled as it is built otherwise, one translation unit at a
time or several at once, with or without relocatable device code, and its
units may sit in static libraries: the plugin links the device half of the
profiler into every device module it compiles for profile generation, and
the link takes the host half from `libposeidon_profile_cuda`.

The ways to opt in, simplest first:

1. `POSEIDON_OPTIMIZE` on a function or kernel, and `-poseidon-tau=<t>` on the
   optimizing compile. Every call of the function is the site: its whole body
   is profiled and rewritten, and nothing else in the application changes.
   Start here.
2. `POSEIDON_OPTIMIZE_TAU(t)`: the same with that site's own target, which wins
   over `-poseidon-tau`. For sites that need different accuracy. `t` may be a
   literal, a macro or a constant expression.
3. `__poseidon_optimize(fn, args...)`, or `__poseidon_optimize_tau(t, fn,
   args...)` with that call's own target: one call instead of every call. The
   call has its own profile, target and rewritten copy of `fn`, so two calls of
   one function can be rewritten differently, and other calls of `fn` run it as
   written; no shadows are needed. `fn` names a function defined in the same
   translation unit, on the host or in device code, and `t` is a compile-time
   constant.
4. `-poseidon-kernels=<regex>|all`: GPU kernels opted in without editing the
   source (library code); only the ones above `-poseidon-min-cost-share` of the
   profiled cost are optimized.
5. `poseidon_metric(name, value)`, once at the end of the profiling run: the
   quantity the target refers to, so the solver can spend error where it does
   not reach that quantity. The profiling run measures how far it moves when
   each site is perturbed. Optional.
6. `__poseidon_fp_optimize(fn, args...)`: one call with explicit Enzyme
   activities and shadows; see [Advanced: one call with shadows](#advanced-one-call-with-shadows).

```c
double e = __poseidon_optimize(energy, x, n);       /* this call is a site */
__poseidon_optimize_tau(1e-12, kick, v, a, dt, n);  /* with its own target */
```

Precedence: the site's own target (`POSEIDON_OPTIMIZE_TAU`,
`__poseidon_optimize_tau`, or `poseidon_tau` in the call) > `-poseidon-tau` >
none. The target is the same number wherever
it is written: moving it from the flag into the source never changes what a
site emits, and changing it needs no new profile, also when a template argument
carries it to the call. A site with no target is solved against
`-poseidon-cost-budget` instead, the share of what the profiled run spent in
the site that a solution may add (a negative share, down to -1, is one it has
to save: -0.25 requires saving a quarter); at its default of 0 that is the most
accurate rewrite costing no more than the code as written, which may be the
code as written. A share below -1 or one that is not finite stops the compile,
and a solve writes the shares of its frontier points to `budgets.txt` in its
`-poseidon-cache` directory.
`poseidon-clang++ -poseidon-help` lists the user flags, and any other
`-poseidon-<flag>` reaches the pass.

A marked host function, or one called through `__poseidon_optimize`, returns
nothing, a floating-point scalar or an integer, and takes pointers,
floating-point scalars and integers; any other signature is left as written
with a warning. Its profiling build sizes the shadow of every pointer argument
from the allocation it points into: any heap allocation in the process (the
driver links a profiling runtime that records them), and the globals and stack
arrays of the translation units that hold a site. A pointer into anything else
stops the profiling run, naming the function and the argument. A profile is
read back by a compile at any -O level as long as the site's code has the same
canonical form there; when it does not (for example `std::sqrt` is a libm call
at -O1 and an intrinsic at -O2), the compile says so and asks for a new
profile.

In device code the pointers a call made through `__poseidon_optimize` passes
point into local objects of the calling kernel or device function (a
`tensor<real_t, 3, 3> &` input, a `real_t *` output) or into an aggregate it
takes by value (a kernel's closure, a `__grid_constant__` parameter). The
profiling build gives each such object a shadow of its own, a local object of
its type, zero, with 1.0 in what the call writes when the program may read it
after the call, at every execution of the call; the call itself still reads
and writes the object, and nothing else is stored through it. An object that
holds pointers, and the address of an object the call never reads (an empty
functor's), are passed as constants, and a call that passes a pointer into
device memory runs as written with a warning. A `fn` whose body is an
Enzyme request (`__enzyme_fwddiff` and the like) is profiled and rewritten in
the code Enzyme generates for it. When the solve leaves every call site of a
device compilation alone, the compilation emits the code as written: its
module is compiled once more as without a profile.

The accuracy target is empirical, not a formally verified bound.
`-poseidon-tau` is compared against an error the accuracy model estimates by
sampling inputs inside the profiled ranges and comparing the candidate against
a high-precision reference, weighted by the profiled sensitivity. It is an
estimate over a sampled domain on a profiled workload, not a certified bound
over all inputs, and a deployment whose inputs leave the profiled ranges is
outside what was modelled.

### Advanced: one call with shadows

`__poseidon_fp_optimize(fn, args...)` makes one call the site like
`__poseidon_optimize`, for pointers the profiler cannot size or a custom seed.
The arguments are `fn`'s, each pointer followed by a shadow of the same size
that the caller allocates, zeroed for inputs and filled with 1.0 for outputs,
as Enzyme's `enzyme_dup` takes them; `poseidon_tau, t` among them sets that
call's target:

```cpp
template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
extern int enzyme_dup, poseidon_tau;

__poseidon_fp_optimize<void>((void *)kernel, poseidon_tau, 1e-10,
                             enzyme_dup, x, dx, enzyme_dup, out, dout, n);
```

A function that carries `POSEIDON_OPTIMIZE` is a site at every call already,
and naming it in `__poseidon_optimize` or `__poseidon_fp_optimize` as well is
a compile error.

### CMake

A project compiled with the clang Poseidon is built against uses it through
`find_package(Poseidon CONFIG)`, from the build tree (`Poseidon_DIR` is
`build/Poseidon`) or an installation. One interface target per action carries
what `poseidon-clang++` adds for it (`cmake/PoseidonConfig.cmake.in` lists
where the two differ):

```cmake
find_package(Poseidon 0.1 CONFIG REQUIRED)
target_link_libraries(app PRIVATE Poseidon::ProfileGenerate)  # or Poseidon::Plain
# or: Poseidon::ProfileUse, with the profile it reads
target_compile_options(app PRIVATE "SHELL:${POSEIDON_MLLVM} -poseidon-profile-use=${PROFILE}")
```

Any other `-poseidon-<flag>` is passed as `"SHELL:${POSEIDON_MLLVM}
-poseidon-<flag>"`, that is `-Xclang -mllvm -Xclang -poseidon-<flag>`, which
clang hands to its compilers and to no linker; `poseidon-clang++` passes every
Poseidon flag so.

## The cost model

Poseidon prices every candidate from a CSV measured on the device the code will
run on; a model measured elsewhere reproduces that device's decisions. No model
ships with Poseidon. `poseidon-calibrate` measures one, the way `llvm-profdata`
prepares a profile:

```sh
poseidon-calibrate --gpu 0            # writes ~/.cache/poseidon/cm_<arch>_<gpu>.csv
poseidon-calibrate --gpu 0 --check    # validate the resolved model against this device
```

A CPU is measured by `tools/calibrate/microbm_cpu.py --native-arch
<target-cpu> --out cm_<target-cpu>_<name>.csv` (Python 3 with `numpy` and
`llvmlite`). A compile that is not given `-poseidon-cost-model=<csv>` (or
`POSEIDON_COST_MODEL`) looks one up by the module's target-cpu in
`$XDG_CACHE_HOME/poseidon`, by default `~/.cache/poseidon`, where
`poseidon-calibrate` writes; no match stops the compile naming the tool, and
several stop it naming the files. `tools/calibrate/README.md` describes the
row families and what each calibration arm measures.

## What a compile decided

Poseidon reports through LLVM's optimization remarks under the pass name
`poseidon`, at the source location of the code concerned:

- `-Rpass=poseidon`: each applied rewrite, as C, with the technique (series
  approximation and the profiled range it was validated on, regime split,
  operations moved to another precision, matrix-product scheme), the change of
  the modelled error and of the modelled cost, and its id.
- `-Rpass-missed=poseidon`: each subgraph left as written and why, and at most
  three rejected alternatives per subgraph with the reason (modelled error
  above the target, dominated, not cheaper, Herbie returned nothing, ...).
- `-Rpass-analysis=poseidon`: the site's target, modelled error and cost
  before and after, and each subgraph's profile (input ranges, executions,
  sensitivity).

`-fsave-optimization-record` writes the same remarks as YAML, with the
expression, rewrite, error and cost as separate arguments.

`-poseidon-explain=<dir>` (also implied by `-poseidon-report-path`) writes
`<func>.explain.json` per site, schema `poseidon-explain/1`; a name over 200
characters is cut to its first 160 and a hash of the whole, as a profile's file
name is. Every number in it is the solver's own; a field the solve did not compute (for example a modelled
error when the DP table came from the cache) is omitted.

| field | content |
|---|---|
| `function` | `name` (demangled), `symbol`, `ir` (the solved clone), `site` and `loc` locations |
| `target` | `tau` and `tau_from` (`site` or `flag`), or `budget`; `cost_model`, `selector` |
| `result` | `cost_before`, `cost_after`, `modelled_rel_error_as_written`, `modelled_rel_error`, `applied` (ids) |
| `subgraphs[]` | `id`, `kind` (`subgraph` or `matmul`), `loc`, `cost`, `executions`, `ops[]` (`opcode`, `type`, `loc`, `executions`, `sensitivity`, `gradient`, `min`, `max`), `inputs[]` (`var`, `name`, `loc`, `type`, `min`, `max`), `outputs[]` (`name`, `loc`, `fpcore`, `c`, the `ops` and `inputs` it depends on), `original` (`fpcore`, `c`), `herbie[]` (per core: precision, status), `left_as_written` |
| `subgraphs[].alternatives[]` | `id` (what `-poseidon-apply-rewrites` takes), `kind` (`algebraic`, `precision`, `expansion`, `matmul`), `summary`, `fpcore`, `c`, `approx`, `approx_note`, `changes`, `cost_delta`, `modelled_error`, `herbie_error_bits` (`before`, `after`), `status` (`applied`, `pareto`, `rejected`), `reason` |
| `pareto[]` | `budget` (what `-poseidon-cost-budget` takes), `cost`, `modelled_rel_error`, `applied`, `selected` |

Locations are `{file, line, col}`; modelled errors are relative and in the
unit `-poseidon-tau` is given in; costs and budgets are shares of what the
profiled run spent in the solved sites. An applied algebraic alternative reads

```json
{"id": "R0_1", "kind": "algebraic", "precision": "binary64", "status": "applied",
 "fpcore": "#s(approx (/ (- 1 (cos v0)) (* v0 v0)) #s(literal 1/2 binary64))",
 "c": "0.5", "approx": true,
 "approx_note": "series approximation of (1.0 - cos(x[i])) / (x[i] * x[i])",
 "cost_delta": -3576832, "modelled_error": 1.16e-16,
 "reason": "part of the cheapest solution that meets the target"}
```

`tools/explain/poseidon-explain <file|dir> --format F` (Python 3, installed in
`bin/`) presents these files: `report` (a table per source location with the
alternatives, the profile and the Pareto points with the flag that selects
each), `source` (the source with each rewrite spliced in as C, using clang's
AST to find the expression), `diff`, `remarks` (`file:line:col: note:` lines),
`fpcore` (before and after, for Herbie) and `json`.

## Errors

A mistake in what a compile is given (a flag value, a file that is missing or
unreadable, a cost model or a profile that does not fit the compile) is an
error, `error: Poseidon: ...`, that says what to do, at the source location of
the site concerned where there is one. The compile exits with status 1 and
writes no output, without clang's crash report (no request for a bug report,
stack dump or reproducer files). Where the site can be left as written the
compile goes on and reports the other sites' errors too; a mistake that leaves
nothing to compile with, such as a flag value, a missing profile directory or a
cost model that cannot price the compile, ends it at once. A broken internal
invariant of Poseidon stops the compile as a crash, to be reported as a bug.

## Compiler Explorer

`tools/explorer/` puts Poseidon in Compiler Explorer. `poseidon-ce.sh` is the
compiler it calls: it builds the program with profiling, runs it once, and
compiles it again under `-poseidon-tau` (default 1e-6); every other
`-poseidon-<flag>` reaches the pass. The `Dockerfile` builds a standalone
explorer from the release tarball the `Poseidon Explorer` workflow publishes
(`nightly-poseidon`), with C++ only, the example `examples/c++/` and plain
clang next to Poseidon for comparison.

The tarball carries no cost model: the machine that runs the explorer measures
its own once, with `calibrate/microbm_cpu.py` from the tarball (or
`tools/calibrate/`; it needs `numpy` and `llvmlite` and runs for hours), and
the explorer is given it explicitly:

```sh
python3 microbm_cpu.py --native-arch x86-64 --out cm_x86-64_explorer.csv
cd enzyme/Poseidon/tools/explorer
POSEIDON_COST_MODEL=<dir>/cm_x86-64_explorer.csv docker compose up -d --build   # http://localhost:10240
```

`--build-arg POSEIDON_TARBALL=<url>` takes another tarball. To run a local
build instead, lay it out with `package.sh <build> <cost-model.csv|-> <dir>`
and mount it over the downloaded one; its LLVM must be one the image has a
clang for (`LLVM_VERSION`, default 21):

```sh
docker run -d -p 10240:10240 -v <dir>:/opt/compiler-explorer/poseidon:ro \
  -v <csv>:/opt/compiler-explorer/poseidon/cost_models/cm_x86-64_explorer.csv:ro poseidon-explorer
```

Without a cost model every Poseidon compile stops with an error that names
the missing file; none is substituted.

The explorer compiles for a
generic x86-64 CPU with a cost model measured for one, so it shows host
functions marked `POSEIDON_OPTIMIZE` and calls made through
`__poseidon_optimize`; GPU kernels need a local build. The examples cover the
marker, per-function targets, per-call sites and a regime split, and each
prints the values of an unmarked copy next to Poseidon's. Compile with `-O1` or
higher. Herbie results are kept in `POSEIDON_CE_CACHE`
(default `$TMPDIR/poseidon-ce-cache`, `/tmp` without `TMPDIR`) and reused by
later compiles; `package.sh` ships the examples' results so their first
compile does not search. The
defaults suit a shared machine: at most `POSEIDON_CE_MAX_SEARCHES` (2)
compiles profile and search at once, and one that finds both slots taken for
3 seconds is compiled as written with a note to try again; a search gets
`POSEIDON_CE_SOLVE_SECONDS` (60), two Herbie workers and
`POSEIDON_CE_SOLVE_MEMORY_MB` (4096) of address space per process, and past
either limit the program is compiled as written. The profiling run, which is
the user's program outside the explorer's sandbox, gets
`POSEIDON_CE_PROFILE_SECONDS` (10) and `POSEIDON_CE_PROFILE_MEMORY_MB` (1024).
`test-explorer.py --url http://localhost:10240` checks a running explorer
through its REST API.

The explorer calls the wrapper as `clang++`, so the Poseidon compiler has every
view plain clang has: LLVM IR, optimization remarks, the optimization pipeline
(Poseidon's own passes included), control flow graph, AST, preprocessor output,
stack usage and the linked binary. Flags that only shape that output reach the
final compile, not the profiling build. The compiler output keeps the solver's
decisions and the remarks (`-Rpass=poseidon -Rpass-analysis=poseidon` are on),
which the editor marks as notes; `-poseidon-print` shows everything. The
wrapper leaves the compile's `<func>.explain.json` next to its output (unless
`POSEIDON_CE_EXPLAIN=0`), and the tools Poseidon report, Poseidon source,
Poseidon diff and Poseidon FPCore show it through `poseidon-explain`.
`llvm-mca` and `nm` are available for both compilers, and the Templates menu has
an assembly diff of plain clang against clang + Poseidon.

## Citation

Poseidon extends the CPU optimizer of "Thinking Fast and Correct: Automated
Rewriting of Numerical Code through Compiler Augmentation" (CGO 2026).

## License

Apache License v2.0 with LLVM Exceptions, the repository's `LICENSE`.
