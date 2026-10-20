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
enzyme/Poseidon/            the pass (Plugin.cpp and the sources beside it)
  runtime/                  profiler and GEMM dispatch runtimes the driver links
  tools/                    poseidon-clang driver, poseidon-calibrate, poseidon-explain,
                            Herbie platform
  cost_models/              the paper's RTX 5090 model and its Herbie platform
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
   written; no shadows are needed. `fn` names a host function defined in the
   same translation unit, and `t` is a compile-time constant.
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
site emits. A site with no target is solved against
`-poseidon-comp-cost-budget` instead; at its default of 0 that is the most
accurate rewrite costing no more than the code as written, which may be the
code as written. `poseidon-clang++ -poseidon-help` lists the user flags, and
any other `-poseidon-<flag>` reaches the pass.

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
`<func>.explain.json` per site, schema `poseidon-explain/1`. Every number in it
is the solver's own; a field the solve did not compute (for example a modelled
error when the DP table came from the cache) is omitted.

| field | content |
|---|---|
| `function` | `name` (demangled), `symbol`, `ir` (the solved clone), `site` and `loc` locations |
| `target` | `tau` and `tau_from` (`site` or `flag`), or `budget`; `cost_model`, `selector` |
| `result` | `cost_before`, `cost_after`, `modelled_rel_error_as_written`, `modelled_rel_error`, `applied` (ids) |
| `subgraphs[]` | `id`, `kind` (`subgraph` or `matmul`), `loc`, `cost`, `executions`, `ops[]` (`opcode`, `type`, `loc`, `executions`, `sensitivity`, `gradient`, `min`, `max`), `inputs[]` (`var`, `name`, `loc`, `type`, `min`, `max`), `outputs[]` (`name`, `loc`, `fpcore`, `c`, the `ops` and `inputs` it depends on), `original` (`fpcore`, `c`), `herbie[]` (per core: precision, status), `left_as_written` |
| `subgraphs[].alternatives[]` | `id` (what `-poseidon-apply-rewrites` takes), `kind` (`algebraic`, `precision`, `expansion`, `matmul`), `summary`, `fpcore`, `c`, `approx`, `approx_note`, `changes`, `cost_delta`, `modelled_error`, `herbie_error_bits` (`before`, `after`), `status` (`applied`, `pareto`, `rejected`), `reason` |
| `pareto[]` | `budget` (what `-poseidon-comp-cost-budget` takes), `cost`, `modelled_rel_error`, `applied`, `selected` |

Locations are `{file, line, col}`; modelled errors are relative and in the
unit `-poseidon-tau` is given in. An applied algebraic alternative reads

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
(default `/tmp/poseidon-ce-cache`) and reused by later compiles; `package.sh`
ships the examples' results so their first compile does not search. The
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

## Artifacts

The reproduction packages of the papers (`artifacts/cgo2027`, `artifacts/cgo2026`,
`artifacts/mfem-workshop-2026`, each with its own `README.md` and
`expected_results.md`) are kept in the PRONTOLab/Poseidon-dev tree, not on the
Enzyme branch.

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
