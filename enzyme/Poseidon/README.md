# Poseidon

Poseidon is a profile-guided numerical optimizer, built as an LLVM pass plugin
next to Enzyme. Given an accuracy target, it rewrites the floating-point code a
program marks to run as fast as the target allows: algebraic rewrites found by
Herbie, precision changes, floating-point expansions, and on GPUs matrix
products raised to tensor cores or emulated with Ozaki-II. It judges every
candidate on the values a profiling run of the program recorded, with Enzyme
differentiating the probes. Host code in C and C++, NVIDIA GPUs (CUDA) and AMD
GPUs (HIP) are optimized alike.

Project website: <https://prontolab.github.io/numerics/>

## Build

Poseidon needs LLVM 21 or newer, built with clang, and MPFR and GMP. Its
algebraic rewrites run Herbie: `-DPOSEIDON_HERBIE_BINARY=<herbie>` names a
prebuilt one, and `-DPOSEIDON_BUILD_HERBIE=ON` builds it from source, which
takes Rust and Racket. The CUDA runtimes are built when CMake finds a CUDA
toolkit; the HIP runtimes when `-DPOSEIDON_ROCM_PATH=<rocm>` names a ROCm
installation, for the processor `-DPOSEIDON_AMDGPU_ARCH` names (default
`gfx942`).

```sh
cmake -G Ninja -S enzyme -B build \
    -DLLVM_DIR=<llvm-build>/lib/cmake/llvm \
    -DENZYME_POSEIDON=ON
ninja -C build
ninja -C build check-poseidon
```

The build installs the drivers `poseidon-clang` and `poseidon-clang++`, the
tools `poseidon-calibrate`, `poseidon-probe` and `poseidon-explain`, the header
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
`_TAU` forms give a site an accuracy target of its own. `-poseidon-kernels=<regex>`
makes every matching kernel a site without a change to the source.

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

The run writes the profile of each site to `./poseidon.profile`, or to the
directory `POSEIDON_PROFILE_DIR` names. A small problem is enough when its
values are of the kind the large ones compute. A program that declares its
result with `poseidon_metric(name, value)` runs under
`poseidon-probe -- ./app small-input` instead, which measures each site's
condition number against that result in 1 + 3 x <sites> more runs. Profiling
host code needs glibc.

### 4. Optimize

```sh
poseidon-clang -O2 -poseidon-profile-use -poseidon-tau=1e-8 app.c -o app_opt
poseidon-clang++ -O2 -poseidon-profile-use -poseidon-tau=1e-8 app.cu -o app_opt
```

`-poseidon-tau` is the relative accuracy target of every site without its own;
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

## License

Apache License v2.0 with LLVM Exceptions, the repository's `LICENSE`.
