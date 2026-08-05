# Enzyme ⇄ Flang FIR/HLFIR

Differentiating Fortran **at the FIR/MLIR level**, while array intrinsics are
still first-class `hlfir.*` ops (`hlfir.sum`, `hlfir.matmul`,
`hlfir.dot_product`, ...), rather than after they lower to `_FortranA*` runtime
calls.

## Two delivery vehicles, one registration

Both entry points register the same thing — the Enzyme dialect, the `enzyme`
passes, and the autodiff interface external models — into a context that also
carries Flang's FIR/HLFIR dialects.

| File | Vehicle | Use when |
|---|---|---|
| `enzyme-fir-plugin.cpp` → `FIREnzyme-<ver>.so` | MLIR dialect+pass plugin for `fir-opt` (`--load-dialect-plugin` / `--load-pass-plugin`) | the host `fir-opt` shares a single `libMLIR`/`libLLVM` with the plugin (a shared-library LLVM build) |
| `HLFIRFlangPluginRegistration.cpp` → `FlangEnzymeMLIR-<ver>.so` | `flang -fc1 -load`, hooking the Enzyme passes into flang's own HLFIR-to-FIR pipeline | differentiating a whole `.f90` in one flang invocation |

### Why the plugin needs a non-static LLVM

`fir-opt` must export its symbols for a dialect/pass plugin to resolve MLIR/LLVM
against the host — upstream `mlir-opt` calls
`export_executable_symbols_for_plugins(mlir-opt)`; `fir-opt` does not, so a
one-line addition of `export_executable_symbols_for_plugins(fir-opt)` to
`flang/tools/fir-opt/CMakeLists.txt` is required (build-only, no behavior
change). Even with symbols exported, on a **static** LLVM build the plugin
carries its own copy of LLVM whose `cl::opt` global constructors re-register
options that `fir-opt` already registered → a fatal
`Option '...' registered more than once`. Build LLVM with
`LLVM_LINK_LLVM_DYLIB=ON`; a static build would need a standalone `fir-opt` with
Enzyme linked in instead, which this stack does not provide.

## Building

Requires an LLVM/MLIR (and, for the FIR/HLFIR layer, Flang) build to point at.

```sh
cmake -G Ninja -B build-fir -S enzyme \
  -DENZYME_MLIR=ON -DENZYME_CLANG=OFF \
  -DLLVM_DIR=<llvm-build>/lib/cmake/llvm \
  -DMLIR_DIR=<llvm-build>/lib/cmake/mlir

# plugin (default ON): build-fir/Enzyme/MLIR/FIREnzyme-<ver>.so
cmake --build build-fir --target FIREnzyme-<ver>

# the Flang-dependent FIR/HLFIR autodiff layer (needs the Flang cmake package):
cmake -B build-fir -S enzyme -DENZYME_FLANG_MLIR=ON \
  -DFlang_DIR=<llvm-build>/lib/cmake/flang
cmake --build build-fir --target FlangEnzymeMLIR-<ver>
```

## Using

```sh
# lower Fortran to HLFIR (ops still present)
flang-new -fc1 -emit-hlfir foo.f90 -o foo.hlfir

# differentiate (plugin, shared-LLVM builds). MlirOptMain exposes passes coming
# from a plugin only through --pass-pipeline, never as bare --enzyme* flags.
fir-opt --load-dialect-plugin=FIREnzyme-<ver>.so \
        --load-pass-plugin=FIREnzyme-<ver>.so \
        --pass-pipeline='builtin.module(enzyme)' foo.hlfir -o foo.diff.hlfir

# or in one step, straight from Fortran
flang -fc1 -load FlangEnzymeMLIR-<ver>.so -emit-llvm foo.f90 -o foo.ll
```

Differentiation is driven by `enzyme.autodiff` / `enzyme.fwddiff` ops naming the
callee and per-argument activities, exactly as in `test/MLIR/`.

## FIR/HLFIR autodiff models

Flang-dependent; built into `MLIREnzymeHLFIRImplementations` and registered by
`registerFIRDialectAutoDiffInterface` (see
`Implementations/FIRAutoDiffOpInterfaceImpl.cpp`) and
`registerHLFIRDialectAutoDiffInterface` (see
`Implementations/HLFIRAutoDiffOpInterfaceImpl.cpp`), the latter including an
`AutoDiffTypeInterface` for `!hlfir.expr`.

## Status

- **Done:** the FIR+Enzyme wiring (`test/MLIR/FIR/smoke.mlir`), and forward and
  reverse mode over the by-reference memory model — `!fir.ref` as active memory
  plus the surrounding `fir.load`/`fir.store`/`fir.alloca` and
  `hlfir.declare`/`hlfir.assign` (`test/MLIR/FIR/fir_ref_*.mlir`), end to end
  through `flang` (`test/MLIR/FIR/flang_plugin_*`). For the array intrinsics,
  `hlfir.matmul` forward and reverse over `!hlfir.expr`
  (`test/MLIR/FIR/matmul_*.mlir`, `fortran_matmul.f90`).
- **Next:** the remaining array intrinsics, `hlfir.sum` and
  `hlfir.dot_product`.
