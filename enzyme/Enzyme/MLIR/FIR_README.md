# Enzyme ⇄ Flang FIR/HLFIR

Bring-up path for differentiating Fortran array intrinsics **at the FIR/MLIR
level**, while they are still first-class `hlfir.*` ops (`hlfir.sum`,
`hlfir.matmul`, `hlfir.dot_product`, ...), rather than after they lower to
`_FortranA*` runtime calls. See `../../../PLAN_flang_enzyme_mlir.md` (in the
llvm-project tree) for the full design and the Tier-1/2/3 rule plan.

## Two delivery vehicles, one registration

Both entry points register the same thing — the Enzyme dialect, the `enzyme`
passes, and the autodiff interface external models — into a context that also
carries Flang's FIR/HLFIR dialects.

| File | Vehicle | Use when |
|---|---|---|
| `enzyme-fir-plugin.cpp` → `FIREnzyme-<ver>.so` | MLIR dialect+pass plugin for `fir-opt` (`--load-dialect-plugin` / `--load-pass-plugin`) | the host `fir-opt` shares a single `libMLIR`/`libLLVM` with the plugin (a shared-library LLVM build) |
| `fir-enzyme-opt.cpp` → `fir-enzyme-opt` | standalone `fir-opt` with Enzyme linked in | a **fully-static** LLVM build (loading a second static LLVM copy into `fir-opt` would double-register `cl::opt`s) |

### Why the plugin needs a non-static LLVM

`fir-opt` must export its symbols for a dialect/pass plugin to resolve MLIR/LLVM
against the host — upstream `mlir-opt` calls
`export_executable_symbols_for_plugins(mlir-opt)`; `fir-opt` does not, so a
one-line addition of `export_executable_symbols_for_plugins(fir-opt)` to
`flang/tools/fir-opt/CMakeLists.txt` is required (build-only, no behavior
change). Even with symbols exported, on a **static** LLVM build the plugin
carries its own copy of LLVM whose `cl::opt` global constructors re-register
options that `fir-opt` already registered → a fatal
`Option '...' registered more than once`. Use `fir-enzyme-opt` there, or build
LLVM with `LLVM_LINK_LLVM_DYLIB=ON`.

## Building

Requires an LLVM/MLIR (and, for the tool, Flang) build to point at.

```sh
cmake -G Ninja -B build-fir -S enzyme \
  -DENZYME_MLIR=ON -DENZYME_CLANG=OFF \
  -DLLVM_DIR=<llvm-build>/lib/cmake/llvm \
  -DMLIR_DIR=<llvm-build>/lib/cmake/mlir

# plugin (default ON): build-fir/Enzyme/MLIR/FIREnzyme-<ver>.so
cmake --build build-fir --target FIREnzyme-<ver>

# standalone tool (needs Flang + Clang cmake packages):
cmake -B build-fir -S enzyme -DENZYME_FIR_TOOL=ON \
  -DFlang_DIR=<llvm-build>/lib/cmake/flang \
  -DClang_DIR=<llvm-build>/lib/cmake/clang
cmake --build build-fir --target fir-enzyme-opt
```

## Using

```sh
# lower Fortran to HLFIR (ops still present)
flang-new -fc1 -emit-hlfir foo.f90 -o foo.hlfir

# differentiate (standalone tool)
build-fir/Enzyme/MLIR/fir-enzyme-opt --enzyme foo.hlfir -o foo.diff.hlfir

# differentiate (plugin, shared-LLVM builds)
fir-opt --load-dialect-plugin=FIREnzyme-<ver>.so \
        --load-pass-plugin=FIREnzyme-<ver>.so \
        --enzyme foo.hlfir -o foo.diff.hlfir
```

Differentiation is driven by `enzyme.autodiff` / `enzyme.fwddiff` ops naming the
callee and per-argument activities, exactly as in `test/MLIR/`.

## Tier-1 HLFIR autodiff models

Flang-dependent; built into `MLIREnzymeHLFIRImplementations` and registered by
`registerHLFIRDialectAutoDiffInterface` (see
`Implementations/HLFIRAutoDiffOpInterfaceImpl.cpp`). Includes an
`AutoDiffTypeInterface` for `!hlfir.expr` (shadow type, elementwise add via
`hlfir.elemental`, conjugate).

## Status

- **Milestone 0 (done):** FIR+Enzyme wiring; `enzyme` pass runs in an
  FIR/HLFIR-aware context. Smoke test: `test/MLIR/FIR/smoke.mlir`.
- **`hlfir.matmul` forward (done):** `d(A*B) = dA*B + A*dB`, both- and
  single-active operands. Test: `test/MLIR/FIR/matmul_fwd.mlir`.
- **`hlfir.matmul` reverse (done):** `Ā += matmul(G, transpose(B))`,
  `B̄ += matmul_transpose(A, G)`, caching the primal operands; `createNullValue`
  for static-extent `!hlfir.expr`. Test: `test/MLIR/FIR/matmul_rev.mlir`.
- **Next:** `hlfir.sum` / `hlfir.dot_product`; and, for whole-Fortran-function
  AD, `!fir.ref` active-memory support plus rules for the surrounding memory ops
  (`hlfir.declare`/`assign`/`destroy`, `fir.load`). See the PLAN's Tier-1 table.
