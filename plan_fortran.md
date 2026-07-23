# Plan: End-to-end Fortran AD with `flang` + the Enzyme plugin

## Goal

Make this work as a single compiler invocation:

```fortran
! foo.f90
subroutine driver(a, da, b, db, dc)
  use enzyme
  real, intent(in)  :: a(2,2), b(2,2)
  real, intent(in)  :: da(2,2), db(2,2)
  real, intent(out) :: dc(2,2)
  dc = enzyme_fwddiff(mm, enzyme_dup, a, da, enzyme_dup, b, db)
end subroutine
```

```sh
flang -fpass-plugin=FlangEnzyme-<v>.so -fplugin=FlangEnzyme-<v>.so foo.f90 -o foo
```

and have `enzyme_fwddiff`/`enzyme_autodiff` differentiate the callee **through
the HLFIR array intrinsics** (`hlfir.matmul`, `hlfir.sum`, ...) while they are
still first-class ops — not after they lower to `_FortranA*` runtime calls.

This document is the driver/UX/integration companion to the op-level design in
`../llvm-project/PLAN_flang_enzyme_mlir.md` (the Tier-1/2/3 rule plan). Here we
describe how a Fortran program reaches those rules.

## The pipeline, stage by stage

```
foo.f90
  │  flang -fc1  (front end + lowering)
  ▼
HLFIR                                   fir.call @f__enzyme_fwddiff(boxproc, markers, args...)
  │                                     hlfir.matmul, hlfir.sum, ... still present
  │  createHLFIRToFIRPassPipeline
  │    ├─ HLFIROptEarly EP  ◄─────────── (1) Enzyme call-lowering pass:
  │    │                                     fir.call @…enzyme_fwddiff  ──►  enzyme.fwddiff @mm(...)
  │    │                                 (2) Enzyme differentiation pass (`enzyme`):
  │    │                                     enzyme.fwddiff  ──►  differentiated func using
  │    │                                     the Tier-1 hlfir.* AD rules
  │    ├─ HLFIR simplification / inlining
  │    └─ HLFIROptLast EP   ◄─────────── (alt. placement, just before lowering)
  │  createLowerHLFIRIntrinsics          any hlfir.* left → runtime calls
  ▼
FIR → LLVM IR
  │  -fpass-plugin=FlangEnzyme  (LLVM Enzyme, for anything still symbol-level)
  ▼
foo
```

The **policy lever** is pass placement: differentiate at `HLFIROptEarly` so the
`hlfir.*` intrinsics are still ops and the Tier-1 rules fire (see the op-level
PLAN). Anything left as a runtime call by the time LLVM Enzyme runs falls to the
symbol-level path.

## Components

### A. Fortran surface (`enzyme/Fortran/enzyme.f90`) — exists

Provides `enzyme_fwddiff`/`enzyme_autodiff` (aliases of external
`f__enzyme_fwddiff`/`f__enzyme_autodiff`) and the activity markers
`enzyme_const`, `enzyme_dup`, `enzyme_dupnoneed`, `enzyme_out` as C-bound
globals. Mirrors the C/C++ `__enzyme_autodiff` surface.

### B. flang HLFIR-to-FIR extension points — exists (llvm-project)

`[flang] Add HLFIR-to-FIR pass pipeline extension points` (branch
`flang-hlfir-pass-pipeline-ep`) adds, on `MLIRToLLVMPassPipelineConfig`:

- `registerHLFIROptEarlyEPCallbacks(cb)` / `invokeHLFIROptEarlyEPCallbacks` —
  runs at the very start of `createHLFIRToFIRPassPipeline`, intrinsics intact.
- `registerHLFIROptLastEPCallbacks(cb)` / `invokeHLFIROptLastEPCallbacks` —
  runs just before `createLowerHLFIRIntrinsics`.

Plus `[flang] Export fir-opt symbols for MLIR dialect/pass plugins` so `fir-opt`
can host dialect/pass plugins (needed for the plugin delivery vehicle on a
shared-LLVM build).

### C. Enzyme call-lowering pass — **to build**

An MLIR pass on HLFIR that rewrites the differentiation-hook calls into
`enzyme.fwddiff` / `enzyme.autodiff` ops, mirroring `HandleAutoDiff` +
`getMetadataName` in `enzyme/Enzyme/Enzyme.cpp` (the LLVM path):

Given
```mlir
%f  = fir.address_of(@_QPmm) : (…) -> …
%bp = fir.emboxproc %f : … -> !fir.boxproc<() -> ()>
%r  = fir.call @_QPf__enzyme_fwddiff(%bp, <markers+args…>) : (…) -> T
```
emit
```mlir
%r = enzyme.fwddiff @_QPmm(<primal,shadow,…>) {activity=[…], ret_activity=[…]}
     : (…) -> T
```

Algorithm (per call to a callee whose name contains `enzyme_fwddiff` /
`enzyme_autodiff`):
1. Operand 0 is the callee: trace `fir.emboxproc` → `fir.address_of` → symbol.
2. Walk the rest, tracking activity like the LLVM path. `getMetadataName`'s
   analog: an operand that is a `fir.address_of`/load of a global named
   `enzyme_{const,dup,dupnoneed,out}` is a **marker** setting the type of the
   *next* argument (default `dup` for fwddiff). `dup`/`dupnoneed` consume a
   `(primal, shadow)` pair; `const` consumes a primal only; `out` (reverse)
   marks an active scalar output.
3. Build the op: `fn=@target`, `inputs=` collected primals/shadows,
   `activity=`/`ret_activity=` the `#enzyme<activity …>` array attrs.
4. Replace the `fir.call` result; erase the call (and dead boxproc/address_of).

Lives in the Flang-dependent `MLIREnzymeHLFIRImplementations` library (it needs
the `fir`/`hlfir` C++ ops). Registered as a normal MLIR pass so it can be added
to a `PassManager`.

### D. Enzyme MLIR differentiation + Tier-1 HLFIR rules — partial

- `enzyme` differentiation pass — exists (core Enzyme-MLIR).
- `AutoDiffTypeInterface` for `!hlfir.expr` — done.
- `hlfir.matmul` forward + reverse — **done and tested**
  (`test/MLIR/FIR/matmul_{fwd,rev}.mlir`).
- Remaining Tier-1 ops (`hlfir.sum`, `hlfir.dot_product`, `hlfir.transpose`,
  `hlfir.reshape`, `cshift`/`eoshift`, `maxval`/`minval`, …) — to do, per the
  op-level PLAN's Tier-1 table.

### E. Plugin registration bridge — **to build**

The FlangEnzyme shared object (`enzyme/Enzyme/Flang/EnzymeFlang.cpp`, PR #2968)
must, during `flang -fc1`, register (C) + (D) into the compilation's
`MLIRToLLVMPassPipelineConfig` via the (B) extension points, and register the
Enzyme dialect + HLFIR autodiff interfaces + `hlfir` in the MLIR context.

Options for the bridge (frontend plugins currently can only *replace* codegen,
not augment it, so a hook is needed):
1. **Global registry consulted by flang** (preferred): a small
   `fir::registerHLFIRExtension(cb)` global list that `FrontendActions.cpp`
   applies to `config` next to `registerDefaultInlinerPass(config)`. The plugin
   appends to it from a static initializer at `-load` time. Minimal, matches the
   existing inliner model. (Requires one more small flang change.)
2. Teach flang's frontend-plugin API to expose the codegen `config` to loaded
   plugins.

Until the bridge lands, (C)+(D) are exercised via the standalone `fir-enzyme-opt`
tool (below), which registers everything explicitly.

### F. Delivery vehicles — exists

Two ways to run the MLIR Enzyme work; both register the same passes/interfaces
(see `enzyme/Enzyme/MLIR/FIR_README.md`):
- **`fir-enzyme-opt`** — `fir-opt` with Enzyme statically linked. The vehicle
  for a fully-static LLVM build (a dlopen'd plugin would double-register
  `cl::opt`s). Used for all FileCheck bring-up.
- **`FIREnzyme-<v>.so`** — dialect+pass plugin for `fir-opt`, for shared-LLVM
  builds.

## Activity semantics (mirror the LLVM/Clang path)

| Fortran marker | `#enzyme<activity …>` | Consumes | Meaning |
|---|---|---|---|
| (default, fwd) / `enzyme_dup` | `enzyme_dup` | primal + shadow | dual number in, dual out |
| `enzyme_dupnoneed` | `enzyme_dupnoneed` | primal + shadow | as dup, primal result not needed |
| `enzyme_const` | `enzyme_const` | primal | inactive argument |
| `enzyme_out` (reverse) | `enzyme_active` | primal | active scalar, gradient returned |

Return activity: forward → `enzyme_dupnoneed` (return the tangent); reverse →
seed comes in, gradient(s) via the input shadows. The pass sets `ret_activity`
from the callee's result and whether the primal result is used.

## Known gap: whole-function AD needs `!fir.ref` support

Real Fortran passes array/scalar arguments **by reference**
(`!fir.ref<!fir.array<…>>`, `!fir.ref<f32>`), and the callee body is
load/store/`hlfir.declare`/`hlfir.assign` over that memory. Differentiating a
whole Fortran function therefore additionally needs:

- `AutoDiffTypeInterface` for `!fir.ref<…>` (mutable / active-memory, shadow =
  parallel buffer), analogous to Enzyme's `MemRefAutoDiffTypeInterface`.
- AD rules for the surrounding memory ops: `fir.load`, `hlfir.declare`,
  `hlfir.assign`, `hlfir.destroy`, `fir.store`, allocations.

The `hlfir.matmul` value-semantics rules already work on `!hlfir.expr`; the
ref/memory layer is the missing connective tissue for source-to-source Fortran.
This is the natural next milestone once (C)+(E) land.

## Goal: splittable passes / minimal dialect surface

The Enzyme MLIR passes and interface registrations should be **splittable so a
host only needs the dialects it actually uses**. Today the `enzyme`
differentiation pass lives in `MLIREnzymeTransforms`, which transitively pulls
`MLIREnzymeImplementations` — the autodiff models for *every* upstream dialect
(linalg, tensor, nvvm, ...). A lean `flang` plugin then fails to load
(`undefined symbol: ...linalg::ReduceOp...`) because `flang` does not link those
dialects, and bundling them risks a second static LLVM copy / `cl::opt` clash
(see component E and [[static-llvm-plugin-cl-opt-clash]]).

Target: factor the differentiation pass and the per-dialect autodiff-interface
registrations so a driver can pull in only, say, `{builtin, arith, math, func,
cf, scf, fir, hlfir, enzyme}` and nothing that references linalg/tensor/etc.
Concretely — a core `enzyme` pass with no hard dependency on the dialect
implementation libraries, plus opt-in registration per dialect (the
`register<Dialect>DialectAutoDiffInterface` functions already exist; the pass
library should not force-link all of them). This unblocks wiring the `enzyme`
diff pass into the lean flang plugin (component E's remaining TODO).

## Determined next step: dialect-agnostic active stores in activity analysis

Root cause of the `!fir.ref` local-result-buffer gap (pinpointed): the `enzyme`
pass uses the *classic* `ActivityAnalysis.cpp`, which propagates stored-value →
pointer activity only for hard-coded `LLVM::StoreOp` and `memref::StoreOp` (via
`getValueToStore()`, ~5 sites). It has no case for `hlfir.assign` / `fir.store`,
so a function's local result variable (`fir.alloca` written by `hlfir.assign`,
read back for the return) is deemed constant and the store-handler errors.

`Analysis/` has **no flang dependency** (and must not — see the splittability
goal), so the fix is *not* to hard-code `fir`/`hlfir` there. Instead:

1. Introduce a small op interface, e.g. `StoreLikeInterface`, exposing the
   `(storedValue, pointerOperand)` pair (generalizing `getValueToStore`).
2. Attach it — for free — from the `MemoryIdentityOp` registration, which
   already carries `storedvals`/`ptrargs` (so `fir.store`, `hlfir.assign`,
   `memref.store`, `LLVM.store` all get it uniformly).
3. Replace the hard-coded store `dyn_cast`s in `ActivityAnalysis.cpp` with the
   interface query.

This unblocks whole-Fortran-function differentiation **and** advances the
splittability goal (activity analysis stops knowing about specific dialects).

## Milestones

- [x] `fir-opt` symbol export; HLFIR pipeline extension points (llvm-project).
- [x] `fir-enzyme-opt` / `FIREnzyme` plugin delivery vehicles.
- [x] `!hlfir.expr` `AutoDiffTypeInterface`; `hlfir.matmul` forward + reverse.
- [x] Fortran → HLFIR pipeline smoke test (`test/MLIR/FIR/fortran_matmul.f90`).
- [x] **(C)** Enzyme call-lowering pass (`enzyme-lower-fortran-calls`):
      `enzyme_fwddiff/autodiff` fir.call → `enzyme.fwddiff/autodiff` op, with
      `enzyme_{const,dup,dupnoneed,out}` activity parsing. Tested end-to-end from
      Fortran via flang-new (`test/MLIR/FIR/lower_fortran_calls.f90`).
- [x] **(E)** Plugin registration bridge: flang global config-augmentor registry
      (`fir::registerPassPipelineConfigCallback`, invoked in
      `CodeGenAction::lowerHLFIRToFIR`) + the `FlangEnzymeMLIR-<v>.so` plugin
      whose static initializer registers the call-lowering pass at
      `HLFIROptEarly`. Verified: `flang -fc1 -load FlangEnzymeMLIR.so -emit-fir`
      emits `enzyme.fwddiff` (`test/MLIR/FIR/flang_plugin_lower.f90`). The plugin
      stays lean (resolves MLIR/FIR/flang from the host `flang`), so no second
      static LLVM copy / cl::opt clash. Still to wire: the `enzyme` diff pass at
      the EP (blocked on `!fir.ref` support below).
- [x] `!fir.ref` active-memory `AutoDiffTypeInterface` + memory-op rules
      (`FIRAutoDiffOpInterfaceImpl.cpp` + `FIRDerivatives.td`) + a
      dialect-agnostic `StoreLikeInterface` used by activity analysis
      (attached to `fir.store`/`hlfir.assign`; the analysis stops hard-coding
      dialects). **Forward mode fully differentiates a by-reference Fortran
      subroutine**: `subroutine sq(x,r); r = x*x` → `dr = 2*x*dx` stored through
      the shadow output. Test: `test/MLIR/FIR/fir_ref_fwd.mlir`.
      **Narrow remaining case:** a `real function`'s *local result buffer*
      (`fir.alloca` behind `hlfir.declare`) still reads constant, so the
      returned tangent is dropped — activity analysis needs `hlfir.declare`
      alias handling (or the result var lowered like an `intent(out)` arg).
      Reverse mode (`!fir.ref`) is future work.
- [ ] Split the `enzyme` pass / interface registration so a host links only the
      dialects it uses (see the splittability goal above).
- [ ] Remaining Tier-1 `hlfir.*` rules (`sum`, `dot_product`, …).
- [ ] End-to-end: `flang foo.f90` with `enzyme_fwddiff` produces correct
      derivatives; numerical Integration test.

## Testing strategy

- **Unit (op rules & call-lowering):** `fir-enzyme-opt` + FileCheck on hand-written
  or flang-emitted HLFIR (`test/MLIR/FIR/*.mlir`).
- **Pipeline hooks:** flang unittest `PassPipelineTest.cpp` (extension points).
- **End-to-end:** `flang-new -fc1 -emit-hlfir foo.f90 | fir-enzyme-opt …`, and
  ultimately `flang … -fplugin=FlangEnzyme` compiling+running a program whose
  output is checked against finite differences.
```
