# Reactant.jl Integration for Online_r2 Checkpointing

This document describes the work done to expose `enzyme.online_checkpointing` through
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), the current status, and the
architectural gap that blocks end-to-end Julia gradient tests.

---

## Fork location

```
github.com/sriharikrishna/Reactant.jl
local:  /Users/snarayan/soft/Reactant.jl
```

Install from the fork:
```julia
import Pkg
Pkg.develop(path="/path/to/Reactant.jl/lib/ReactantCore")
Pkg.develop(path="/path/to/Reactant.jl")
# or from GitHub:
# Pkg.add(url="https://github.com/sriharikrishna/Reactant.jl")
```

---

## Changes made to the Reactant fork

### 1. `lib/ReactantCore/src/ReactantCore.jl`

- Added `Online` to the export list (alongside `Periodic`, `Binomial`).
- Added the `Online(budget::Int)` struct with docstring referencing Stumm-Walther 2010.
- Updated the `trace_while` validation error message to mention `Online(n)`.

```julia
struct Online
    budget::Int
end
```

### 2. `src/Ops.jl` (lines ~2461–2470)

Added an `elseif` branch in the attribute-setting block inside `while_loop`:

```julia
elseif checkpointing isa ReactantCore.Online
    MLIR.IR.setattr!(while_op, "enzyme.enable_checkpointing", MLIR.IR.Attribute(true))
    MLIR.IR.setattr!(while_op, "enzyme.online_checkpointing", MLIR.IR.UnitAttribute())
    MLIR.IR.setattr!(
        while_op, "enzyme.checkpoint_period", MLIR.IR.Attribute(checkpointing.budget)
    )
```

This is analogous to the existing `Binomial` branch that sets `enzyme.binomial_checkpointing`.

### 3. `src/compiler/OptimizationPasses.jl` (line 196)

Added `lower-enzyme-online-progress` and `lower-enzyme-binomial-progress` to the
`enzyme_pass` postpasses string:

```julia
const enzyme_pass::String = "enzyme{postpasses=\"...,lower-enzyme-online-progress,lower-enzyme-binomial-progress,...\"}"
```

These passes must run after the `enzyme` AD pass and before `remove-unnecessary-enzyme-ops`,
as they lower Enzyme-specific ops (`enzyme.online_progress`, `enzyme.binomial_progress`)
that the AD pass emits.

---

## Building libReactantExtra from source (Bazel)

The pre-built `ReactantExtra_jll` artifact does not include our new pass
(`LowerOnlineProgressPass`) or our modified `SCFAutoDiffOpInterfaceImpl`. A local build
is required.

```bash
cd /path/to/Reactant.jl/deps/ReactantExtra
bazelisk build \
  -c opt \
  --action_env=JULIA=$(which julia) \
  --check_visibility=false \
  --verbose_failures \
  --spawn_strategy=local \
  --config=macos_local \          # macOS only
  --copt=-Wno-private-header \
  --copt=-Wno-unused-command-line-argument \
  --define=enable_z3=false \
  :libReactantExtra.so
```

The built library is at:
```
deps/ReactantExtra/bazel-bin/.../libReactantExtra.so   (Mach-O dylib on macOS)
```

After building, copy it over the artifact to make Julia load it:
```bash
chmod u+w ~/.julia/artifacts/<hash>/lib/libReactantExtra.dylib
cp bazel-out/.../libReactantExtra.so ~/.julia/artifacts/<hash>/lib/libReactantExtra.dylib
```

### Bazel patches required (macOS 15 / Apple clang 16)

These must be applied to files in Bazel's external cache
(`/private/var/tmp/_bazel_<user>/<hash>/external/`) after the first `fetch` but before
compilation:

| File | Patch | Reason |
|------|-------|--------|
| `enzyme/tools/enzyme-tblgen/enzyme-tblgen.cpp` | `->getValues()` → `->getElements()` (all occurrences) | LLVM API rename in newer LLVM |
| `enzyme_ad/src/enzyme_ad/jax/BUILD` | Add `":Perfify"` to `RegistryUtils` deps list | Missing tablegen-generated `.h.inc` dependency causes `fatal error: file not found` |
| `z3/BUILD.bazel` | Remove `"-include", "util/z3_std_format.h"` from z3's `copts` | The force-included shim defines `std::format` in `namespace std` even when `__cpp_lib_format` is set, causing ambiguous-call errors with Apple clang 16 + C++20 |

The Enzyme source files from our local fork
(`LowerOnlineProgressPass.cpp`, `SCFAutoDiffOpInterfaceImpl.cpp`, `EnzymeOps.td`,
`Passes.td`, `LoopCheckpointing.h`, `AffineAutoDiffOpInterfaceImpl.cpp`) must be copied
into `external/enzyme/` in the Bazel cache to replace the upstream pinned commit.

---

## Architectural gap: `stablehlo.while` vs `scf.while`

This is the key blocker preventing end-to-end Julia gradient tests from passing.

### The problem

Our `WhileOpInterfaceReverse` (in `SCFAutoDiffOpInterfaceImpl.cpp`) handles **`scf.while`**:

```cpp
scf::WhileOp::attachInterface<WhileOpInterfaceReverse>(*context);
```

Reactant's compilation pipeline is:

```
Julia source
  → @trace macro (ReactantCore)
  → stablehlo.while op  (Ops.jl emits enzyme.online_checkpointing attribute)
  → enzyme AD pass (sees stablehlo.while, NOT scf.while)
  → no WhileOpInterfaceReverse registered for stablehlo.while
  → falls back to dynamic tape: stablehlo.dynamic_pad
  → XLA compilation fails: 'stablehlo.dynamic_pad' can't be translated to XLA HLO
```

The `enzyme.online_checkpointing` attribute IS correctly emitted on the `stablehlo.while`
op (verified by the IR attribute test passing), but the Enzyme AD pass has no handler
for `stablehlo.while` and falls back to the unconstrained differentiation path.

### Error observed

```
UNKNOWN: <unknown>:0: error: 'stablehlo.dynamic_pad' op can't be translated to XLA HLO
```

### Test status

| Test | Status | Notes |
|------|--------|-------|
| `test/MLIR/ReverseMode/scf_while_online_checkpointing.mlir` | **PASS** | FileCheck, scf.while |
| `test/MLIR/ReverseMode/checkpointing_all_schemes.mlir` | **PASS** | FileCheck, all 4 schemes |
| `test/Python/exec_oceananigans_style.py` | **PASS** | Numerical, n=1..10 |
| `test/Julia/oceananigans_style_checkpointing.jl` | **PASS** | Via Checkpointing.jl (LLVM backend) |
| `test/Julia/reactant_online_checkpointing.jl` — IR attributes | **PASS** | Attribute emission confirmed |
| `test/Julia/reactant_online_checkpointing.jl` — gradients | **FAIL** | `stablehlo.while` gap |

### Paths to resolution

**Option 1 — Implement `StablehloWhileOpInterfaceReverse`** (recommended)

Register a new `ReverseAutoDiffOpInterface` external model for `stablehlo::WhileOp`,
analogous to `WhileOpInterfaceReverse` for `scf::WhileOp`. The `stablehlo.while` op
has a different structure (single before-region with condition terminator is separate
from the `stablehlo.while`'s condition and body regions), so `cacheWhileOnline` and
`reverseWhileOnline` would need variants that emit `stablehlo` ops rather than `scf` ops.
This is the most complete solution and the right long-term path.

**Option 2 — Lift `stablehlo.while` → `scf.while` before AD**

Reactant already has `enzyme-lift-cf-to-scf`. Check whether it handles `stablehlo.while`
and whether inserting it in the pass pipeline before the `enzyme` AD pass correctly
preserves the `enzyme.online_checkpointing` attribute through the lift. If so, this
requires only a one-line pipeline change in `OptimizationPasses.jl`.

**Option 3 — Upstream to Enzyme main + coordinate with enzyme_ad**

Once our `scf.while` changes are merged to Enzyme main, coordinate with the `enzyme_ad`
(Enzyme-JAX) team to add `stablehlo.while` checkpointing support there. The `enzyme_ad`
repo has direct StableHLO integration that the Reactant build already depends on.

---

## Implementation status and key blocker

Option 1 was attempted and revealed a fundamental XLA constraint.

### What was implemented

- `StablehloWhileOpInterfaceReverse` in `Enzyme/MLIR/Implementations/StablehloAutoDiffOpInterfaceImpl.cpp`
  (registered via `registerStablehloDialectAutoDiffInterface`)
- `cacheWhileOnlineStablehlo` — scaffold `stablehlo.while` for Online_r2 forward
- `reverseWhileOnlineStablehlo` — outer `scf.for` + storemap scan + inner Revolve for reverse
- Both patched into `enzyme_ad`'s `AutoDiffWhileRev::cacheValues` and `createReverseModeAdjoint`
  (MLIR ExternalModel first-wins, so the patch must be in enzyme_ad's file)

### XLA StableHLO constraint: the entire gradient module must be pure StableHLO

XLA requires that the **entire gradient module** contains only StableHLO ops — not
just the `stablehlo.while` body.  Any `arith.*`, `scf.*`, or `memref.*` op anywhere
in the gradient function causes:

```
'arith.constant' op can't be translated to XLA HLO
```

This means even the outer `scf.for` reverse loop, the `memref` checkpoint buffers,
and `arith.*` arithmetic in the gradient function body are rejected.

Attempts made:
1. **SM inside stablehlo.while body** — arith/scf/memref in body region rejected
2. **SM in outer scf.while driving stablehlo.while body** — arith/scf still in gradient module, rejected

The constraint is absolute: Online_r2 for Reactant/XLA requires a **complete StableHLO
reimplementation** of the entire forward+reverse algorithm — equivalent to what enzyme_ad's
`cacheBinomial` already does for binomial checkpointing.

### Paths forward

**Option A — Implement the complete Online_r2 gradient in pure StableHLO**

XLA's StableHLO→HLO translator rejects **any** non-StableHLO op in the entire gradient
module — not just inside `stablehlo.while` regions.  `arith.constant`, `scf.for`,
`memref.alloc` in the gradient function body are all rejected.

This means the ENTIRE gradient function (forward scaffold + reverse pass) must use only
StableHLO ops:
- SM arrays (`ch[]`, `ord_ch[]`, `num_rep[]`) as `tensor<smSize x i64>` iter-args
- Checkpoint buffers as `tensor<budget+1 x f32>` iter-args (not memref)
- All arithmetic as `stablehlo.add`/`compare`/`select`/`if`
- All loops (SM rotation, scan, Revolve reconstruction) as `stablehlo.while`
- `stablehlo.dynamic_update_slice`/`dynamic_slice` for tensor buffer stores/loads

The reference for this pattern is enzyme_ad's `LoopCheckpointing<AutoDiffWhileRev,
stablehlo::WhileOp>::cacheBinomial` which generates a fully-StableHLO binomial scaffold.
The Online_r2 forward needs the same treatment.  This is a large but well-defined effort.

**Option B — Split forward and backward passes**

Run Online_r2 in two phases:
1. A non-XLA host loop (using `scf.while` + `memref`) that drives the primal
   `stablehlo.while` forward, recording checkpoint steps.
2. A host-side reconstruction loop that invokes the primal `stablehlo.while`
   for replay segments.

This requires a host↔device boundary that Reactant currently does not expose cleanly.

**Option C — Use enzyme_ad's existing periodic/binomial path for now**

For known-trip-count loops, `@trace checkpointing=Binomial(n)` already works end-to-end
via enzyme_ad's `AutoDiffWhileRev` + `LoopCheckpointing`. Users can use this today.
Online_r2 is only strictly necessary for loops with *truly* unknown trip counts.

**Recommended next step:** Option A — translate the SM into pure StableHLO. The cleanest
entry point is a new `LowerOnlineProgressToStablehloPass` that emits `stablehlo.*` ops
instead of `arith`+`scf`+`memref`. This can live in `enzyme_ad` alongside the existing
stablehlo infrastructure.

---

## StablehloWhileOpInterfaceReverse — design reference

The implementation attempt documented below remains valid as a design reference. The
only piece that needs to change is the body of `cacheWhileOnlineStablehlo` to use
StableHLO ops instead of arith/scf/memref.

### What `stablehlo.while` looks like

```mlir
%result = stablehlo.while(%arg = %init) : tensor<f32>
  cond {
    // condition region: one block, ends with stablehlo.return %bool_tensor
    %cond = stablehlo.compare LT, %arg, %limit : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cond : tensor<i1>
  } do {
    // body region: one block, ends with stablehlo.return (new iter-args)
    %new = stablehlo.add %arg, %one : tensor<f32>
    stablehlo.return %new : tensor<f32>
  }
```

## Next step: implementing `StablehloWhileOpInterfaceReverse`

Option A above (StableHLO SM) is the recommended path. This section documents the concrete
implementation plan once the SM is available in StableHLO.

### What `stablehlo.while` looks like

```mlir
%result = stablehlo.while(%arg = %init) : tensor<f32>
  cond {
    // condition region: one block, ends with stablehlo.return %bool_tensor
    %cond = stablehlo.compare LT, %arg, %limit : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cond : tensor<i1>
  } do {
    // body region: one block, ends with stablehlo.return (new iter-args)
    %new = stablehlo.add %arg, %one : tensor<f32>
    stablehlo.return %new : tensor<f32>
  }
```

Key differences from `scf.while`:

| Aspect | `scf.while` | `stablehlo.while` |
|--------|-------------|-------------------|
| Condition | `before` block → `scf.condition(%i1, args...)` | `cond` region → `stablehlo.return %tensor_i1` |
| Body | `after` block → `scf.yield args...` | `body` region → `stablehlo.return args...` |
| Passed values | `scf.condition` passes args to after-block AND results | Iter-args loop directly; no mid-loop pass |
| Scalar type | `index` | `tensor<i64>` |
| Arithmetic | `arith.*` | `stablehlo.*` |
| Attribute attachment | Generic op format | Normal op format (attributes on the op) |

### File to create

```
enzyme/Enzyme/MLIR/Implementations/StablehloAutoDiffOpInterfaceImpl.cpp
```

The codebase follows a strict one-file-per-dialect convention:

```
SCFAutoDiffOpInterfaceImpl.cpp      — scf.*
AffineAutoDiffOpInterfaceImpl.cpp   — affine.*
MemRefAutoDiffOpInterfaceImpl.cpp   — memref.*
LLVMAutoDiffOpInterfaceImpl.cpp     — llvm.*
ArithAutoDiffOpInterfaceImpl.cpp    — arith.*
```

`stablehlo.while` belongs in a new `StablehloAutoDiffOpInterfaceImpl.cpp`, registered
in `CMakeLists.txt` alongside the others. Its registration function is called from the
same site that calls `registerSCFAutoDiffOpInterfaceImplementations`.

### Required refactor: extract shared while-loop helpers

The Online_r2 helper functions (`cacheWhileOnline`, `reverseWhileOnline`,
`cacheWhileFullTape`, `reverseWhileFullTape`) currently live as **static methods inside
`WhileOpInterfaceReverse`** in `SCFAutoDiffOpInterfaceImpl.cpp`. The stablehlo
implementation cannot call them from a different file.

Before writing `StablehloWhileOpInterfaceReverse`, extract these helpers into a shared
header — either extend `LoopCheckpointing.h` (where `cacheOnline`/`reverseOnline` for
`scf.for` already live) or create a new `WhileCheckpointing.h`. The helpers take a
`FinalClass`-style template parameter for the dialect-specific arithmetic differences
(scalar type, region structure, terminator ops).

This refactor is also a good opportunity to coordinate with PR #2374 (Pangoraw's
`scf.while` full-tape baseline): if their implementation lands first on `main`, the
shared-helper refactor becomes cleaner because both `scf.while` and `stablehlo.while`
can build on the same extracted base.

### Struct skeleton

```cpp
#include "mlir/Dialect/StableHLO/include/stablehlo/dialect/StablehloOps.h"

struct StablehloWhileOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          StablehloWhileOpInterfaceReverse,
          mlir::stablehlo::WhileOp> {

  static bool needsOnlineCheckpointing(mlir::stablehlo::WhileOp op) {
    return op->hasAttr("enzyme.online_checkpointing") &&
           op->hasAttr("enzyme.enable_checkpointing");
  }

  LogicalResult createForwardModification(Operation *op, ...) const {
    auto whileOp = cast<mlir::stablehlo::WhileOp>(op);
    if (needsOnlineCheckpointing(whileOp))
      return cacheWhileOnlineStablehlo(whileOp, gutils);
    return cacheWhileFullTapeStablehlo(whileOp, gutils);
  }

  LogicalResult createReverse(Operation *op, ...) const {
    auto whileOp = cast<mlir::stablehlo::WhileOp>(op);
    if (needsOnlineCheckpointing(whileOp))
      return reverseWhileOnlineStablehlo(whileOp, builder, gutils, ...);
    return reverseWhileFullTapeStablehlo(whileOp, builder, gutils, ...);
  }
};

void registerStablehloAutoDiffOpInterfaceImplementations(DialectRegistry &r) {
  r.addExtension(+[](MLIRContext *ctx, stablehlo::StablehloDialect *) {
    stablehlo::WhileOp::attachInterface<StablehloWhileOpInterfaceReverse>(*ctx);
  });
}
```

### What `cacheWhileOnlineStablehlo` needs to do differently

The Online_r2 forward scaffold is an outer `stablehlo.while` that:

1. **Carries SM state as tensors** — the 10 scalar SM fields (`check`, `capo`, etc.) must
   be `tensor<i64>` iter-args instead of `index`-typed SSA values. Each field becomes a
   `stablehlo.constant dense<0> : tensor<i64>` at init.

2. **Emits `OnlineProgressOp`** — unchanged; `OnlineProgressOp` uses `index` types
   internally and the scaffold converts with `arith.index_cast` / `tensor.extract`.
   Alternatively, keep SM scalars as `index` and convert at the `stablehlo.while`
   boundary (extract scalars from tensors on entry, pack back to tensors on yield).

3. **Condition region** — the scaffold's condition is:
   ```mlir
   cond {
     %cond_scalar = <evaluate primal stablehlo.while's cond region>
     stablehlo.return %cond_scalar : tensor<i1>
   }
   ```

4. **Body region** — dispatches on `action` (STORE vs FORWARD), same logic as
   `cacheWhileOnline` for `scf.while`.

5. **`stablehlo.return`** instead of `scf.yield` and no `scf.condition`.

### What `reverseWhileOnlineStablehlo` needs to do differently

The reverse structure is identical in logic to `reverseWhileOnline` (`scf.while`):
- Outer `scf.for` over `numFwd` steps (this can remain `scf.for` — the reverse
  reconstruction loop doesn't need to be `stablehlo.while`)
- Scan loop to find `capo` (Julia's `storemap`)  
- Inner `scf.while` for offline Revolve reconstruction

The only change is that checkpoint buffers hold `tensor<f32>` (or whatever the
iter-arg type is) instead of `f32`, and loads/stores use `stablehlo` ops where needed.

### Scalar arithmetic strategy

The cleanest approach: **keep all SM scalars as `index` inside the implementation**,
and only convert at the `stablehlo.while` iter-arg boundary:

```
// On entry to cond/body: extract scalar from tensor
%check_scalar = tensor.extract %check_tensor[] : tensor<i64>
%check_idx    = arith.index_cast %check_scalar : i64 to index

// ... all internal SM logic uses index arithmetic (unchanged from scf.while) ...

// On yield: pack scalar back to tensor
%check_i64   = arith.index_cast %new_check : index to i64
%check_new_t = tensor.from_elements %check_i64 : tensor<i64>
stablehlo.return ..., %check_new_t, ... : ..., tensor<i64>, ...
```

This avoids rewriting the SM arithmetic and keeps the `OnlineProgressOp` interface
unchanged.

### Attribute preservation through `AffineToStableHLORaising`

The `@trace checkpointing=Online(budget)` macro emits `enzyme.online_checkpointing` on a
`stablehlo.while` directly (via `Ops.jl`) — no raising pass is involved. The attribute is
**already present** on the `stablehlo.while` before Enzyme sees it. Preservation through
raising is not needed for the Reactant path.

### Registration location

In `SCFAutoDiffOpInterfaceImpl.cpp` (or the new stablehlo file), find where stablehlo
dialect registration happens (search for `StablehloDialect` in the codebase) and add:

```cpp
stablehlo::WhileOp::attachInterface<StablehloWhileOpInterfaceReverse>(*context);
```

### Remaining blocker: checkpoint storage for tensor iter-args

`scf.while` iter-args are scalars/memrefs — checkpoint buffers are `memref<N x scalar>`.
`stablehlo.while` iter-args are tensors (e.g. `tensor<1xf32>`) — checkpoint buffers must
be tensors too (e.g. `tensor<N x 1 x f32>`), accessed via `stablehlo.dynamic_update_slice`
and `stablehlo.dynamic_slice`, not `memref.store`/`memref.load`.

The `AutoDiffWhileRev` class in enzyme_ad already implements `storeSlot`/`loadSlot` for
this pattern (using `setIndex` → `DynamicUpdateSlice`). The `shCacheOnline` function
currently uses `memref::AllocOp` + `memref::StoreOp` which is invalid for tensor iter-args
and causes a segfault when Enzyme tries to differentiate the resulting IR.

**Fix needed**: Replace `memref` checkpoint storage in `shCacheOnline` with tensor-based
storage:
- Checkpoint buffers = `tensor<budget+1 x ...elem_shape>` init values carried as scaffold iter-args
- Write: `stablehlo.dynamic_update_slice` at `new_check` slot
- Read: `stablehlo.dynamic_slice` at `capo` slot

This requires changing `shCacheOnline` to carry checkpoint tensors as iter-args of the
scaffold `stablehlo.while` (not as memref allocations outside it), and using stablehlo's
`DynamicUpdateSlice`/`DynamicSliceOp` for storage.

### Testing

Once implemented, `test/Julia/reactant_online_checkpointing.jl` should pass all 6 tests
(3 IR attribute checks already pass; the 3 gradient checks currently segfault due to the
memref-of-tensor issue).

Add a new FileCheck test:
```
test/MLIR/ReverseMode/stablehlo_while_online_checkpointing.mlir
```
mirroring `scf_while_online_checkpointing.mlir` but using `stablehlo.while` syntax.
