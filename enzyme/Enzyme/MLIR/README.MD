# Enzyme MLIR

Enzyme MLIR is the MLIR-dialect automatic differentiation pass for Enzyme. It differs from the classic LLVM Enzyme backend in that it operates on MLIR operations directly — differentiating `scf.for`, `scf.while`, `affine.for`, `linalg`, `func.func`, and other MLIR dialect ops — rather than on LLVM IR. This makes it usable from compilers that target MLIR (Reactant/XLA, IREE, etc.) without first lowering to LLVM.

## Directory layout

```
Enzyme/MLIR/
├── Dialect/          EnzymeOps.td, EnzymeTypes.td — the Enzyme MLIR dialect
│                     (OnlineProgressOp, BinomialProgressOp, Cache ops, …)
├── Implementations/  Per-dialect AD interface implementations
│   ├── SCFAutoDiffOpInterfaceImpl.cpp   — scf.for / scf.while differentiation
│   │                                      including binomial + online checkpointing
│   ├── AffineAutoDiffOpInterfaceImpl.cpp
│   ├── LoopCheckpointing.h             — shared checkpointing helpers / layout structs
│   ├── CHECKPOINTING.md                — detailed checkpointing design doc
│   └── REACTANT_INTEGRATION.md        — Reactant.jl integration notes + arch gap
├── Passes/           MLIR passes
│   ├── LowerBinomialProgressPass.cpp   — lower enzyme.binomial_progress → scf/arith
│   ├── LowerOnlineProgressPass.cpp     — lower enzyme.online_progress → scf/arith/memref
│   └── Passes.td                       — pass registration
├── Interfaces/       AutoDiff op/type interfaces
├── Analysis/         Activity analysis, type analysis
└── Integrations/     C API
```

## Checkpointing overview

Enzyme MLIR supports three checkpointing strategies for loops:

| Strategy | Loop type | Attribute | Algorithm |
|----------|-----------|-----------|-----------|
| Full-tape | `scf.for` | _(none)_ | Dynamic tape via `enzyme.store`/`enzyme.load` |
| Binomial (Revolve) | `scf.for` | `enzyme.binomial_checkpointing` | Griewank-Walther offline Revolve |
| Full-tape | `scf.while` | _(none)_ | `enzyme.push`/`enzyme.pop` dynamic stack |
| Online_r2 | `scf.while` | `enzyme.online_checkpointing` | Stumm-Walther 2010 adaptive algorithm |

See [Implementations/CHECKPOINTING.md](Implementations/CHECKPOINTING.md) for the full design.

## Build

```bash
cd enzyme/mybuild1
ninja enzymemlir-opt          # builds enzymemlir-opt
ninja                         # builds everything
```

The build expects the LLVM installation at `enzyme/mybuild1` (CMake cache points to the matching LLVM build). Passes are registered via tablegen-generated code in `Passes.td`.

## Running tests

**FileCheck MLIR tests** (via lit):
```bash
cd enzyme/mybuild1
ninja check-enzyme-mlir       # runs all MLIR lit tests
# or a single test:
bin/llvm-lit ../test/MLIR/ReverseMode/scf_while_online_checkpointing.mlir -v
```

**Python numerical execution tests** (require `mlir-runner`):
```bash
python3 test/Python/exec_oceananigans_style.py
python3 test/Python/exec_gradient_schemes.py
```
These run `enzymemlir-opt` → `mlir-opt --test-lower-to-llvm` → `mlir-runner` and check numerical gradient values.

**Julia tests** (require `Checkpointing.jl` and `Enzyme.jl`):
```bash
~/.juliaup/bin/julia test/Julia/checkpointing_schemes.jl
~/.juliaup/bin/julia test/Julia/oceananigans_style_checkpointing.jl
```
Note: these Julia tests use the LLVM Enzyme backend (via `Enzyme.jl`) and `Checkpointing.jl`'s own revolve implementation — they do **not** exercise the MLIR passes above. See [Implementations/REACTANT_INTEGRATION.md](Implementations/REACTANT_INTEGRATION.md) for the Reactant-based path that does exercise the MLIR passes.
