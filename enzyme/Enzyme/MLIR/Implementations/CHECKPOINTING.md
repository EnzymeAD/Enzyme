# Enzyme MLIR Checkpointing Design

This document covers all three checkpointing strategies implemented in
`SCFAutoDiffOpInterfaceImpl.cpp` and their associated lowering passes.

---

## 1. Binomial checkpointing for `scf.for`

**Trigger:** `enzyme.binomial_checkpointing` unit attribute + `enzyme.checkpoint_period = N : i64`

**Algorithm:** offline Revolve (Griewank & Walther 2000, Algorithm 799). Trip count is
known at differentiation time (from the for-loop bounds), so the optimal checkpoint
schedule can be precomputed.

### Forward pass (`cacheForBinomial`)

- Emits `enzyme.binomial_progress` op each iteration to query the Revolve state machine.
- On STORE action: writes carried values + loop counter into static `memref<N×T>` buffers.
- Buffers are allocated with `N = checkpoint_period` slots.

### Reverse pass (`reverseForBinomial`)

- Outer `scf.for` over all primal steps in reverse.
- Inner `scf.while` runs offline Revolve reconstruction to find the nearest checkpoint,
  then re-executes the primal forward from that checkpoint to the target step.
- Adjoint of the body is applied once reconstruction delivers the pre-step state.

### Lowering: `LowerBinomialProgressPass`

`--lower-enzyme-binomial-progress` lowers `enzyme.binomial_progress` to `scf`/`arith` ops
implementing the Revolve state machine.

---

## 2. Online_r2 checkpointing for `scf.while`

**Reference:** Stumm & Walther (2010), "New algorithms for optimal online checkpointing,"
SIAM J. Sci. Comput. 32(2), 836–854.

**Trigger:** `enzyme.online_checkpointing` unit attribute + `enzyme.checkpoint_period = budget : i64`

**Why online:** `scf.while` loops have an unknown trip count at differentiation time.
The Online_r2 algorithm places checkpoints adaptively as the loop runs, using a budget
of `budget` checkpoint slots. It is optimal in the sense that for a budget-`b` run it
handles up to β(b) = `(b+2)(b+1)/2 − 1` steps with the minimum number of re-executions.

### Forward scaffold (`cacheWhileOnline`)

The primal `scf.while` is replaced by a scaffold while that carries:

```
(primalArg₀, …, primalArgₙ,   -- primal loop-carried values
 numFwd : index,               -- count of body iterations executed
 sm₀, …, sm₉ : index)         -- Online_r2 state-machine scalars (10 total)
```

plus three `memref<max(budget,3)×index>` arrays for the SM internal state.

Each iteration calls `enzyme.online_progress` (an Enzyme dialect op defined in
`EnzymeOps.td`) which implements one step of the Online_r2 state machine and returns:

```
action    : index   -- 0 = terminate (online→offline handoff)
                    -- 1 = STORE a checkpoint
                    -- 2 = ADVANCE (continue without storing)
new_check : index   -- slot to write the checkpoint into (on STORE)
… (8 more SM state outputs)
```

**STORE branch:** writes carried values + `numFwd` into `checkpointBuffers[new_check]`.
Slot index is `aa[2]` (the `new_check` output), NOT a monotonic counter — the algorithm
may reuse lower-indexed slots on r=2 actions.

**Capacity assertion:** before the loop, the maximum supportable steps
β = `(budget+2)(budget+1)/2 − 1` is computed. An `cf.assert` fires if `numFwd > β`.

**Buffer sizing:**
- Checkpoint buffers: `budget + 1` slots (extra slot prevents inner Revolve overflow).
- SM arrays: `max(budget, 3)` elements (loads of `ch[1]` and `ch[ind=2]` are speculative
  even for budget=1).

**Before-block condition:** `shouldContinue = primalCond` — NOT gated on `action ≠ 0`.
The `action=0` signal means "hand off to offline Revolve", not "stop the forward loop";
the primal loop must run to natural termination.

### Reverse reconstruction (`reverseWhileOnline`)

After the forward scaffold, `numFwd` holds the total number of body iterations.

**Outer `scf.for`** iterates from 0 to `numFwd` in reverse. At step `i` (targeting primal
step `numFwd − i`):

1. **Scan loop:** iterate over all `budget` checkpoint slots and find the slot `capo` whose
   stored step value is the largest value ≤ `targetStep`. This is the equivalent of
   Julia's `storemap`/`storemapinv`. Required because r=2 actions write to non-monotonic
   slot indices.

2. **Load checkpoint** at slot `capo` to restore primal carried values.

3. **Inner `scf.while`** runs offline Revolve (via `enzyme.binomial_progress`) from the
   loaded checkpoint to `targetStep`, reconstructing the pre-step primal state.
   Uses `totalBudget = budget` (a constant) for the Revolve budget, not the scaffold's
   runtime `sp` counter.

4. **Adjoint** of the while body is applied using the reconstructed state.

### Lowering: `LowerOnlineProgressPass`

`--lower-enzyme-online-progress` lowers `enzyme.online_progress` to plain `scf`/`arith`/
`memref` ops. It is a direct port of PyRevolve's `Online_r2::revolve()` function:

- **Branch 1** (`r=1`, TAKESHOT): store a checkpoint.
- **Branch 2** (`r=1`, ADVANCE): advance without storing.
- **Branch 3a** (`r=2`, default): normal r=2 action.
- **Branch 3b** (`r=2`, `snaps=1`): when only one snapshot slot remains, return ADVANCE
  instead of TAKESHOT to avoid overwriting the only checkpoint with an infinity sentinel.
  Without this fix, budget=1 produces wrong gradients for all n ≥ 2.

**Init loop fix:** for `snaps < 4`, the `ord_ch` array must be initialized to `ord_ch[i] = i`
in addition to `num_rep[i] = 2`. Missing this caused garbage in the SM state for budget=1.

---

## 3. Full-tape (uncheckpointed) `scf.while`

**Trigger:** presence of `enzyme.while` without `enzyme.online_checkpointing` (i.e. the
default when no checkpointing attribute is set).

### Forward scaffold (`cacheWhileFullTape`)

- Replaces the primal while with a scaffold while carrying `(primalArgs…, numIters)`.
- **Before-block:** `enzyme.push` active carried values to a dynamic cache (LIFO stack)
  on every condition evaluation — including the final failing check.
- **After-block:** increments `numIters`.
- After the while: `enzyme.push(numIters)` to a separate cache.

### Reverse pass (`reverseWhileFullTape`)

- Pops `numIters` from its cache.
- **Discard pop:** pops and discards the top entry of each active-arg cache. This entry
  corresponds to the values pushed during the final (failing) condition check; these are
  the loop outputs, not the pre-step inputs needed for the adjoint.
- **`scf.for` 0..numIters:** each iteration pops the pre-step state (in LIFO = reverse
  chronological order), re-runs the body forward to recompute intermediates, then
  applies the adjoint.

**Key bug fixed:** without the discard pop, iteration `i=0` of the reverse loop pops the
loop output (state after all body steps) instead of the state entering the last body step,
producing completely wrong gradients.

---

## Correctness bugs fixed

| # | Bug | Fix |
|---|-----|-----|
| 1 | `done`-flag removed by canonicalize | Removed `done` entirely; detect termination from primal condition directly |
| 2 | Gradient collected from wrong block args | Collect from `primalAfter->getArguments()` (after-block), not before-block |
| 3 | STORE writes to `sp` instead of `new_check` | Use `checkSlot = aa[2]` (algorithm's chosen slot) |
| 4 | Inner Revolve uses `sp` (may be < budget) for `budgetRem` | Use `totalBudget = arith::ConstantIndexOp(budget)` |
| 5 | Scan loop picks wrong checkpoint for n>10 budget=4 | Implement step→slot scan (Julia's `storemap`) instead of `capo = sp − 1` |
| 6 | `action=0` causes early forward termination | `shouldContinue = primalCond` only; `action=0` does not stop the primal |
| 7 | Checkpoint buffer overflow (inner Revolve writes slot `budget`) | Allocate `budget + 1` slots |
| 8 | budget=1 guard rejects legal inputs | Changed `budget > 1` guard to `budget >= 1` |
| 9 | SM arrays too small for budget=1 (speculative loads at index 1, 2) | `smSize = max(budget, 3)` |
| 10 | `ord_ch` uninitialized for snaps<4 | Added `ord_ch[i] = i` in the init loop |
| 11 | budget=1 n=2 wrong gradient (0.0) | snaps=1 r=2 TAKESHOT returns ADVANCE instead |
| 12 | Full-tape reverse off-by-one (pops output not input) | Discard the extra top push (from failing condition check) before the reverse for loop |

---

## β optimality bound

```
β(budget) = (budget + 2) * (budget + 1) / 2 − 1
```

| budget | β  | Handles n ≤ |
|--------|----|-------------|
| 1      | 2  | ∞ (never returns action=0; see snaps=1 fix) |
| 2      | 5  | ∞ (same reason for all n in practice) |
| 3      | 9  | ∞ |
| 4      | 14 | 14 (asserts for n=15+, matching Julia) |
| 5      | 20 | 20 |
| 10     | 65 | 65 |

budget=1 is asymptotically O(n²) re-executions — degenerate but correct.
