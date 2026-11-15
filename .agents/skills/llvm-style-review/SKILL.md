---
name: llvm-style-review
description: >
  General C++/LLVM code-review principles for LLVM-based projects (LLVM, Clang,
  MLIR, and out-of-tree code built on the LLVM libraries). Each principle is
  stated generally with a worked example so you catch analogous and variant
  patterns, not just the literal case shown. Use when reviewing a PR or diff, or
  self-checking a change before review -- especially AI-/agent-generated
  changes, which repeatedly need flagging for verbose/hallucinated comments,
  em-dashes and unicode noise, fat redundant data structures,
  bool/empty-container error signaling, non-idiomatic LLVM C++,
  one-test-file-per-case sprawl, SSA-name-pinned FileCheck lines that rot,
  citations to unreleased/internal docs, and silently swallowed edge cases.
  Covers correctness/assertions, error flow, LLVM idioms, data-structure/API
  design, comments, naming, DRY, tests (including simplifying over-verbose
  fixtures), and scope/unicode hygiene. Trigger phrases: "review this PR/diff",
  "LLVM-style review", "would this pass review", "check my change before I send
  it", "simplify this test".
---

# LLVM-style code review

General C++/LLVM review principles for code built on the LLVM libraries. The
concrete vocabulary stays literal (`StringRef`, `llvm::Error`, `SmallVector`,
TableGen, `MCInst`, ...) because that is the idiom this code is written in; the
principles themselves generalize to any LLVM-based C++.

## How to read this skill

Each item is a **general principle** first, with a concrete example, and -- where
it helps -- an **"also watch for"** line naming analogous patterns. The examples
are *representative, not exhaustive*: review the spirit of the principle and flag
variants beyond the literal text shown.

These principles react strongly to AI-generated output. Several are phrased as
things to fix in an agent's standing instructions. For recurring ones, prefer
fixing the agent's standing instructions over re-flagging the same thing PR after
PR.

Durable references (git commit hashes, PR numbers with a full repo URL) are fine
to cite -- they survive a repo move and don't bitrot like line numbers. Cite by
behavior and durable identifier, never by line number.

## Guiding principles (everything below is an instance of one of these)

1. **Optimize for the reader, not the writer.** Names, comments, structure, and
   test layout exist to be understood fast. Cut anything that makes the reader
   decode or scroll.
2. **Fail loudly, never silently.** Assert invariants; surface errors through
   types that can't be ignored; never swallow, default-away, or step over a
   failure.
3. **Don't carry weight you don't need -- or do work you've already done.** No
   redundant stored state, no worst-case-sized buffers, no copies where a view
   suffices, no boilerplate where a table or abstraction fits. And no redundant
   *computation*: an expensive result several callers need is produced once and
   passed down, not recomputed inside each. (De-duplicating the call site without
   removing the repeated work misses the point -- see sec 4.)
4. **Don't write what rots.** No references to lines/positions or to the
   "previous vs current" state; comments describe what *is*, not what *was* or
   what other code *will* do. Pin durable references (commit/PR + full URL).
5. **Speak the ecosystem's language.** Reach for LLVM's existing types,
   algorithms, and idioms before hand-rolling; keep one source of truth instead
   of parallel laundry lists.
6. **Keep the change honest and scoped.** Only related edits; no drive-by
   reformatting; commit message / PR description explain *why*, written by hand,
   without jargon or unicode noise.

## How to use it

1. Read the diff **and** the commit message / PR description together.
2. Walk the sections in priority order. **Block** on sec 1 (correctness) or sec 2
   (error flow); everything else is "approve modulo comments".
3. Write findings as terse inline comments anchored to the line, often a
   one-word `suggestion`. Direct, specific, names the fix, no padding.
4. End with a one-line verdict (see template).

---

## 1. Correctness, safety & assertions -- these BLOCK

*Principle: fail loudly, never silently (P2). Assert what you assume; refuse
what you can't handle.*

- **Validate operand shape before trusting it.** A handler that keys off an
  opcode must guard/assert the operands it assumes exist. "Can this op have 0
  operands? Can it have a non-immediate operand? We will silently accept both,
  which is wrong."
  *Also watch for:* any `getOperand(i)`/`getImm()`/array index reached without
  first checking arity or kind; any `switch` without a covered-by-`default`
  guard.
- **Prefer assertions over silent fallbacks and over defensive casts.**
  Out-of-bounds should `assert` (at least on underflow) rather than silently
  returning false; a pile of narrowing casts most likely needs assertions on
  values; mask-and-truncate is usually unnecessary -- narrowing casts keep the
  least-significant bytes anyway, so what would help are assertions that the
  most-significant bytes are zero.
  *Also watch for:* `return {}` / `return false` / `return 0` on an
  impossible-by-contract path where `assert`/`llvm_unreachable` states intent.
- **A dropped mask is dropped semantics.** Removing a `& 0xFFFFFFFF` / `& 63` /
  any truncation must be proven safe and **documented in the commit
  description**. Dropping a bitmask on the floor silently changes semantics; if
  the change is safe it must be documented, and even better if it fails loudly on
  overflow. *Also watch for:* widening/narrowing, signedness flips, and
  implicit-conversion changes that silently alter range.
- **Apply the same handling to symmetric paths.** If the signed path uses a
  saturating intrinsic, check that the unsigned counterpart does too -- verify
  the semantics of the two and fix as appropriate. *Also watch for:* min/max,
  signed/unsigned, load/store, 32/64-bit, and every-other sibling call site --
  find the sibling and confirm it matches. Cross-reference by behavior, never by
  line.
- **Never silently skip on failure.** "Why is it okay to just silently go to the
  next instruction in case of failure?" Decode/parse failure must be surfaced,
  not stepped over (and ideally report how much of the input was processed if not
  all). *Also watch for:* `continue` / swallowed `Expected`/`Error` / ignored
  return values in a loop.
- **Distrust catch-alls.** A no-op catch-all that swallows a whole instruction or
  value class is dangerous -- demand a comment on why it exists and why it's
  safe. *Also watch for:* broad `default:` cases and wildcard predicates that
  quietly accept the unknown.
- **Malformed input is an assertion or an `Error`, never a silent default.** It
  is never a good idea to silently ignore malformed inputs. These must be
  assertions unless there is concrete, specific evidence that such inputs are
  fine to ignore -- at which point it must be backed by tests. A helper that
  returns `0` when a node has the wrong type is wrong: make the accessor return
  `std::optional` (no in-band `0`) and make the iterator return `llvm::Error` on
  a malformed structure instead of skipping. *Also watch for:* any
  parser/accessor that maps "shape I didn't expect" onto a valid-looking value
  (`0`, empty, first-enum-member); a tolerance not justified by a test is a bug,
  not leniency.

### Memory & resource safety (also blocking)

- **No fixed-length C buffers.** `char mnemonic[128]` / `char detail[1024]` is a
  literal buffer overflow waiting to happen. Use `std::string`/LLVM string types.
  *Also watch for:* `strcpy`/`sprintf`/manual `memcpy` into fixed arrays.
- **RAII for every resource.** `mmap`/temp files/fds get a `llvm::ScopeExit` (or
  equivalent) at acquisition, instead of hoping every exit path cleans up. *Also
  watch for:* any acquire whose release is duplicated across `return` paths.
- **Don't encode structure into a flat string.** Building a map key by
  concatenating fields with `'\0'` is dubious and dangerous if it's ever
  converted to a C string -- use a `std::tuple`/struct + `DenseMapInfo`.
- **Distinct exit codes / error values for distinct failures.**
- **Re-check new degenerate cases a refactor introduces.** "It looks like this
  can be empty string now? Is that fine?" *Also watch for:* newly
  nullable/optional/zero-length values that downstream code assumes non-empty.

## 2. Error handling & diagnostics -- these BLOCK when the flow is wrong

*Principle: fail loudly (P2). An error must carry information and be impossible
to ignore.* This is a whole-PR concern: the overall error flow matters as much
as any single site.

- **Don't signal errors with `bool` or magic values.** Use `llvm::Error` /
  `ErrorOr` / `LogicalResult` / `FailureOr` (they carry info and are
  `nodiscard`); bias against `bool`. LLVM had a huge discussion on whether `true`
  means error or success -- don't reopen it. *Also watch for:* sentinel returns
  later string-matched, out-params used as status, a status field the caller must
  remember to read (action-at-a-distance).
- **Empty containers are not an error signal.** An empty vector could be valid --
  return `ErrorOr<SmallVector<...>>` / `FailureOr<...>`. *Exception:* a
  **naturally-nullable** return is fine where null/empty unambiguously means
  "absent/failed" in context (e.g. a pointer that is null on lookup-miss).
  Conversely, **don't** wrap a type that already has a natural empty/false
  value -- avoid `FailureOr<bool>` or `FailureOr<T*>` unless you genuinely need
  the tri-state (a legitimate null/false distinct from failure).
- **Prefer typed results over magic strings.** Rather use `Expected<StringRef>`
  than a magic string that is then string-matched to detect error.
- **`errs()` is not a diagnostic channel.** Don't dump to the error stream unless
  there is an actual error -- `LLVM_DEBUG`/`LDBG` exists for debug output; tell
  errors and warnings apart. Don't pass a null stream to swallow real messages.
  *Also watch for:* the test analogue -- `2>/dev/null` in lit RUN lines (sec 8).
- **Centralize fatal reporting.** Inconsistent fatal prefixes -> a single
  reporting helper that prefixes consistently and gives one breakpoint. Mark
  genuinely non-returning helpers `noreturn`; use `llvm_unreachable` for
  unreachable switch arms.
- **Don't suppress-then-continue.** Code that ignores a callee's parse failures
  and continues anyway should be defensive about it.
- **Log at the entry point, not deep in the helpers.** A deep helper that writes
  to the error stream itself is in the wrong place -- normally the entry-point
  function should be the only place that logs. Deep helpers *return*
  `Error`/`Expected` and let it propagate; the top-level function decides whether
  to log, report, or `consumeError` (a best-effort read should `consumeError`
  quietly, not log). *Also watch for:* a helper called from several places that
  logs on its own -- the caller can't suppress it, and you get
  duplicate/misplaced diagnostics.
- **Return `Expected<T>` filling the struct directly -- no byte-buffer + memcpy
  dance.** A reader that copies a section into a `std::vector<uint8_t>` and then
  `memcpy`s fields out should instead read straight into the typed struct and
  return `Expected<T>`. If you only care about four fields, don't copy the whole
  region. *Also watch for:* out-param + `bool`, or an intermediate buffer that
  exists only to be re-parsed one line later.

## 3. LLVM idioms & coding standards

*Principle: speak the ecosystem's language (P5).* This is LLVM code -- write it
like LLVM. Coding standards are not just formatting
(<https://llvm.org/docs/CodingStandards.html>). Representative items; the general
rule is **reach for what LLVM already provides before hand-rolling it**:

- **Strings.** `llvm::StringRef` over `const std::string&`; **never**
  `const char *` (avoids C-string length counting). String constants ->
  `llvm::StringLiteral`/`constexpr StringRef`. Build with `SmallString` + its
  `raw_ostream`; prefer a streaming/`toString` form over materializing a
  `std::string`. *Also watch for:* any silent `std::string` allocation on a hot
  path.
- **Containers.** `SmallVector` over hardcoded C arrays (a worst-case-sized fixed
  array per instruction is wasteful); `ArrayRef`/`MutableArrayRef` as the
  pointer+size API abstraction; `std::array` when the size is fixed and should be
  enforced at the API level. **Never hardcode a `SmallVector` inline size without
  reason** -- prefer the default `SmallVector<T>` (let LLVM pick the inline
  count) unless a specific N is justified by a documented, concrete typical-size
  argument. The point is *don't invent* a count, not *tune* one. *Also watch
  for:* any worst-case-sized buffer (P3).
- **Algorithms.** `llvm::find`/`find_if`, `DenseMap::lookup`/`lookup_or`,
  `interleaveComma`/`interleave`, `emplace_back`. *Also watch for:* any
  hand-written loop reimplementing a `<algorithm>` or `llvm/ADT` primitive.
- **Loops.** Don't evaluate `end()` every iteration; declare loop-local variables
  inside the loop instead of resetting them.
- **`auto`.** Almost always avoid in LLVM; expand unless the type is obvious from
  statement-level context.
- **Includes.** Prefer forward declarations over `#include`; no relative include
  paths -- start from the project root; a file-scope `using namespace llvm;` is
  acceptable in `.cpp` files (not headers).
- **File-local helpers are `static`, not wrapped in a file-wide anonymous
  namespace.** Anonymous namespaces are for class/type decls; free functions with
  internal linkage use `static`
  (<https://llvm.org/docs/CodingStandards.html>). *Also watch for:* a big
  `namespace {` spanning helpers and free functions both.
- **`raw_ostream`/`formatv` over string concatenation for messages.** Use
  `formatv`/`Twine` or stream into a `SmallString`'s `raw_ostream`; a hand-rolled
  hex-formatting string helper is replaced by `formatv("{0:x}", ...)`. (Overlaps
  Strings above.)
- **Sorted data -> take the adjacent element, don't binary-search.** If a
  container is sorted, take the next element (`Items[I+1]`) instead of calling
  `upper_bound` to find it. *Also watch for:* any
  `lower_bound`/`upper_bound`/`find` over a container you *just* sorted or are
  already iterating in order.
- **Reference the owning memory; don't copy heavy bytes.** "Why are we copying
  the contents? It's heavy, we should store a reference instead unless there's a
  reason!" A section-bytes struct exposes `llvm::ArrayRef<uint8_t>` into the
  owning `MemoryBuffer` rather than a `std::vector<uint8_t>` copy. State the
  lifetime contract in the doc comment (the view is valid as long as the owning
  buffer is). *Also watch for:* copying a `StringRef`/`ArrayRef` payload into an
  owned container "to be safe" when the owner outlives the view. (Overlaps
  sec 4.)
- **Naming idioms.** Capitalized-acronym locals for class types (`MCID`, `MCIP`,
  `STI`, `DL`); mirror the LLVM name of the thing. Don't micro-optimize
  identifiers by dropping a character ("saving one character, really?").
- **Bitfields / optionals.** Pack genuinely-needed flags via `llvm::Bitfield`;
  use `std::optional` (or steal a bit from a larger int) instead of a magic
  `-1`. A "present?" bool paired with a container/value is a `std::optional` of
  that value: turn `bool HasDims + array` into `std::optional<std::array<...>>`.
  *Also watch for:* any `bool HasX; T X;` pair, or a caller that must read two
  fields to know if one is valid.
- **Misc.** `llvm::sys::SmartMutex` for locking; 80-column limit; default LLVM
  address space is 0 -- don't hardcode/require it; be consistent within a file
  between fixed-width ints and `int`/`unsigned`.

## 4. Data-structure & API design

*Principle: don't carry weight you don't need (P3); derive, don't store.*

- **Don't store fat, redundant per-instance state.** A decoded-instruction struct
  that duplicates fields already in the `MCInst` it stores is needlessly fat --
  those fields should be accessor functions. Don't keep a shadow copy of
  `STI`/subtarget fields -- query the source. *Also watch for:* any cached field
  recomputable from data the object already holds.
- **Per-opcode vs per-instance.** Info that depends only on the opcode belongs in
  a helper that dispatches on it, not stored per-instance except as a
  micro-optimization on a hot path (which requires data before implementing it).
  If caching is truly justified, use a look-aside table + pointers, not piecemeal
  per-instance allocations.
- **Don't store textual mnemonics per instance** (let alone twice). Use an
  enum/opcode, or regenerate text on the fly from the `MCInst`; if text must be
  materialized, render once into one buffer and store `StringRef`s into it. On a
  stored `std::string` mnemonic used only in diagnostics: reconstruct it whenever
  a diagnostic is printed rather than carrying it around the whole stack. **The
  general rule: if a field is derivable from something the object already stores
  and is only read on a cold/diagnostic path, don't store it -- recompute it
  there.**
- **Avoid string-parsing of mnemonics entirely.** There is usually a better way
  to obtain the same information from LLVM. String-matching to extract semantics
  or detect errors is fragile -- flag it unless truly unavoidable. *Also watch
  for:* substring/prefix tests standing in for a structured query.
- **Minimal, honest APIs.** Values used once and passed to a function should be
  constructed locally there; a stored size field redundant with `.size()` should
  go; if a struct only references bytes the object already owns, take a
  `MutableArrayRef` rather than copying; make an invariant unambiguous rather
  than relying on flag combinations (when a value is a conditional branch, will
  `IsBranch` still be `true`? -- just dispatch through a helper).
- **Don't do the same work twice -- compute once, pass the result down.** Any
  expensive, deterministic result (a parsed object, a resolved section, a built
  lookup table, a metadata document, a sorted list) that several functions all
  need should be produced *once* by the common caller and handed to each as an
  argument -- not recomputed inside every function that needs it. Push the work
  up to where it can be shared, and take the *result* as a parameter. For
  example, when every entry point re-parses the same ELF buffer, change the entry
  points to take `object::ObjectFile &Obj` and have the caller parse once.
  **The anti-fix to reject: wrapping the repeated work in a helper that every
  function still calls.** De-duplicating the *call* doesn't remove the repeated
  *work* -- the goal is do-the-work-once, not spell-it-once. This is distinct from
  sec 7 DRY (which is about not repeating *source*): here identical source that
  each re-runs an expensive step is still a bug even though it isn't copy-paste.
  *Also watch for:* re-`parse`/`create`/`open`/`load`/`sort`/`build` of the same
  input across sibling functions or across loop iterations; a "convenience"
  wrapper whose only effect is to re-do shared work per call; recomputing in a
  callee what the caller already has in hand.

## 5. Comments -- delete noise, document entities, never lie

*Principle: optimize for the reader (P1); don't write what rots (P4).*
"We are not paid per LoC." (See the `llvm-comment-hygiene` skill for the full
treatment.)

- **No verbose/restating comments.** No tautologies above a declaration; no
  noise that restates the next line.
- **Document all top-level entities -- systematically.** Every top-level function
  and struct field. *Also watch for:* a doc comment that, by placement, silently
  attaches to the wrong declaration or is attached to nothing.
- **Don't write references that rot.** Never reference a line in another file --
  this bitrots faster than you blink; no "previous vs current" state ("we now use
  X instead of Y", "introduced in this commit"); don't describe what other code
  "will do". That history belongs in the commit message. *Durable references are
  fine and encouraged:* git commit hashes and PR numbers are immutable -- cite
  them with the **full GitHub URL** (org/repo so it survives a repo move).
- **No invented terminology or leftover agent text.** Comments that read like
  leftover agent instructions, mismatch the filename, use platform jargon loosely
  (say `SIGABRT`/`abort`, not "SIG6"), or coin shorthand for a real concept
  should be cut. Use the spec's wording; state the fact plainly.
- **A comment must match the code beside it.** Delete comments the code has
  outgrown; don't leave a description of code that isn't there. *Also watch for:*
  a comment that survived a refactor and now narrates the old shape.
- **Struct-field docs go next to the field, in plain words.** Put a one-line doc
  on each field, saying what it *means*, not a distant paragraph of coined terms.
- **Delete LLVM boilerplate a reviewer already knows.** Don't narrate
  `isa`/`cast`, `Expected` unwrapping, or `ErrorInfo` plumbing; assume LLVM
  fluency. Reserve comments for what's genuinely non-obvious.
- **Don't cite unreleased or internal docs.** A reference to an internal or
  not-yet-public specification (e.g. an unreleased hardware programming guide) is
  not a durable public reference and must not ship in code, comments, tests, or
  commit messages -- replace it with the layout/behavior stated directly, a
  pointer to the code that implements it, or a public spec. Durable references
  remain commit/PR + full URL; internal/unreleased docs are not durable
  references.
- **Explain the genuinely non-obvious.** Magic address computations, domain terms
  ("tied"/"tied-to-def"), and any deviation from a documented formula (a
  micro-opt InstCombine/ISel won't do) *do* warrant a comment. Note which `#if`
  an `#endif` closes.

## 6. Naming

*Principle: optimize for the reader (P1). A name should not need decoding.*

- **No cryptic abbreviations. This is the single most-repeated review comment --
  treat it as a blocking pre-submit gate, not an afterthought.** `shamt` ->
  `ShiftAmount`; `KD` -> `KernelDescriptor`; `CB` -> `Callback`; `TsFlags` ->
  `TargetSpecificFlags`; `numOps()` -> `numOperands()`. Ops are not Operands.
  Avoid unnecessary abbreviations like the plague, here and everywhere.
  - **Mandatory pre-submit scan.** Before pushing, read every identifier and
    comment token you added or renamed and expand anything that isn't a full
    word: `Op`->`Operand`, `Ops`->`Operands`, `Idx`->`Index` (unless matching an
    LLVM API), `Inst`->`Instruction` in *your* names (LLVM's own
    `MCInst`/`TSFlags` stay), `Mnem`, `Insn`, `Desc`, `Attr`, `Ctx`, `Regs`,
    `Msg`, `Buf`, `Fn`, `Ptr`, `Cnt`, `Tmp`, `Cur`, single-letter non-loop
    locals. This applies to **prose too** -- no `insn`/`max insn` in comments *or*
    chat.
  - The fix is **global**: expand every occurrence in the change, not just the
    flagged site -- a half-renamed identifier reads worse than a consistent one.
  - The only exceptions are (a) an existing LLVM API name you're calling
    (`MCInst`, `MCInstrDesc::TSFlags`, `STI`, `MCID`) -- mirror it verbatim, don't
    coin new ones; (b) established loop indices `I`/`J`/`K`.
- **Names must be accurate, not misleading.** Don't call something `sign` when
  it's a bag of bits; don't use a term the reader can't decode. *Also watch for:*
  names that drifted from what the value now holds.
- **Encode the unit/semantics in the name.** A bare `Width` makes the reader
  guess the unit; `WidthInBytes` is self-describing (and gets a doc comment --
  sec 5). *Also watch for:* counts/sizes/offsets/strides whose unit (bytes vs
  words vs elements) isn't in the name.
- **Don't churn user-visible IR value names** without a functional reason -- pure
  diff noise.
- **No magic numbers.** Replace `63` with a named constant defined to reveal
  intent -- `(1 << 6) - 1` ("low 6 bits"). Use a named constant/`std::optional`
  instead of `-1`.

## 7. DRY & control flow

*Principle: don't carry weight you don't need (P3).*

- **Share identical blocks.** "This code is identical to the block above,
  consider refactoring." Repeated CMake link blocks -> a helper, renamed for
  clarity. *Also watch for:* near-duplicates that differ only in a constant --
  parameterize.
- **Use early exit** instead of nesting the body in an `if`.
- **The macro/boilerplate smell.** "Most of these are 1:1 renamings -- can we
  capture this better than a huge laundry list of boilerplate?" (see sec 10 for
  the TableGen direction).

## 8. Tests

*Principle: optimize for the reader (P1); don't write what rots (P4); don't
carry weight you don't need (P3).* A recurring complaint on generated tests is
verbosity -- "N lines of mostly boilerplate per case is way overboard."

- **One file per case is unreasonable.** Group related cases in a single
  file/function (all CMP, all min-max, related folds); put a new test next to the
  existing one for the related op. *Also watch for:* one function per case where
  one FileCheck pass over all of them works.
- **Never pin exact SSA value names.** "One shalt not match SSA names." A CHECK
  like `%conv = call i64 @llvm.foo.i64` rots the instant the producer renames the
  value. Match the name with a wildcard and bind the *relationship that actually
  matters* with a capture variable:
  `%{{.+}} = call i64 @llvm.foo.i64(i1 [[CARRY]])`. When a test breaks because a
  name drifted, fix it with a wildcard, not by re-pinning the new name (that just
  resets the rot clock). This is P4 applied to FileCheck. *Also watch for:*
  literally-pinned anonymous `%N` numbered values; over-tight operand matches --
  constrain only the operand the test is about (bind the operand under test,
  leave the rest a wildcard so an `undef` source still matches). **Keep every
  match/capture regex plain `.+`** -- this applies to *both* `[[NAME:...]]`
  captures and inline `{{...}}` matches. Write `[[LD:%.+]]` (not
  `[[LD:%[a-zA-Z0-9_.]+]]`) and `bb{{.+}}:` (not `bb{{[0-9]*}}:`); the exact set
  of legal characters is not what the test is about, so spelling out a character
  class is just noise that can drift. Note `.+` requires at least one character,
  which is usually what you want (e.g. `bb{{.+}}:` matches the numbered `bb7:` and
  skips a bare `bb:` -- often the more precise match).
- **CHECK lines go immediately above (or beside) the input they verify.** Inline
  them next to the instruction whose lowering they check, so no separate
  explanatory comment line is needed. Put function-level checks (the `*-LABEL:`
  line) at the function *definition label*, and the per-instruction checks (the
  lowered call, its result) right above that instruction's line -- not all hoisted
  into a top-of-file block. FileCheck collects same-prefix lines in file order
  regardless of physical position, so relocating them next to the source is free
  as long as their relative order still matches the emitted output. A long header
  that re-derives the mechanism is the test analogue of a verbose comment
  (sec 5): replace it with a one-line statement of the invariant plus a pointer
  to the responsible function/file -- reference the code, don't replicate it (P4).
  - *When a check has no line to sit next to* (it verifies synthesized output
    with no 1:1 source instruction -- a remapped id, a function attribute, an
    entry-block seed, a stderr diagnostic): put those checks immediately after the
    RUN line that produces them (the RUN *is* the input they verify), not in a
    separate far block. Still inline the ones that *do* map to an instruction.
    Mixed pass/fail fixtures: keep each RUN's `--check-prefix` block with that RUN.
- **Trim the test to what it proves** (P3). Strip metadata the test doesn't
  exercise and minimize the body: when the test checks a *lowering shape* rather
  than data flow, `undef` operands are enough -- don't load real inputs to feed a
  value you never check. Don't over-specify the matching either: reach for plain
  `CHECK` rather than `CHECK-DAG` where SSA def-before-use printing already orders
  the matches (a check that references a captured value must print after the check
  that bound it), so `-DAG` adds nothing -- reserve it for genuinely
  order-independent emissions. **But** keep the whole structure when the structure
  *is* the regression -- a CFG-shape bug needs its real control flow, not a
  reduced body.
  - **When trimming a copied fixture, cut the boilerplate but not the
    load-bearing parts.** A copied fixture drags in structure the case under test
    never touches (unused function arguments, metadata the verifier ignores,
    alignment/padding directives, ABI plumbing for values never read) -- cut it.
    But some seemingly-boilerplate lines are what make the input *valid*: headers
    a parser needs to recognize the object, and metadata fields a verifier
    *requires*. Dropping those doesn't shrink the test, it silently changes the
    outcome (an assembly/parse error, or a flipped pass/fail). Diff a bloated
    fixture against the smallest passing fixture already in the directory to see
    exactly what is safe to delete, and **always re-run the test after trimming**
    -- over-trimming shows up as a parse/verify error or a changed outcome, not a
    silent pass.
- **Don't redirect stderr to `/dev/null`** (`2>/dev/null`). Errors must be
  visible, and error-text CHECK lines can't match if stderr is discarded; `2>&1`
  (merge into the checked stream) is the see-it choice. Especially for
  refuse/negative tests. And never `2>&1 >/dev/null` (that redirects stdout to
  the null file *after* stderr was already dup'd to the terminal -- the opposite
  of what you want); use plain `2>&1`.
- **Strip distracting boilerplate** (marker comments, machine noise); no
  gratuitous duplicate instances ("why 3 instances of the same instruction?").
- **Add a test for every new code path.** Prefer real unit tests, not just an
  integration corpus -- an end-to-end corpus is an integration test at best; keep
  bodies short and useful, not excessively long.
- **The test driver is a legitimate vehicle for test-only flags.** A test-only
  CLI tool is fine to add flags to. A focused `--dump-...` mode that prints
  exactly what a change introduces lets a lit fixture cover the new path with no
  separate gtest -- keep the added driver code minimal and the printed surface
  small. *Also watch for:* reaching for a whole new gtest+build target when one
  flag on the existing driver + a lit fixture proves the same thing more cheaply.
- **In a stacked series, verify each commit's *contents*, not just that the tip
  builds.** A build can pass off the working tree while a file is missing from the
  commit that should introduce it (a bad `git add` pathspec silently dropped a
  source). `git show --stat` / `git show` each commit and confirm it contains what
  its message claims and builds *in isolation*; a green tip is not evidence that
  commit N is self-contained.
- **Verify the simplified test still passes before claiming done.** Rebuilding is
  part of the loop: a branch switch or a source edit silently invalidates the
  binary, so rebuild the relevant target, then run the lit test. Empty output or
  a zero-work run is not a pass -- confirm the CHECK lines actually matched.

## 9. Scope, formatting & unicode hygiene

*Principle: keep the change honest and scoped (P6).*

- **No unrelated changes.** Bundled, unexplained edits that seem unrelated and
  aren't explained in the commit message -- split or justify.
- **No spurious reformatting.** Restrict clang-format to the current diff --
  whole-file reformatting destroys commit history. Keep CMake dependency lists
  sorted alphabetically.
  - **Never run `clang-format -i` on a whole file you only partly changed** -- it
    reflows unrelated lines that were committed in a non-canonical form. Use
    `git clang-format <base>` (formats only changed lines). If you did run
    `clang-format -i`, catch the damage before pushing: for each file, split
    `git diff --unified=0 HEAD` into hunks and flag any whose removed and added
    lines are equal after stripping whitespace -- those are format-only reflows of
    lines you didn't touch; revert them to the committed form.
- **No spurious comment churn either.** A comment edit you don't need is diff
  noise exactly like a reformat; leave untouched text alone.
- **But *do* format the lines you touched.** A clang-format CI check runs
  `git-clang-format <base>` and fails on any diff, so an unwrapped call or
  over-long comment on a line *you added* blocks the PR. Self-check before
  pushing: `git-clang-format <base>` over your diff should be a no-op. Two
  subtleties that waste time: (a) it enforces the *repo's* clang-format version,
  not whatever is on your `PATH` -- when a failure looks spurious, reproduce with
  the pinned version. (b) the base is the *PR base branch tip*, so a stale base
  (branch not rebased onto its target) makes `git-clang-format` reformat lines the
  base already moved past -- a red check that is really "rebase me," not "fix
  formatting." Rebasing onto the current base clears it.
- **No em-dashes, unicode arrows, or "unicode nonsense"** in code, comments, or
  error messages. Bake it into the agent's instructions; grep `[^\x00-\x7F]`
  before committing.
- **One spelling convention** -- American or British, not both in a directory.
- **Commit message / PR description.** Behavior facts belong in the commit
  message, not a random test comment. Update the PR description to describe the
  revised change and *why*; avoid excessive terminology; write it manually. If
  it's agent-generated, configure the agent to generate less verbosity.
- **No `## What / ## How / ## Test` template -- write flowing prose in the
  author's own voice.** The boilerplate heading structure is agent slop that most
  reviewers hate; strip it. State the problem first as a plain sentence, then
  explain the mechanism and the *why* in connected paragraphs, and fold
  verification and stacking into ordinary sentences at the end ("I verified ...
  end-to-end"; "full suite N/N green"; "stacked on #NNN -- review that first").
  First person is fine and usually reads more naturally than the passive
  template. Match the *committing author's* register -- mirror the voice already
  visible in their recent commits on the same branch, not a fixed house style.
  Keep it ASCII: `->` and `--`, never unicode arrows/em-dashes (sec 9). This is a
  standing default -- generate messages this way from the start rather than
  reformatting a templated draft.

## 10. Design-level / forward-looking (raise even if "not for this commit")

*Principle: one source of truth (P5); derive, don't duplicate (P3).*

- **Single source of truth via TableGen.** Generate the opcode->handler map from
  TableGen inputs -- one source shared with the lowering path, **compile-time**
  completeness checking instead of runtime, and far less boilerplate. 1:1-renaming
  macro "laundry lists" are the smell; look for an abstraction (an enum + a
  verifier). Any hand-maintained list -> alphabetical under `// clang-format off`.
- **Provenance of enumerations.** For a hand-written op list -- where do these
  come from, what's the relationship to the lowering-pass pseudos, how does one
  add more, and how does one tell an *intentional* omission from a missing one?
  Prefer a programmatic association over prose in comments.
- **Query, don't shadow** (sec 4): query `STI`/LLVM rather than caching copies,
  "not for this commit" notwithstanding.
- **Don't conflate "feature flag missing" with "same target."** Absence of a
  feature does not identify an architecture.

---

## Verdict template

```
<one line: what the change does, and whether it's correct>

Blocking:
- <correctness/safety (sec 1) or broken error flow (sec 2), if any>

Comments (address before/at merge):
- <sec 3-9 items, each anchored to a file + durable ref with the concrete fix>

Design (consider, may be follow-up):
- <sec 10 items>

Verdict: CHANGES_REQUESTED | LGTM modulo comments
```

Block for sec 1/sec 2 (or unexplained unrelated changes). Everything else is
"okay modulo comments" -- approve with the fixes listed. Remember the cited cases
are examples of each principle, not the whole of it: flag the variant, not just
the verbatim.
