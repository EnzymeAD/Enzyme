---
name: llvm-coding-standards
description: >
  LLVM project coding standards and the canonical "pick the right data
  structure / API" guidance, distilled from CodingStandards.md and
  ProgrammersManual.md. Apply this WHENEVER editing, adding, or reviewing C++,
  TableGen, or lit test source in Enzyme (`enzyme/Enzyme/`, `enzyme/BCLoad/`,
  `enzyme/tools/`, `enzyme/test/`) or within an llvm-project checkout. Covers
  C++ standard version, source
  formatting (80-col, includes, header guards, comments), language rules (no
  RTTI/exceptions, assertions, llvm_unreachable, struct vs class), naming
  conventions, early-exit/style idioms, and the preferred LLVM data structures
  and APIs (SmallVector/ArrayRef/DenseMap/StringMap/StringRef/Twine,
  isa/cast/dyn_cast, llvm::Error/Expected, function_ref, LDBG/LLVM_DEBUG,
  Statistic). IMPORTANT: subprojects and individual directories add their own
  overrides -- for `enzyme/Enzyme/MLIR/` and `enzyme/test/MLIR/`, also apply
  `mlir-coding-standards` which overrides naming, `const`, Doxygen, and MLIR
  idioms; always check local `.clang-format` and surrounding code first. Trigger
  phrases: "is this
  LLVM-idiomatic", "which container/data structure should I use", "does this
  follow LLVM coding standards", "right way to pass a string / signal an error
  in LLVM".
---

# LLVM coding standards & data-structure selection

Apply this to **every change to C++, TableGen, or lit test source in Enzyme**
(or an `llvm-project` checkout). Automated checks for these rules live in
`enzyme/.clang-tidy` (run via `enzyme/scripts/run_clang_tidy.py`) for C++ AST
rules and `enzyme/scripts/check_llvm_coding_standards.py` for non-AST macro,
comment, and `.ll` lit-test rules. The authoritative source
documents are bundled in `reference/`:

- `reference/CodingStandards.md` — the full LLVM Coding Standards.
- `reference/ProgrammersManual.md` — useful APIs + the full "Picking the Right
  Data Structure" catalog.

This SKILL.md is the fast path: the rules you need for most edits. When a case
is not covered, or you need the full rationale / a container not listed here,
`grep` the bundled references (they are large; search, don't read end-to-end).

## First: respect local overrides (this is mandatory)

The project-wide standards below are the *default*, not the last word. Parts of
the tree deliberately diverge, and the closest configuration always wins:

1. **`.clang-format`** — there are many (one per subproject and in some
   subdirectories). Match the *nearest* one. Never reformat code you didn't
   change; restrict formatting to your own diff (use `git clang-format`).
2. **`.clang-tidy`** — the nearest one defines the enforced checks (often
   including the naming convention) for that directory.
3. **Subproject contributing / style docs**, e.g.
   `libcxx/docs/Contributing.rst`, `libc/docs/contributing.rst`,
   `lldb/docs/resources/contributing.rst`,
   `clang-tools-extra/docs/clang-tidy/Contributing.rst`, and flang's own style.
   libc++, libc, lldb, flang, and MLIR all carry conventions that differ from
   the LLVM defaults (naming, headers, allowed std usage, testing).
4. **The surrounding code.** The Golden Rule: if you're editing a file whose
   local style disagrees with these standards, follow the local style for
   consistency within that file — don't mix two styles in one place.

When in doubt, prefer the most local signal: file > directory > subproject >
project-wide. Match what's there.

---

## Coding standards (project-wide defaults)

### Language & libraries
- Write **C++17**; no vendor extensions; only features supported by all host
  toolchains (Clang, GCC, MSVC).
- **No RTTI (`dynamic_cast`), no C++ exceptions.** Use `isa<>/cast<>/dyn_cast<>`
  for type tests (see APIs below).
- **No `<iostream>`** / C++ iostreams. Use `llvm::raw_ostream`.
- Prefer LLVM ADT over std equivalents where one exists (`SmallVector` over
  `std::vector`, `DenseMap`/`StringMap` over `std::map`/`std::unordered_map`).
- C++-style casts only (`static_cast`/`reinterpret_cast`/`const_cast`); C-style
  cast allowed only as `(void)x` to silence unused warnings.
- No static constructors/destructors (no globals with non-trivial ctors/dtors).
- Python: PEP 8; format only changed lines with `darker` (use `black` only when
  the entire patch is a reformat).

### Source formatting
- **80-column** limit. **Spaces, not tabs.** No trailing whitespace. LF endings.
- `#include` order: (1) main module header, (2) local/private headers,
  (3) LLVM/subproject headers, (4) system headers — each group sorted
  lexicographically. In a `.cpp`, its own module header comes **first**.
- Header guard = uppercased include path with `_` for separators, e.g.
  `llvm/include/llvm/Analysis/Utils/Local.h` -> `LLVM_ANALYSIS_UTILS_LOCAL_H`.
- `///` for Doxygen doc comments, `//` for ordinary comments; `/* */` only for
  C-compat or inline `/*ParamName=*/`. Don't comment out code with `/* */` —
  use `#if 0`. Document public APIs in the **header**, not the .cpp; don't
  repeat the entity's name in its doc comment.
- Diagnostic messages: start lowercase, no trailing period. Use
  `Support/WithColor.h`, never write to `stderr` directly.

### Language rules & assertions
- `struct` when all members are public, `class` otherwise; keep the keyword
  consistent across all declarations of a type.
- **Assert liberally, always with a message:** `assert(I < N && "index oob")`.
  Use `[[maybe_unused]]` for variables only read in asserts.
- Use `llvm_unreachable("msg")` (not `assert(false && ...)`) for truly
  unreachable paths. Do **not** add a `default:` to a switch that already covers
  an enum — put `llvm_unreachable` after the switch instead.
- Treat warnings as errors. A header-defined class with a vtable needs at least
  one out-of-line virtual method. Don't write `inline` on in-class methods.
- Default to `llvm::sort` (not `std::sort`); use ordered/`MapVector`/`SetVector`
  containers when iteration order must be deterministic (pointer keys are
  non-deterministic).

### High-level style
- Headers must be **self-contained** (`.h`, own guards, include what they use);
  non-self-contained fragments use `.inc`.
- `#include` as little as possible, especially in headers; forward-declare for
  pointer/reference-only uses. Keep private headers next to the `.cpp`, not in
  the public include tree. Respect library layering — only depend on declared
  deps, no cycles.
- **Early exit** to reduce nesting; **no `else`/`else if` after** a
  `return`/`break`/`continue`/`goto` (exception: `if constexpr`). Factor complex
  loop predicates into named (possibly `static`) helpers.

### Naming
- Types (class/struct/enum/typedef): `UpperCamelCase` noun.
- Variables & public members: `UpperCamelCase` (start uppercase).
- Functions: `lowerCamelCase` verb phrase (`openFile()`, `isFoo()`).
- Discriminator enums get a `Kind` suffix; enumerators start uppercase and (if
  not scoped) carry an enum-name prefix (`VK_Argument` for `ValueKind`).
- STL-mimicking members may use `lower_case` (`push_back`, `begin`); multiple
  iterator ranges get a singular prefix (`global_begin()`).
- Descriptive names; avoid non-obvious abbreviations.

### Low-level / microscopic
- Preincrement (`++X`) when the old value is unused. Use `'\n'`, not
  `std::endl`. Keep anonymous namespaces tiny (class decls only); use `static`
  for file-local functions/data. Don't indent namespace bodies; add
  `} // namespace foo` when the close isn't obvious.
- `auto` only when it aids readability (e.g. `cast<Foo>(...)`, iterator types) —
  **not** "almost always auto". In range-for use `auto &` / `auto *` to avoid
  silent copies. Prefer range-based `for`; cache `end()` in explicit-iterator
  loops unless the container is intentionally mutated (then comment it).
- Space before `(` after control keywords (`if (`), none in calls (`foo(`).
  Omit braces on single simple statements only when every branch in the chain
  also qualifies; always brace `do`-`while`.

---

## Picking the right data structure

Decide the **category** by access pattern, then tune the member for memory /
constant factors / cache. (Full catalog incl. niche containers:
`grep` the "Picking the Right Data Structure" section of
`reference/ProgrammersManual.md`.) Cost order: sequential < set < map.

### Sequential (ordered, allows duplicates, no key lookup)
- **API parameter, read-only:** `ArrayRef<T>` (accepts C array, `std::vector`,
  `SmallVector`, ...).
- **API parameter, mutable:** `SmallVectorImpl<T>&` — never
  `SmallVector<T, N>&` in a signature (locks the caller's N).
- **Local growable:** `SmallVector<T>` (omit `N` unless you have a measured
  reason). `SmallVector<T, 0>` is preferred over `std::vector<T>`; use
  `std::vector` only for >UINT32_MAX elements or interop.
- `TinyPtrVector` for 0/1 non-null pointers; `PackedVector` for few-bit values;
  `ilist` for intrusive lists needing stable pointers + O(1) splice (e.g.
  `Instruction`s). Avoid `std::list`/`std::deque` (high constant factors).
- Hoist a `std::vector`/`SmallVector` out of a loop and `.clear()` it rather
  than reallocating each iteration.

### Strings
- **Read-only param:** `StringRef` (by value). **Never** `const char*`; avoid
  `const std::string&`. A `StringRef` does **not** own its bytes — never store
  one past the buffer's lifetime nor return one for a *computed* string.
- **Concatenation param:** `const Twine&`. Never store a `Twine` or name a
  `Twine` variable — valid only for the duration of the call expression.
- **Owning scratch buffer:** `SmallString<N>`; **persisted owned string:**
  `std::string` (call `.str()` on a `StringRef` to own it). Build with a
  `SmallString` + `raw_ostream` / `Twine`, then persist.

### Set-like (uniquing)
- **Sorted vector + `std::sort`/`unique`** when usage is insert-phase then
  query-phase (best cache locality, fewest allocations; query via
  `std::lower_bound`).
- `SmallPtrSet<T*, N>` for pointers; `SmallSet<T, N>` for small sets of small
  non-pointers; `DenseSet` for small non-pointer values; `StringSet` for
  strings. `SetVector` (or `SmallSetVector`) **only** when you need
  deterministic insertion-order iteration (costs ~2x). Avoid `std::set`
  (malloc-per-element) and `std::unordered_set`.

### Map-like (key -> value)
- **Sorted vector** for strict insert-then-query (as above, comparing keys).
- `DenseMap` for small keys/values (great for pointer->pointer); iterators
  invalidate on insert; needs `DenseMapInfo` for custom keys; `SmallDenseMap`
  for the usually-small case. `StringMap` for string keys. `MapVector` for
  deterministic insertion-order iteration. `IndexedMap` for dense-integer keys.
  `ValueMap` to key on `Value*` across RAUW/deletion. Avoid `std::map`
  (malloc-per-pair) and `std::unordered_map`.
- Need reverse lookup? Keep two maps. Map-like is the most expensive category —
  use only when you need keyed lookup/containment.

### Bit sets (sets of numeric IDs)
- `BitVector` (dense), `SmallBitVector` (<~25 bits common case),
  `SparseBitVector` (sparse, ordered access O(1)), `CoalescingBitVector` (large
  contiguous ranges). **Never `std::vector<bool>`.**

---

## Key APIs

### Type checking — isa / cast / dyn_cast (no RTTI)
- `isa<T>(V)` — boolean test, no extraction. `isa<A, B>(V)` for "any of".
- `cast<T>(V)` — when type is already known; **asserts** on mismatch.
- `dyn_cast<T>(V)` — combined test+cast on a **pointer**; returns null on
  mismatch. Use in an `if`-init; don't precede it with a separate `isa<T>`.
- Possibly-null input: `dyn_cast_if_present<T>` / `cast_if_present<T>` /
  `isa_and_present<T>` (these propagate null instead of asserting). Plain
  `dyn_cast`/`cast`/`isa` require non-null. Prefer `InstVisitor` over long
  `dyn_cast` chains.

### Errors — `llvm::Error` / `Expected<T>`
- An unchecked `Error` **aborts at destruction** — every `Error` must be
  checked, moved, or `consumeError(std::move(E))`d. Return `Error::success()`
  for the ok case.
- Fallible-with-value: `Expected<T>` — test as bool, `*` on success,
  `.takeError()` on failure. Build errors with `createStringError(...)` /
  `make_error<StringError>(...)`; match with `handleErrors` (prefer over
  `handleAllErrors`). `cantFail(...)` only when provably infallible.
  `ExitOnError` in tools/`main` only, never in library code.
- **Don't** signal failure with `bool`/sentinels/out-params. Programmer errors
  (broken invariants) use `assert`/`llvm_unreachable`, not `Error`;
  `report_fatal_error` is a discouraged last resort.
- Fallible "constructor": use a static factory returning `Expected<Foo>`.

### Callables
- `function_ref<Ret(Args)>` (by value) for a callback **not stored** past the
  call — never store it (points at the caller's storage). Use `std::function`
  (or a template param) only when the callback must outlive the call.

### Debug & stats
- `LDBG() << "..."` (`llvm/Support/DebugLog.h`) for one-line debug;
  `LLVM_DEBUG({ ... })` (`llvm/Support/Debug.h`) for guarded blocks. **No
  side-effects** inside — they compile out in non-assert builds.
- Define `#define DEBUG_TYPE "name"` after all includes, before any
  `LDBG`/`STATISTIC` (no commas in the name). Gate output with
  `-debug` / `-debug-only=name`.
- `STATISTIC(Name, "desc")` + `++Name` (`llvm/ADT/Statistic.h`); printed by
  `-stats` (assertions build). Define `DEBUG_TYPE` before it.

### String formatting
- Prefer `formatv("{0}...", args...)` over `printf`/`sprintf` (type-deduced,
  compile-time checked, supports alignment/ranges).

---

## When to dig into the references
`grep reference/CodingStandards.md` or `reference/ProgrammersManual.md` for:
exact include/formatting edge cases, the full container catalog (FoldingSet,
IntervalMap, SparseSet, ImmutableSet/Map, RadixTree, ...), the complete
`Error` recipe set (joinErrors, ExitOnError details, ErrorAsOutParameter),
threading/`ManagedStatic`/`LLVMContext`, and the Core class hierarchy
(`Value`/`User`/`Instruction`/`Type`/...). Search for a specific symbol or
section heading rather than reading the whole file.
