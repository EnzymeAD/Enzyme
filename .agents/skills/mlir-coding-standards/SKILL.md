---
name: mlir-coding-standards
description: >
  MLIR subproject coding standards. These complement and refine LLVM project
  coding standards. **IMPORTANT:** apply these WHENEVER editing or reviewing
  MLIR files in Enzyme (`enzyme/Enzyme/MLIR/`, `enzyme/test/MLIR/`) or within
  llvm-project/mlir or any project that uses MLIR and doesn't instruct you
  otherwise. Prefer file- and project-local style to this for consistency.
  Trigger phrases: "how to do something in MLIR", "follow MLIR style",
  "idiomatic in MLIR", "right way to pass an object to an MLIR function".
---

# MLIR coding standards and relation to LLVM

Apply this to **every change to MLIR source and tests in Enzyme
(`enzyme/Enzyme/MLIR/` and `enzyme/test/MLIR/`)** as well as inside the `mlir/`
subdirectory of an `llvm-project` checkout. **IMPORTANT:** these
rules override and take precedence over `llvm-coding-standards`, but the latter
has many cases that still apply and are not specified here. Automated checks for
these standards live in `enzyme/scripts/check_mlir_coding_standards.py`. The
authoritative documents may be found under `reference/` in this directory or
under `mlir/docs` in an `llvm-project` checkout:

- `reference/DeveloperGuide.md` - the MLIR developer guide, core document
  largely referring to the LLVM coding standards available through
  `llvm-coding-standards`.
- `reference/TestingGuide.md` - supplementary information for tests
  specifically.
- `reference/UsageOfConst.md` - rationale and rules for why MLIR omits `const`
  on core IR types.
- `reference/VariableNames.md` - background on `camelBack` variable naming.
- `mlir/docs/LangRef.md` - MLIR "language reference", an extensive document
  detailing core MLIR concepts with examples.
- `mlir/docs/DefiningDialects/_index.md` (and sibling files `Operations.md`,
  `AttributesAndTypes.md`, `Constraints.md`, `Assembly.md` in
  `mlir/docs/DefiningDialects/`) - the entry point for defining or modifying any
  component of an MLIR dialect: operations, attributes, types.
- `mlir/docs/Interfaces.md` - MLIR interfaces documentation, explains how to
  introduce and use interfaces for dialect components.
- `mlir/docs/CAPI.md` - documentation on defining and using C API for MLIR
  components.
- `mlir/docs/DialectConversion.md` - details the dialect conversion (rewrite
  patterns with type change) subsystem.
- `mlir/docs/PatternRewriter.md` - details the rest of the pattern rewriting
  infrastructure.
- `mlir/docs/SymbolsAndSymbolTables.md` - explains how to work with Symbol
  operations efficiently.
- `mlir/docs/PassManagement.md` - details passes, pass managers, and
  instance-specific pass options.
- `mlir/docs/Diagnostics.md` - details diagnostic emission and verification.

When you see a reference to another markdown document in one of the above,
there are two possibilities:
- The document is readily available under `mlir/docs`, just read it.
- The document is generated on-the-fly from dialect definitions, it won't be
  present in `mlir/docs` but you will find a way to generate it by invoking
  `mlir-tblgen` (which may need to be built) with `-gen-*-doc` flags; if there
  is a build tree available, the `mlir-doc` cmake target will generate all the
  documentation there.

## Key APIs

### IR constructs - specific subclasses of Type, Attribute, plus `*Op` wrappers
 - `Value`, `Type`, `Attribute` and derived classes are wrapping a pointer and
   must be passed around by value.
 - `*Op` classes are wrapping an `Operation *` and must be passed by value.

### IR traversal
 - Use `Operation::walk`, `Type::walk`, `Attribute::walk` to navigate IR
   instead of hand-rolling functions.
 - Avoid recursion on use-def chains and blocks, recursion on types and
   `Region`s is tolerated but not ideal.

### IR mutation
 - Prefer named constructors of the form `dialect::NameOp::create`.
 - Avoid direct mutation of IR objects as much as possible, use `RewriterBase`
   instead.
 - Make auxiliary functions accept a `RewriterBase &` or an `OpBuilder &`
   (or `ImplicitLocOpBuilder &`) and use those rather than create local
   instances.
 - Consider the cost of driver infrastructure, a simple `walk` is cheaper than
   greedy rewriter, which itself is cheaper than dialect conversion.

### Interfaces and Traits
 - Prefer reasoning about interfaces and traits to listing specific operations,
   attributes or types.
 - Introducing a new interface requires evidence that it will be reused for
   multiple entities.

### Verifiers and Passes
 - Op verifiers must only verify **local** properties of an operation (never
   follow def-use chains to inspect operand producers or result users).
 - Passes assume their input IR is verifier-valid and must produce
   verifier-valid IR. It must not be possible to trigger a crash or assertion by
   running a pass on verifier-valid IR.
 - Avoid `cl::opt` for pass options; use TableGen / instance-specific pass
   options instead, and prefix dialect-specific pass and option names with the
   dialect name.

### Errors - `LogicalResult` / `FailureOr` / nullptr
 - Use `LogicalResult` instead of boolean returns.
 - Use `FailureOr` to combine `LogicalResult` with a value to be returned on
   success.
 - Use `nullptr` to signify error for IR constructs listed above, they are all
   nullable.
 - Make the code emit a diagnostic at a location that has the most context,
   typically the innermost call with the rest of the stack propagating the
   error state.
 - Walkers can use `WalkResult::interrupt` to signify error.

## Testing

 - Do add tests for non-trivial functionality.
 - Strongly prefer LIT tests, even if additional operations are added to test
   dialects.
 - Never hardcode/pin literal SSA value names in FileCheck tests; match them
   with wildcards or capture them into FileCheck variables (`%[[VAL:.*]]`) when
   checking def-use relationships.
 - Never add test-only operations to main dialects, they belong into a test
   dialect that only builds when testing is requested.
 - Only resort to unit tests when the behavior cannot be tested transitively
   via LIT tests. We generally test at the pass boundary, no need to probe
   IR-inspecting or mutating functions if they are already exercised through
   passes.

## Style
 - Use `camelBack` for variable and parameter names (overriding LLVM's
   `UpperCamelCase`), while keeping `UpperCamelCase` for type names.
 - Use Doxygen-style (`///`) comments for top-level and class member
   definitions, regardless of whether they are public APIs or inside `.cpp`
   files.
 - Never use `const` on core IR types (`Operation *`, `Block *`, `Region *`,
   `Value`, `*Op`), even in function parameters or member methods. For non-IR
   types, avoid `const` on local variables outside globals and function
   arguments; pass non-nullable output arguments (other than IR units) by
   non-const reference.
 - Avoid explicit `inline` unless necessary to avoid duplicate symbols, let the
   compiler inline what it deems necessary.
 - Expand `auto` into concrete types unless the right-hand side is a lambda, a
   constructor, a cast (including LLVM custom cast), or the type is an
   iterator-like nested construct that is too long to spell out.

---

## When to read the reference and `mlir/docs` files

The files in `reference/` and `mlir/docs/` are compact enough (typically 200 to
1,500 lines each) that you can and should load the relevant file in full when
working on that area of MLIR (or search across them with `grep` when looking up
a specific keyword such as an operation, attribute, type, rewriter, dialect
conversion, or walker).
