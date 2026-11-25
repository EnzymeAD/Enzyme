---
name: llvm-comment-hygiene
description: >
  How to write (and cut) code comments for LLVM-based C/C++ codebases -- LLVM,
  Clang, MLIR, and out-of-tree projects built on the LLVM libraries. The
  through-line: two kinds of comment, two jobs -- doc comments on
  functions/types state WHAT/HOW succinctly (a contract a caller reads instead
  of the body), while inline comments explain only WHY a non-obvious passage is
  written that way. Neither describes what OTHER code does (it bitrots), the
  history of the code, or invented jargon. Use when writing or reviewing
  comments in a PR/diff, and ALWAYS self-check comments on AI-/agent-generated
  changes before sending them for review -- LLMs systematically over-comment:
  walls of text, restated algorithms, enumerations that duplicate the code,
  references to future/other/past code, and coined terminology. Complements the
  broader `llvm-style-review` and `llvm-coding-standards` skills (which also
  cover non-comment issues: asserts vs errors, LLVM idioms, data structures,
  tests). Trigger phrases: "improve these comments", "are these comments good",
  "trim the comments", "would these comments pass review", "self-check my
  comments before review", "too verbose".
---

# LLVM comment hygiene

Rules for writing and cutting code comments in LLVM-based C++ (LLVM, Clang,
MLIR, and projects built on the LLVM libraries). For the wider review checklist
(asserts, error flow, LLVM idioms, tests) see the `llvm-style-review` skill; for
container/style rules see `llvm-coding-standards`.

## The one rule: two kinds of comment, two jobs

There are two kinds of comment and they do opposite things. Decide which you are
writing before you write it.

- **Top-level / doc comments** (on a function, struct, class, enum, or file) are
  user-facing documentation. They **should** state **what** the thing is and,
  briefly, **how** it behaves -- so a caller understands it *without reading the
  body*. Keep it succinct -- the contract (inputs, result, notable edge cases),
  not a line-by-line tour of the implementation. Succinct means as short as it
  can be while staying clear, not a fixed line budget: give a genuinely subtle
  type or contract the space it needs, and don't pad a trivial one.

- **Inline comments** (inside a body) complement code whose *what* is already
  clear from reading it. They exist **only** to explain **why** a non-obvious
  passage is the way it is. Use them **sparingly**. If an inline comment restates
  what the next line does, cut it.

So before writing or keeping a comment, ask:

1. Is it a **doc** comment? Then does it succinctly say *what/how* for a caller
   who won't read the body -- without drifting into implementation narration?
2. Is it an **inline** comment? Then does it say *why*, about the code right
   below it, in its current form? If it only says *what*, cut it.

Everything below refines these two jobs. The single most common failure on
agent-written code is inline comments that narrate the algorithm, and doc
comments that balloon into implementation essays.

## Principles (each with a real before -> after)

### 1. Inline: explain "why", not "what"/"how"
Inside a body, the code is the source of truth for what and how. An inline
comment that narrates the algorithm duplicates the code and drifts out of sync.
(Doc comments are different -- see sec 9 and the two-kinds rule above.)

- Bad -- narrates the walk step by step across ~30 lines ("first we collect all
  the incoming values, then for each predecessor we look up the mapping, then we
  build a new PHI with the remapped operands and finally we replace all uses...").
- Good:
  ```cpp
  // RAUW would invalidate the iterator over Old's uses, so collect the users
  // into a worklist first and rewrite them after the walk.
  ```
  One "why" (the invalidation hazard), no replay of the loop.

### 2. Never describe code that lives elsewhere
The moment you explain what another function/pass/library does, you've created a
comment that lies as soon as that code changes. Describe only the code in front
of you.

- Bad: `// ... mirrors the numbering scheme used by GVN and the value-tracking
  prepasses.`  /  a CAVEAT paragraph detailing what `SelectionDAGBuilder`
  materialises for a given node.
- Good: `// Compute the CFG successors of BB whose last instruction is
  LastInst.` -- what *this* function does, nothing about its future callers.
- Allowed: a *very brief* example is fine (`// e.g. add i32 %a, %b -> add nsw i32
  %a, %b`), but not a tour of the other code's internals.

**A subclass override is "code elsewhere" too.** A doc comment on a base
class, an abstract/virtual method, or a default method body must state only
*this* declaration's polymorphic contract -- what a caller may assume across
all implementations. It must NOT narrate what each concrete subclass does,
enumerate which override returns what, or explain how a specific subclass
implements it: that behaviour lives in the subclass and rots the moment a
subclass is added, removed, or changed. Say what "override" *means* here, not
who overrides it or how. (Symmetrically, an override's own doc states its own
contract, not a comparison against its siblings' or parent's implementation.)
This is the single most common way base-class docs balloon -- and it is easy to
rationalise as "helpful context," so hold the line.

- Bad (base `getCost` narrating two subclasses):
  ```cpp
  // Return the modelled cost of the node. The generic implementation returns a
  // flat cost of 1; the target-specific override consults the subtarget
  // scheduling model and adds latency, while the size-tuned override instead
  // returns encoded byte length so the pass optimises for code size ...
  ```
- Good -- the contract, no subclass tour:
  ```cpp
  // Return the modelled cost of the node, in units defined by the active cost
  // model. Lower is cheaper; callers compare costs, never assume a unit.
  ```
- Good (a virtual whose subclasses vary): name the hook, not the subclasses --
  `// Cost unit chosen by the cost model; the base returns an abstract count. A
  subclass may return latency or encoded size instead.`

### 3. Never reference non-local, future, or uncommitted code
No "Phase 3", no "lands with the coroutine-splitting patch", no naming a pass
that isn't in this diff. If the reader can't see it here, don't mention it.

- Bad: `// BlockStarts uses std::set because a later pass iterates it in
  ascending order to assign deterministic block labels...`
- Good: `// BlockStarts is ordered so it can be iterated in offset order and
  queried with upper_bound.` -- states the local requirement (ordering), not who
  consumes it later.

### 4. Never reference other/historical states of the code
Committed comments describe the *current* state only. No "as of this commit",
"previously", "the earlier implementation", "now returns", commit SHAs, or PR
numbers. Git carries history; comments must not.

- Bad: a paragraph recounting how a key got inserted twice, "see commit
  eaee0a0e88 for the repair", "the earlier implementation used a vector...".
- Good:
  ```cpp
  // A duplicate key would silently keep only the first entry and route the rest
  // to the wrong handler, so a duplicate key is a table-authoring bug.
  ```
  States why the check exists, in the present tense, with no history.

### 5. Don't state common sense
Don't justify using LLVM's own facilities inside LLVM code, or explain why you'd
avoid an obviously-worse alternative.

- Bad: `// Using the enum constants instead of raw strings gives us compile-time
  checking...` and `// We include the .def header here rather than duplicating
  the list by hand...`
- Good: delete them. It is common sense not to key on strings; nobody needs the
  header's transitive-include story.

### 6. Don't invent terminology -- reuse what the code around you already uses
Don't coin a name for a concept ("the return-value collapse", "pseudo-alias
step", "parallel-pseudo") and then build comments around your coinage. Describe
the thing plainly, and prefer the name the surrounding code already gives it.

- Bad: `// Return-value collapse: source must be an atomic with a return
  value...`  /  `// Re-apply the pseudo-alias step.`
- Good: `// Source is an atomicrmw whose result is used; the folded form drops
  the result.`  /  `// Re-run the canonicalization: the earlier fold can expose
  a new foldable pattern...`

**Identifiers and names count too.** A coined term baked into a variable,
field, function, type, or enumerator name is the same defect as one in a
comment -- names carry no semantics, so they answer to this rule exactly like
prose. If a term is wrong for a comment, it is wrong for the identifier that
comment sits on; rename it rather than documenting around it. Fix both in the
same pass so they stay consistent.

- Bad: a local `const bool CrossFoldLoadStore = ...` (and comments referring to
  "cross-folding") where "cross-folding" is an invented term for a plain
  load/store forwarding.
- Good: rename to `ForwardStoredValue` and use plain "forwarding" in the prose --
  a normal word, not a coinage. (Leave genuinely established local vocabulary
  alone -- e.g. "materialize", used pervasively in the surrounding code, is not
  a fresh coinage.)

Before naming a concept, look for an existing term for it in the code nearby and
reuse that -- consistency across the codebase beats a fresh coinage, even a
clever one. Search outward in order of proximity, because the closer the code the
better the term fits the exact concept:

1. the same file (a sibling function, the type's own name, a field or enum);
2. the same directory / the headers this file includes;
3. types and functions on the call graph (callers and callees) and the LLVM
   headers this code is built on (IR, MC, TableGen, `TargetInstrInfo`, ...).

If LLVM or the local code already calls it a "pseudo", an "encoding", a
"predicate", a "marker", use that word; don't rebrand it.

Avoid abbreviations and acronyms unless they are already established in this
codebase or LLVM. Spell the concept out rather than minting a short form for it
(`SrcMod` in a comment where nothing else uses that abbreviation is worse than
"source modifier"). Established ones are fine to reuse -- MC, IR, SDNode,
TSFlags, RAUW, CSE -- precisely because they are already the local vocabulary.

### 7. Don't enumerate what the code already lists
A comment that re-lists an array's entries, an enum's values, or a switch's cases
is pure duplication and rots on the next edit. The array *is* the list.

- Bad: a comment spelling out `dropped: undef, poison` and
  `kept: argument, constant, global, instruction, ...` directly above the
  `KKnownOperandKinds[]` array holding exactly those names.
- Good: say why the array exists, let the array be the list:
  ```cpp
  // An operand kind is either a placeholder we drop or a real value we keep; an
  // unrecognised one means a new Value subclass this code does not account for.
  ```

### 8. No dangling or misplaced comments
Every comment attaches to the code immediately below it. When you edit or move
code, delete comments it orphaned. A block describing X must not sit above
unrelated code Y.

- Bad: a full explanation of a masking rule left sitting above an unrelated
  constant (it describes a rule defined much later); a header comment `// Compute
  the absolute byte offset of a branch target...` with no such function under it.
- Good: delete the orphan; move the rationale next to the code it explains (the
  masking rule already documents itself at its own site).

### 9. Document types and functions -- succinctly (state what/how)
A struct/class/function earns a doc comment saying **what** it is and, when not
obvious, **how** it behaves -- so a user need not read the body. This is the one
place a comment *should* state what/how. Keep it succinct -- but succinct means
clear-and-no-longer, not a line cap: a subtle type or contract may need several
lines, a trivial one needs a phrase. What a doc comment must *not* do is grow
into a rationale essay or an implementation walkthrough.

- Bad: a 10-line block over `struct Entry` explaining the whole lookup-table
  build chain and the enum-vs-string tradeoff (that is implementation, not the
  contract).
- Good:
  ```cpp
  // One row of the opcode table: an instruction opcode and the handler kind the
  // pass dispatches on.
  struct Entry { unsigned Opcode; HandlerKind Kind; };
  ```
- Good (function contract -- what + result + edge case, no body tour):
  ```cpp
  // Decode the bytes in Text at [Begin, End) using the caller-owned Subtarget,
  // producing one MCInst per instruction. End of 0 scans to the end of Text.
  ```

For a base class or a virtual/abstract method, the doc states the *polymorphic
contract* -- what every implementation must satisfy and what a caller may assume
-- and nothing about which subclass does what or how (that is code elsewhere;
see sec 2). For a predicate hook, say what returning true vs false *means*, not
which subclass returns which. Do not treat this as an invitation to catalogue
the overrides "for context": a base doc that names its subclasses is a defect,
not thoroughness.

### 10. Shorten walls of text
A doc comment that has grown into a paragraph is usually narrating the
implementation instead of stating the contract; an inline comment that has grown
into a paragraph is usually restating what/how. Either way, cut to the
load-bearing content -- the contract for a doc comment, the "why" for an inline.
Multi-sentence rule comments in a table are a smell -- one clause each.

- Bad: 10 lines on a helper (which flag it reads, which subtarget predicate
  gates it, the pre-legalize vs post-legalize paths, where the operand comes
  from...).
- Good: `// Post-legalize variant: adds a byte-select operand but keeps the same
  opcode identity.`

### 11. Don't annotate includes
No trailing `// what this header provides` on `#include` lines, and no block
listing which symbols each header exposes. At most, a one-line label separating
one group of headers from another, with a blank line between the groups.

- Bad: `#include "llvm/IR/Instructions.h" // LoadInst, StoreInst, PHINode, ...`
  and a 12-line block cataloguing which declaration comes from which header.
- Good:
  ```cpp
  // Project-private headers.
  #include "MyPass.h"
  #include "MyPassDetail.h"

  #include "llvm/ADT/StringMap.h"
  ```

### 12. ASCII only, present tense, LLVM voice
No em-dashes or other non-ASCII in comments (use `--` or rephrase); grep
`[^\x00-\x7F]` before sending. Write in the present tense about current behavior.
Match the surrounding comment density and terseness -- do not raise it.

## Fast self-review checklist

Run this over every comment in a diff (especially agent-written ones). First
decide which kind it is:

0. **Doc or inline?** A doc comment (on a function/type/file) should state
   *what/how* for a caller who won't read the body -- judge it by whether it is a
   succinct contract, not by the "why" test (sec 9). An inline comment is judged
   by the rest of this list.

1. Inline only: does it explain **why**, not what/how? (else cut -- sec 1, sec 10)
2. Does it describe code **outside** this diff -- another pass, a library, a
   future phase? (cut -- sec 2, sec 3)
   - Base/virtual-method special case: does a base-class or abstract/virtual
     doc **narrate its subclasses** (name an override, say which one returns
     what/how)? A subclass override is code elsewhere -- state only the
     polymorphic contract. (cut -- sec 2, sec 9)
3. Does it mention **history** -- "previously", "as of this commit", a SHA/PR#?
   (cut -- sec 4)
4. Is it **common sense** or an include annotation? (cut -- sec 5, sec 11)
5. Does it **coin a term** or a stray abbreviation instead of reusing the word
   the nearby code/LLVM already uses? (reuse the local term -- sec 6)
6. Does it **re-list** what the code below already lists? (cut to the why -- sec 7)
7. Is it **attached** to the code right below it? (fix or delete orphans -- sec 8)
8. Is a type/function doc a **succinct contract** (clear, no implementation
   tour) -- as long as clarity needs, no longer? (sec 9)
9. **ASCII-only**, present tense, matches local density? (sec 12)

If a comment survives its kind's checks, keep it. Most inline ones will not; doc
comments usually just need trimming to the contract.
