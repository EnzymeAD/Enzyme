---
title: "Developer Guide"
date: 2019-11-29T15:26:15Z
draft: false
weight: 20
---

This document attempts to describe a few developer policies used in MLIR (such
as coding standards used) as well as development approach (such as, testing
methods).

## Style guide

MLIR follows the [LLVM style](https://llvm.org/docs/CodingStandards.html) guide.
We also adhere to the following (which deviate from or are not specified in the
LLVM style guide):

*   Adopts [camelBack](https://llvm.org/docs/Proposals/VariableNames.html);
*   Uses Doxygen-style (`///`) comments for top-level and class member
    definitions, regardless of them being visible as public APIs.
*   Except for IR units (Region, Block, and Operation), non-nullable output
    arguments are passed by non-const reference in general.
*   IR constructs are not designed for [const correctness](../../docs/Rationale/UsageOfConst.md).
*   Do *not* use recursive algorithms if the recursion can't be bounded
    statically: that is avoid recursion if there is a possible IR input that can
    trigger a stack overflow (for example traversing use-def chains in a
    recursive way). At the moment, we tolerate it for the two following cases:
    *   The nesting of the IR: we use recursion when traversing nested regions.
    *   Type nesting: recursion may be used for the nesting of composite types.
*   Follows git conventions for
    [commit messages](Contributing.md#commit-messages).

Please run clang-format on the files you modified with the `.clang-format`
configuration file available in the root directory. Check the clang-format
[documentation](https://clang.llvm.org/docs/ClangFormat.html) for more details
on integrating it with your development environment. In particular, if clang is
installed system-wide, running `git clang-format origin/master` will update the
files in the working directory with the relevant formatting changes; don't
forget to include those to the commit.

## IR should be valid before and after each pass

Passes should assume that their input IR passes the verifier. Passes should not
check invariants that are guaranteed by the verifier. If many passes are checking
the same invariant consider adding that invariant to the verifier or factoring
the IR design / dialects to better model the invariant at each phase of compilation.

Similarly, the IR after a pass runs should be verifier-valid. If a pass produces IR
that fails the verifier then the pass has a bug.

It is somewhat common for invalid IR to exist transiently while a pass is executing.
This is moderately discouraged but often practically necessary.

## Assertions and crashes in passes

It should not be possible to trigger a crash or assertion by running an MLIR
pass on verifier-valid IR. If it is possible, then the pass has a bug.

If a pass requires additional invariants not guaranteed by the verifier, then
it should check them itself and if those invariants are not present,
either safely perform no transformation (for pure optimization passes)
or emit a diagnostic explaining why a transformation cannot be performed
(for lowering passes where the transformation is needed for correctness).

## Pass name and other command line options

To avoid collision between options provided by different dialects, the naming
convention is to prepend the dialect name to every dialect-specific passes and
options in general. Options that are specific to a pass should also be prefixed
with the pass name. For example, the affine dialect provides a loop tiling pass
that is registered on the command line as `-affine-tile`, and with a tile size
option that can be set with `-affine-tile-size`.

We also avoid `cl::opt` to provide pass options in favor of the
[pass options](../docs/PassManagement.md#instance-specific-pass-options)
mechanism. This allows for these options to be serialized in a pass pipeline
description, as well as passing different options to multiple instances of a
pass in the same pipeline.

## Testing guidelines

See here for the [testing guide](TestingGuide.md).

## Guidelines on contributing a new dialect (or important components)

To contribute a dialect (or a major component in MLIR), it is usual to write an
overview "RFC" (it can be just a few informal paragraphs) and send it to the
MLIR mailing-list. When accepting a new component to MLIR, the community is
also accepting the burden of maintaining it. The following points should be
considered when evaluating whether a dialect is a good fit for the core MLIR
repository:
 * What is the overall goal of the dialect? What is the first implementation
   milestone?
 * How does it fit into the MLIR dialect ecosystem?
   * Connection: how does it connect to the existing dialects in a compilation
     pipeline(s)?
   * Consolidation: is there already a dialect with a similar goal or matching
     abstractions; if so, can it be improved instead of adding a new one?
   * Reuse: how does it generalize to similar but slightly different use-cases?
 * What is the community of users that it is serving?
 * Who are the future contributors/maintainers beyond those who propose the
   dialect?

On a practical aspect, we will expect the code to follow the other sections of
this document, with an emphasis on the documentation alongside the source code.

It is prefered to upstream your dialects/components in small incremental
patches that can be individually reviewed. That is, after the initial RFC has
been agreed on, we encourage dialects to be built progressively by faster
iterations in-tree ; as long as it is clear they evolve towards their
milestones and goals.

We have seen the following broad categories of dialects:
  * Edge dialects that model a representation external to MLIR. Examples include
    LLVM, SPIR-V dialects, TensorFlow, XLA/HLO, … Such dialects may be a better
    fit for the project that contains the original representation instead of
    being added to the MLIR repository. In particular, because MLIR will not
    take an external dependency on another project.
  * Structured Abstraction dialects that generalize common features of several
    other dialects or introduce a programming model. Generalization is sometimes
    demonstrated by having several dialects lower to or originate from a new
    dialect. While additional abstractions may be useful, they should be traded
    off against the additional complexity of the dialect ecosystem. Examples of
    abstraction dialects include the GPU and Loop dialects.
  * Transformation dialects that serve as input/output for program
    transformations. These dialects are commonly introduced to materialize
    transformation pre- and post-conditions in the IR, while conditions can be
    obtained through analysis or through operation semantics. Examples include
    Affine, Linalg dialects.

While it can be useful to frame the goals of a proposal, this categorization is
not exhaustive or absolute, and the community is open to discussing any new
dialect beyond this taxonomy.
