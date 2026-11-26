# MLIR Testing Guide

## Quickstart commands

These commands are explained below in more detail. All commands are run from the
cmake build directory `build/`, after [building the project](/getting_started/).

### Run all MLIR tests:

```sh
cmake --build . --target check-mlir
```

### Run integration tests (requires `-DMLIR_INCLUDE_INTEGRATION_TESTS=ON`):

```sh
cmake --build . --target check-mlir-integration
```

### Run C++ unit tests:

```sh
bin/llvm-lit -v tools/mlir/test/Unit
```

### Run `lit` tests in a specific directory

```sh
bin/llvm-lit -v tools/mlir/test/Dialect/Arith
```

### Run a specific `lit` test file

```sh
bin/llvm-lit -v tools/mlir/test/Dialect/Polynomial/ops.mlir
```

## Test categories

### `lit` and `FileCheck` tests

[`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html) is a tool that
"reads two files (one from standard input, and one specified on the command
line) and uses one to verify the other." One file contains a set of `CHECK` tags
that specify strings and patterns expected to appear in the other file. MLIR
utilizes [`lit`](https://llvm.org/docs/CommandGuide/lit.html) to orchestrate the
execution of tools like `mlir-opt` to produce an output, and `FileCheck` to
verify different aspects of the IR—such as the output of a transformation pass.

The source files of `lit`/`FileCheck` tests are organized within the `mlir`
source tree under `mlir/test/`. Within this directory, tests are organized
roughly mirroring `mlir/include/mlir/`, including subdirectories for `Dialect/`,
`Transforms/`, `Conversion/`, etc.

#### Example

An example `FileCheck` test is shown below:

```mlir
// RUN: mlir-opt %s -cse | FileCheck %s

// CHECK-LABEL: func.func @simple_constant
func.func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %[[RESULT:.*]] = arith.constant 1
  // CHECK-NEXT: return %[[RESULT]], %[[RESULT]]

  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  return %0, %1 : i32, i32
}
```

A comment with `RUN` represents a `lit` directive specifying a command line
invocation to run, with special substitutions like `%s` for the current file. A
comment with `CHECK` represents a `FileCheck` directive to assert a string or
pattern appears in the output.

The above test asserts that, after running Common Subexpression Elimination
(`-cse`), only one constant remains in the IR, and the sole SSA value is
returned twice from the function.

#### Build system details

The main way to run all the tests mentioned above in a single invocation can be
done using the `check-mlir` target:

```sh
cmake --build . --target check-mlir
```

Invoking the `check-mlir` target is roughly equivalent to running (from the
build directory, after building):

```shell
./bin/llvm-lit tools/mlir/test
```

See the [Lit Documentation](https://llvm.org/docs/CommandGuide/lit.html) for a
description of all options.

Subsets of the testing tree can be invoked by passing a more specific path
instead of `tools/mlir/test` above. Example:

```shell
./bin/llvm-lit tools/mlir/test/Dialect/Arith

# Note that it is possible to test at the file granularity, but since these
# files do not actually exist in the build directory, you need to know the
# name.
./bin/llvm-lit tools/mlir/test/Dialect/Arith/ops.mlir
```

Or for running all the C++ unit-tests:

```shell
./bin/llvm-lit tools/mlir/test/Unit
```

The C++ unit-tests can also be executed as individual binaries, which is
convenient when iterating on cycles of rebuild-test:

```shell
# Rebuild the minimum amount of libraries needed for the C++ MLIRIRTests
cmake --build . --target tools/mlir/unittests/IR/MLIRIRTests

# Invoke the MLIRIRTest C++ Unit Test directly
tools/mlir/unittests/IR/MLIRIRTests

# It works for specific C++ unit-tests as well:
LIT_OPTS="--filter=MLIRIRTests -a" cmake --build . --target check-mlir

# Run just one specific subset inside the MLIRIRTests:
tools/mlir/unittests/IR/MLIRIRTests --gtest_filter=OpPropertiesTest.Properties
```

Lit has a number of options that control test execution. Here are some of the
most useful for development purposes:

*   [`--filter=REGEXP`](https://llvm.org/docs/CommandGuide/lit.html#cmdoption-lit-filter) :
    Only runs tests whose name matches the REGEXP. Can also be specified via the
    `LIT_FILTER` environment variable.
*   [`--filter-out=REGEXP`](https://llvm.org/docs/CommandGuide/lit.html#cmdoption-lit-filter-out) :
    Filters out tests whose name matches the REGEXP. Can also be specified via
    the `LIT_FILTER_OUT` environment variable.
*   [`-a`](https://llvm.org/docs/CommandGuide/lit.html#cmdoption-lit-a) : Shows
    all information (useful while iterating on a small set of tests).
*   [`--time-tests`](https://llvm.org/docs/CommandGuide/lit.html#cmdoption-lit-time-tests) :
    Prints timing statistics about slow tests and overall histograms.

Any Lit options can be set in the `LIT_OPTS` environment variable. This is
especially useful when using the build system target `check-mlir`.

Examples:

```
# Only run tests that have "python" in the name and print all invocations.
LIT_OPTS="--filter=python -a" cmake --build . --target check-mlir

# Only run the array_attributes python test, using the LIT_FILTER mechanism.
LIT_FILTER="python/ir/array_attributes" cmake --build . --target check-mlir

# Run everything except for example and integration tests (which are both
# somewhat slow).
LIT_FILTER_OUT="Examples|Integrations" cmake --build . --target check-mlir
```

Note that the above use the generic cmake command for invoking the `check-mlir`
target, but you can typically use the generator directly to be more concise
(i.e. if configured for `ninja`, then `ninja check-mlir` can replace the `cmake
--build . --target check-mlir` command). We use generic `cmake` commands in
documentation for consistency, but being concise is often better for interactive
workflows.

### Diagnostic tests

MLIR provides rich source location tracking that can be used to emit errors,
warnings, etc. from anywhere throughout the codebase, which are jointly called
*diagnostics*. Diagnostic tests assert that specific diagnostic messages are
emitted for a given input program. These tests are useful in that they allow
checking specific invariants of the IR without transforming or changing
anything.

Some examples of tests in this category are:

-   Verifying invariants of operations
-   Checking the expected results of an analysis
-   Detecting malformed IR

Diagnostic verification tests are written utilizing the
[source manager verifier handler](mlir/docs/Diagnostics.md#sourcemgr-diagnostic-verifier-handler),
which is enabled via the `verify-diagnostics` flag in `mlir-opt`.

An example .mlir test running under `mlir-opt` is shown below:

```mlir
// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Expect an error on the same line.
func.func @bad_branch() {
  cf.br ^missing  // expected-error {{reference to an undefined block}}
}

// -----

// Expect an error on an adjacent line.
func.func @foo(%a : f32) {
  // expected-error@+1 {{invalid predicate attribute specification: "foo"}}
  %result = arith.cmpf "foo", %a, %a : f32
  return
}
```

### Integration tests

Integration tests are `FileCheck` tests that verify functional correctness of
MLIR code by running it, usually by means of JIT compilation using
`mlir-cpu-runner` and runtime support libraries.

Integration tests don't run by default. To enable them, set the
`-DMLIR_INCLUDE_INTEGRATION_TESTS=ON` flag during `cmake` configuration as
described in [Getting Started](_index.md).

```sh
cmake -G Ninja ../llvm \
   ... \
   -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
   ...
```

Now the integration tests run as part of regular testing.

```sh
cmake --build . --target check-mlir
```

To run only the integration tests, run the `check-mlir-integration` target.

```sh
cmake --build . --target check-mlir-integration
```

Note that integration tests are relatively expensive to run (primarily due to
JIT compilation), and tend to be trickier to debug (with multiple compilation
steps _integrated_, it usually takes a bit of triaging to find the root cause
of a failure). We reserve e2e tests for cases that are hard to verify
otherwise, e.g. when composing and testing complex compilation pipelines. In
those cases, verifying run-time output tends to be easier then the checking
e.g. LLVM IR with FileCheck. Lowering optimized `linalg.matmul` (with tiling
and vectorization) is a good example. For less involved lowering pipelines or
when there's almost 1-1 mapping between an Op and it's LLVM IR counterpart
(e.g. `arith.cmpi` and LLVM IR `icmp` instruction),  regular unit tests are considered
enough.

The source files of the integration tests are organized within the `mlir` source
tree by dialect (for example, `test/Integration/Dialect/Vector`).

#### Hardware emulators

The integration tests include some tests for targets that are not widely
available yet, such as specific AVX512 features (like `vp2intersect`) and the
Intel AMX instructions. These tests require an emulator to run correctly
(lacking real hardware, of course). To enable these specific tests, first
download and install the
[Intel Emulator](https://software.intel.com/content/www/us/en/develop/articles/intel-software-development-emulator.html).
Then, include the following additional configuration flags in the initial set up
(X86Vector and AMX can be individually enabled or disabled), where `<path to
emulator>` denotes the path to the installed emulator binary. `sh cmake -G Ninja
../llvm \ ... \ -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
-DMLIR_RUN_X86VECTOR_TESTS=ON \ -DMLIR_RUN_AMX_TESTS=ON \
-DINTEL_SDE_EXECUTABLE=<path to emulator> \ ...` After this one-time set up, the
tests run as shown earlier, but will now include the indicated emulated tests as
well.

### C++ Unit tests

Unit tests are written using the
[googletest](https://google.github.io/googletest/) framework and are located in
the `mlir/unittests/` directory.

## Contributor guidelines

In general, all commits to the MLIR repository should include an accompanying
test of some form. Commits that include no functional changes, such as API
changes like symbol renaming, should be tagged with NFC (No Functional Changes).
This signals to the reviewer why the change doesn't/shouldn't include a test.

`lit` tests with `FileCheck` are the preferred method of testing in MLIR for
non-erroneous output verification.

Diagnostic tests are the preferred method of asserting error messages are output
correctly. Every user-facing error message (e.g., `op.emitError()`) should be
accompanied by a corresponding diagnostic test.

When you cannot use the above, such as for testing a non-user-facing API like a
data structure, then you may write C++ unit tests. This is preferred because the
C++ APIs are not stable and subject to frequent refactoring. Using `lit` and
`FileCheck` allows maintainers to improve the MLIR internals more easily.

### FileCheck best practices

FileCheck is an extremely useful utility, it allows for easily matching various
parts of the output. This ease of use means that it becomes easy to write
brittle tests that are essentially `diff` tests. FileCheck tests should be as
self-contained as possible and focus on testing the minimal set of
functionalities needed. Let's see an example:

```mlir
// RUN: mlir-opt %s -cse | FileCheck %s

// CHECK-LABEL: func.func @simple_constant() -> (i32, i32)
func.func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %result = arith.constant 1 : i32
  // CHECK-NEXT: return %result, %result : i32, i32
  // CHECK-NEXT: }

  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  return %0, %1 : i32, i32
}
```

The above example is another way to write the original example shown in the main
[`lit` and `FileCheck` tests](#lit-and-filecheck-tests) section. There are a few
problems with this test; below is a breakdown of the no-nos of this test to
specifically highlight best practices.

*   Tests should be self-contained.

This means that tests should not test lines or sections outside of what is
intended. In the above example, we see lines such as `CHECK-NEXT: }`. This line
in particular is testing pieces of the Parser/Printer of FuncOp, which is
outside of the realm of concern for the CSE pass. This line should be removed.

*   Tests should be minimal, and only check what is absolutely necessary.

This means that anything in the output that is not core to the functionality
that you are testing should *not* be present in a CHECK line. This is a separate
bullet just to highlight the importance of it, especially when checking against
IR output.

If we naively remove the unrelated `CHECK` lines in our source file, we may end
up with:

```mlir
// CHECK-LABEL: func.func @simple_constant
func.func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %result = arith.constant 1 : i32
  // CHECK-NEXT: return %result, %result : i32, i32

  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  return %0, %1 : i32, i32
}
```

It may seem like this is a minimal test case, but it still checks several
aspects of the output that are unrelated to the CSE transformation. Namely the
result types of the `arith.constant` and `return` operations, as well the actual
SSA value names that are produced. FileCheck `CHECK` lines may contain
[regex statements](https://llvm.org/docs/CommandGuide/FileCheck.html#filecheck-regex-matching-syntax)
as well as named
[string substitution blocks](https://llvm.org/docs/CommandGuide/FileCheck.html#filecheck-string-substitution-blocks).
Utilizing the above, we end up with the example shown in the main
[FileCheck tests](#filecheck-tests) section.

```mlir
// CHECK-LABEL: func.func @simple_constant
func.func @simple_constant() -> (i32, i32) {
  /// Here we use a substitution variable as the output of the constant is
  /// useful for the test, but we omit as much as possible of everything else.
  // CHECK-NEXT: %[[RESULT:.*]] = arith.constant 1
  // CHECK-NEXT: return %[[RESULT]], %[[RESULT]]

  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  return %0, %1 : i32, i32
}
```

### Test Formatting Best Practices

When adding new tests, strive to follow these two key rules:

1. **Follow the existing naming and whitespace style.**
   - This applies when modifying existing test files that follow a particular
     convention, as it likely fits the context.
2. **Consistently document the edge case being tested.**
   - Clearly state what makes this test unique and how it complements other
     similar tests.

While the first rule extends LLVM’s general coding style to tests, the second
may feel new. The goal is to improve:

- **Test discoverability** – Well-documented tests make it easier to pair tests
  with patterns and understand their purpose.
- **Test consistency** – Consistent documentation and naming lowers cognitive
  load and helps avoid duplication.

A well-thought-out naming convention helps achieve all of the above.

---

#### Example: Improving Test Readability & Naming

Consider these **three tests** that exercise `vector.maskedload -> vector.load`
lowering under the `-test-vector-to-vector-lowering` flag:

##### Before: Inconsistent & Hard to Differentiate

```mlir
// CHECK-LABEL:   func @maskedload_regression_1(
//  CHECK-SAME:       %[[A0:.*]]: memref<16xf32>,
//  CHECK-SAME:       %[[A1:.*]]: vector<16xf32>) -> vector<16xf32> {
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[LOAD:.*]] = vector.load %[[A0]][%[[C0]]]
//  CHECK-SAME:     : memref<16xf32>, vector<16xf32>
//       CHECK:   return %[[LOAD]] : vector<16xf32>
func.func @maskedload_regression_1(
    %arg0: memref<16xf32>,
    %arg1: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index

  %vec_i1 = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.maskedload %arg0[%c0], %vec_i1, %arg1
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>

  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @maskedload_regression_2(
//  CHECK-SAME:       %[[A0:.*]]: memref<16xi8>,
//  CHECK-SAME:       %[[A1:.*]]: vector<16xi8>) -> vector<16xi8> {
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[LOAD:.*]] = vector.load %[[A0]][%[[C0]]]
//  CHECK-SAME:     : memref<16xi8>, vector<16xi8>
//       CHECK:   return %[[LOAD]] : vector<16xi8>
func.func @maskedload_regression_2(
    %arg0: memref<16xi8>,
    %arg1: vector<16xi8>) -> vector<16xi8> {
  %c0 = arith.constant 0 : index

  %vec_i1 = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.maskedload %arg0[%c0], %vec_i1, %arg1
    : memref<16xi8>, vector<16xi1>, vector<16xi8> into vector<16xi8>

  return %ld : vector<16xi8>
}

// CHECK-LABEL:   func @maskedload_regression_3(
// CHECK-SAME:        %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:        %[[A1:.*]]: vector<16xf32>) -> vector<16xf32> {
//      CHECK:    return %[[A1]] : vector<16xf32>
func.func @maskedload_regression_3(
    %arg0: memref<16xf32>,
    %arg1: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index

  %vec_i1 = vector.constant_mask [0] : vector<16xi1>
  %ld = vector.maskedload %arg0[%c0], %vec_i1, %arg1
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>

  return %ld : vector<16xf32>
}
```

While all examples test `vector.maskedload` -> `vector.load lowering`, it is
difficult to tell their actual differences.

##### After Step 1 (Introduce Consistent Variable Names)

To reduce cognitive load, use consistent names across MLIR and FileCheck (e.g.,
`%arg0` and `A0` above are not consistent). Also, instead of using generic
names like `%arg0` or `%vec_i1`, encode some additional context by using names
from existing documentation. For example from the Op documentation,
[`vector.maskedload`](https://mlir.llvm.org/docs/Dialects/Vector/#vectormaskedload-vectormaskedloadop),
in this case, you can use `%base`, `%mask` and `%pass_thru`.

```mlir
// CHECK-LABEL:   func @maskedload_regression_1(
//  CHECK-SAME:       %[[BASE:.*]]: memref<16xf32>,
//  CHECK-SAME:       %[[PASS_THRU:.*]]: vector<16xf32>) -> vector<16xf32> {
// (...)
func.func @maskedload_regression_1(
    %base: memref<16xf32>,
    %pass_thru: vector<16xf32>) -> vector<16xf32> {
  // (...)
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.maskedload %base[%c0], %mask, %pass_thru (...)
  // (...)
}

// CHECK-LABEL:   func @maskedload_regression_2(
//  CHECK-SAME:       %[[BASE:.*]]: memref<16xi8>,
//  CHECK-SAME:       %[[PASS_THRU:.*]]: vector<16xi8>) -> vector<16xi8> {
// (...)
func.func @maskedload_regression_2(
    %base: memref<16xi8>,
    %pass_thru: vector<16xi8>) -> vector<16xi8> {
  // (...)
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.maskedload %base[%c0], %mask, %pass_thru (...)
  // (...)
}

// CHECK-LABEL:   func @maskedload_regression_3(
//  CHECK-SAME:       %[[BASE:.*]]: memref<16xf32>,
//  CHECK-SAME:       %[[PASS_THRU:.*]]: vector<16xf32>) -> vector<16xf32> {
// (...)
func.func @maskedload_regression_3(
    %base: memref<16xf32>,
    %pass_thru: vector<16xf32>) -> vector<16xf32> {
  // (...)
  %mask = vector.constant_mask [0] : vector<16xi1>
  %ld = vector.maskedload %base[%c0], %mask, %pass_thru (...)
  // (...)
}
```

##### After Step 2 (Improve Test Naming)

Instead of using "regression" (which does not add unique information), rename
tests based on key attributes:

* All examples test the `vector.maskedload` to `vector.load` lowering.
* The first test uses a _dynamically_ shaped `memref`, while the others use
  _static_ shapes.
* The mask in the first two examples is "all true" (`vector.constant_mask
  [16]`), while it is "all false" (`vector.constant_mask [0]`) in the third
  example.
* The first and the third tests use `i32` elements, whereas the second uses
  `i8`.

This suggests the following naming scheme:
* `@maskedload_to_load_{static|dynamic}_{i32|i8}_{all_true|all_false}`.

Below are the updated names:

```mlir
// CHECK-LABEL:   func @maskedload_to_load_dynamic_i32_all_true(
// (...)
func.func @maskedload_to_load_dynamic_i32_all_true(...) -> vector<16xf32> {
  // (...)
}

// CHECK-LABEL:   func @maskedload_to_load_static_i8_all_true(
// (...)
func.func @maskedload_to_load_static_i8_all_true(...) -> vector<16xi8> {
  // (...)
}

// CHECK-LABEL:   func @maskedload_to_load_static_i32_all_false(
// (...)
func.func @maskedload_to_load_static_i32_all_false(...) -> vector<16xf32> {
  // (...)
}
```

##### After Step 3 (Add The Newly Identified Missing Case)

Step 2 made it possible to see that there is a case which is not tested:

* A mask that is neither "all true" nor "all false".

Unlike the existing cases, this mask must be preserved. In this scenario,
`vector.load` is not the right abstraction. Thus, no lowering should occur:

```mlir
// CHECK-LABEL:   func @negative_maskedload_to_load_static_i32_mixed(
// CHECK-SAME:        %[[BASE:.*]]: memref<16xf32>,
// CHECK-SAME:        %[[PASS_THRU:.*]]: vector<16xf32>) -> vector<16xf32> {
//      CHECK:    vector.maskedload
func.func @negative_maskedload_to_load_static_i32_mixed(
    %base: memref<16xf32>,
    %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [4] : vector<16xi1>

  %ld = vector.maskedload %base[%c0], %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>

  return %ld : vector<16xf32>
}
```

The `negative_` prefix indicates that this test should fail to lower, as the
pattern should not match.

##### Test Naming Convention
To summarize, here is the naming convention used in the examples above:

* `@{negative_}?maskedload_to_load_{static|dynamic}_{i32|i8}_{all_true|all_false|mixed}`.

The exact format may vary depending on context. However:
* **Avoid using suffixes** (e.g., `_fail`) to indicate negative tests — prefixes like
  `negative_` are easier to spot and grep for.
* Whatever naming convention you choose, **apply it consistently** throughout
  the test suite.

**Note:** In some cases, a prefix other than `negative_` might be more
appropriate. For instance, in "folding" tests where a pattern is expected not
to apply, using `no_` can be a more concise and equally clear alternative —
e.g., `@no_fold_<case>_<subcase>.`

#### What if there is no pre-existing style to follow?

If you are adding a new test file, you can use other test files in the same
directory as inspiration.

If the test file you are modifying lacks a clear style and instead has mixed,
inconsistent styles, try to identify the dominant one and follow it. Even
better, consider refactoring the file to adopt a single, consistent style —
this helps improve our overall testing quality. Refactoring is also encouraged
when the existing style could be improved.

In many cases, it is best to create a separate PR for test refactoring to
reduce per-PR noise. However, this depends on the scale of changes — reducing
PR traffic is also important. Work with reviewers to use your judgment and
decide the best approach.

Alternatively, if you defer refactoring, consider creating a GitHub issue and
adding a TODO in the test file linking to it.

When creating a new naming convention, keep these points in mind:

* **Write Orthogonal Tests**
If naming is difficult then the tests may be lacking a clear purpose. A good
rule of thumb is to avoid testing the same thing repeatedly. Before writing
tests, define clear categories to cover (e.g., number of loops, data types).
This often leads to a natural naming scheme—for example: `@loop_depth_2_i32`.

* **What vs Why**
Test names should reflect _what_ is being tested, not _why_.

Encoding _why_ in test names can lead to overly long and complex names.
Instead, add inline comments where needed.

#### Do not forget the common sense

Always apply common sense when naming functions and variables. Encoding too
much information in names makes the tests less readable and less maintainable.

Trust your judgment. When in doubt, consult your "future self": _"Will this still
make sense to me six months from now?_"

#### Final Points - Key Principles

The above approach is just an example. It may not fit your use case perfectly,
so feel free to adapt it as needed.  Key principles to follow:

* Make tests self-documenting.
* Follow existing conventions.

These principles make tests easier to discover and maintain. For you, "future
you", and the rest of the MLIR community.

### Test Documentation Best Practices

In addition to following good naming and formatting conventions, please
document your tests with comments. Focus on explaining **why** since the
**what** is usually clear from the code itself.

As an example, consider this test that uses the
`TransferWritePermutationLowering` pattern:


```mlir
/// Even with out-of-bounds accesses, it is safe to apply this pattern as it
/// does not modify which memory location is being accessed.

// CHECK-LABEL:   func.func @xfer_write_minor_identity_transposed_out_of_bounds
//  CHECK-SAME:      %[[VEC:.*]]: vector<4x8xi16>
//  CHECK-SAME:      %[[MEM:.*]]: memref<2x2x?x?xi16>
//  CHECK-SAME:      %[[IDX:.*]]: index)
//       CHECK:      %[[TR:.*]] = vector.transpose %[[VEC]], [1, 0]
//  CHECK-SAME:        : vector<4x8xi16> to vector<8x4xi16>

/// Expect the in_bounds attribute to be preserved. However, since we don't
/// print it when all flags are "false", it should not appear in the output.
/// CHECK-NOT:       in_bounds

// CHECK:           vector.transfer_write

/// The permutation map was replaced with vector.transpose
// CHECK-NOT:       permutation_map

// CHECK-SAME:        %[[TR]], %[[MEM]][%[[IDX]], %[[IDX]], %[[IDX]], %[[IDX]]]
// CHECK-SAME:        : vector<8x4xi16>, memref<2x2x?x?xi16>
func.func @xfer_write_minor_identity_transposed_out_of_bounds(
    %vec: vector<4x8xi16>,
    %mem: memref<2x2x?x?xi16>,
    %idx: index) {

  vector.transfer_write %vec, %mem[%idx, %idx, %idx, %idx] {
    in_bounds = [false, false],
    permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
  } : vector<4x8xi16>, memref<2x2x?x?xi16>

  return
}
```

The comments in the example above document two non-obvious behaviors:

* _Why_ is the `permutation_map` attribute missing from the output?
* _Why_ is the `in_bounds` attribute missing from the output?


#### How to Identify What Needs Documentation?
Think of yourself six months from now and ask: _"What might be difficult to
understand without comments?"_

If you expect something to be tricky for "future-you", it’s likely to be tricky
for others encountering the test for the first time.

#### Making Tests Self-Documenting
We can improve documentation further by:
* clarifying what pattern is being tested,
* providing high-level reasoning, and
* consolidating shared comments.

For example:

```mlir
///--------------------------------------------------------------------------------
/// [Pattern: TransferWritePermutationLowering]
///
/// IN: vector.transfer_write (_transposed_ minor identity permutation map)
/// OUT: vector.transpose + vector.transfer_write (minor identity permutation map)
///
/// Note: `permutation_map` from the input Op is replaced with the newly
/// inserted vector.transpose Op.
///--------------------------------------------------------------------------------
// CHECK-LABEL:   func.func @xfer_write_minor_identity_transposed
//       (...)
//       CHECK:      %[[TR:.*]] = vector.transpose (...)
//       CHECK:      vector.transfer_write %[[TR]] (...)
//       (...)
```

The example above documents:
* The transformation pattern being tested.
* The key logic behind the transformation.
* The expected change in output.


#### Documenting the "What"
You should always document why, but documenting what is also valid and
encouraged in cases where:

* The test output is long and complex.
* The tested logic is non-trivial and/or involves multiple transformations.

For example, in this test for Linalg convolution vectorization, comments are
used to document high-level steps (original FileCheck "check" lines have been
trimmed for brevity):

```mlir
func.func @conv1d_nwc_4x2x8_memref(
    %input: memref<4x6x3xf32>,
    %filter: memref<1x3x8xf32>,
    %output: memref<4x2x8xf32>) {
  linalg.conv_1d_nwc_wcf
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x6x3xf32>, memref<1x3x8xf32>)
    outs(%output : memref<4x2x8xf32>)
  return
}

//      CHECK: func @conv1d_nwc_4x2x8_memref
// CHECK-SAME: %[[INPUT:.+]]: memref<4x6x3xf32>
// CHECK-SAME: %[[FILTER:.+]]: memref<1x3x8xf32>,
// CHECK-SAME: %[[OUTPUT:.+]]: memref<4x2x8xf32>

/// Read the whole data in one shot.
//  CHECK-DAG:   %[[V_INPUT_R:.+]] = vector.transfer_read %[[INPUT]]
//  CHECK-DAG:  %[[V_FILTER_R:.+]] = vector.transfer_read %[[FILTER]]
//  CHECK-DAG:  %[[V_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]]

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]

//      CHECK:    %[[V_FILTER:.+]] = vector.extract %[[V_FILTER_R]][0]

//      CHECK:  %[[V_OUTPUT_0:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
//      CHECK:  %[[V_OUTPUT_1:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]

/// w == 0, kw == 0
//      CHECK:   %[[CONTRACT_0:.+]] = vector.contract
// CHECK-SAME:     %[[V_INPUT_0]], %[[V_FILTER]], %[[V_OUTPUT_0]]

/// w == 1, kw == 0
//      CHECK:   %[[CONTRACT_1:.+]] = vector.contract
// CHECK-SAME:     %[[V_INPUT_1]], %[[V_FILTER]], %[[V_OUTPUT_1]]

/// w == 0, kw == 0
//      CHECK:   %[[RES_0:.+]] = vector.insert_strided_slice
// CHECK-SAME:    %[[CONTRACT_0]], %[[V_OUTPUT_R]]
/// w == 1, kw == 0
//      CHECK:   %[[RES_1:.+]] = vector.insert_strided_slice
// CHECK-SAME:    %[[CONTRACT_1]], %[[RES_0]]

/// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[RES_1]], %[[OUTPUT]]
```

Though the comments document _what_ is happening (e.g., "Write the result back
in one shot"), some variables — like `w` and `kw` — are not explained. This is
intentional - their purpose becomes clear when studying the corresponding
Linalg vectorizer implementation (or, when analysing how
`linalg.conv_1d_nwc_wcf` works).

Comments help you understand code, they do not replace the need to read it.
Comments guide the reader, they do not repeat what the code already says.

#### Final Points - Key Principles
Below are key principles to follow when documenting tests:
* Always document _why_, document _what_ if you need to (e.g. the underlying
	logic is non-trivial).
* Use block comments for higher-level comments (e.g. to describe the patterns
	being tested).
* Think about maintainability - comments should help future developers (which
	includes you) understand tests at a glance.
* Avoid over-explaining. Comments should assist, not replace reading the code.
