---
title: "Debugging"
date: "2020-03-30"
menu: "main"
weight: 10
---

# MLIR Debugging Tips

## Inspecting compilation

There's no silver bullet for debugging the compilation process. Standard debugging techniques (printf debugging, gdb/lldb, IDE graphical debuggers, etc.) are of course applicable, but below are MLIR-specific facilities that are quite useful before diving into a generic debug flow. These facilities assume that you have reduced your problem to a form that can be reproduced with `mlir-opt` or another program that hooks into MLIR's option parsing, if this is not the case, see section "Isolating test case" below.

- `-mlir-print-stacktrace-on-diagnostic` causes a stacktrace to be  printed when a diagnostic is emitted. This can be useful to quickly get an idea where in a pass an error is happening.

- When dealing with verifier errors, `--verify-each=0` turns off the verifier, allowing one to see the full invalid IR
  - `-mlir-print-op-generic` is often useful in combination, since many verifier errors also result in instructions that cannot be printed in their "pretty form".

- When using the dialect conversion / pattern rewriting infrastructure `-debug-only=dialect-conversion` prints an exceedingly useful trace of the decisions that were made and why.

- `-debug-only=mydebugtag` coupled with use of the `LLVM_DEBUG` facility to just get debug info for things you are working on.

- `-mlir-elide-elementsattrs-if-larger` prints large constants in a redacted form, making the IR easier to scan.

## Isolating test cases

Isolating your problem to the inspection of manageable set of passes (ideally a single pass) is one of the most important parts of debugging a compiler, but sometimes this can be difficult in larger compilation flow.

### Extracting a `.mlir` file and a pass pipeline.

MLIR's core infrastructure has the ability to [create a "crash reproducer"](../docs/PassManagement.md#crash-and-failure-reproduction) and this functionality should be added to your compilation flows. Additionally, you should ensure that `.mlir` files are dumped during your compilation flow at key points such that even if compilation succeeds (possibly spuriously, such as with a miscompile), then there is still a starting point for dropping into `mlir-opt`.

### Isolating a buggy pass

Once one has a `.mlir` file and a pass pipeline to run (such as that dumped in the crash reproducer file), then one should use `-print-ir-before-all` option to `mlir-opt` to print the IR before each pass. Additionally, `-print-ir-module-scope` and `-mlir-disable-threading` can be useful here.

If one is developing an MLIR-based compiler correctly (that is, with an attention to correct program translation, high quality verifiers, and clear diagnostics when correct translation isn't possible), then by far the most common class of bugs is that an error diagnostic is emitted. When this happens, the recommended course of action is to run all passes up to just before the pass that emitted the error diagnostic and save off a new `.mlir` file representing the IR just before the problematic pass was run.

If the problem is more complex than just a diagnostic / verifier error in a single pass, then more analysis of the `-print-ir-before-all` dumps will be needed, possibly by inserting extra debug printing into various passes to see where things went off the rails.

Either way, at the end of this process one should ideally have a `.mlir` file and a single pass (or manageable set of passes) to run in `mlir-opt` for further analysis.

TODO: testcase reduction, debugging miscompiles, bisection tools, general philosophical discussion of when and how to use bisection
