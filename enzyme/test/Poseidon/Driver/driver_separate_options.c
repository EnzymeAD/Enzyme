// RUN: %python %S/../../../Poseidon/tools/driver/clang_separate_options.py check %clang %S/../../../Poseidon/tools/driver/ClangSeparateOptions.def

// poseidon-clang keeps the value of every clang driver option whose value is
// the next argv entry paired with its option. Its table
// (tools/driver/ClangSeparateOptions.def) matches the clang of this build:
// every entry this clang knows takes a separate value (within the LLVM versions
// the entry names), and every option this clang offers for completion that
// takes one is in the table or among the JoinedAndSeparate ones matched by
// prefix. The script queries the clang binary, so no clang sources are needed.
