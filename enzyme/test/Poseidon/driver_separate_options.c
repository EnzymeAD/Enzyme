// RUN: %python %S/../../Poseidon/tools/driver/clang_separate_options.py check %clang %S/../../Poseidon/tools/driver/ClangSeparateOptions.def

// poseidon-clang keeps the value of every clang driver option whose value is
// the next argv entry paired with its option. Its table is clang's: every
// option of the table that the clang of this build knows takes a separate value
// there (with the LLVM versions a table entry names), and every option this
// clang offers for completion that takes one is in the table or among the
// JoinedAndSeparate ones the driver matches by prefix. The script asks the
// clang itself (-### with a value that would otherwise be an unknown argument),
// so the test needs no clang sources.

int main(void) { return 0; }
