# SimpleGVN Pass Tests

This directory contains tests for the SimpleGVN optimization pass.

## What is SimpleGVN?

SimpleGVN is a GVN-like (Global Value Numbering) optimization pass that forwards loads from `noalias` and `nocapture` function arguments to their corresponding stores. Unlike LLVM's built-in GVN pass, SimpleGVN does not have a limit on the number of instructions or memory offsets it will analyze.

## How It Works

The pass:
1. Identifies function arguments with both `noalias` and `nocapture` attributes
2. Verifies all uses are exclusively loads, stores, or GEP instructions
3. For each load, finds dominating stores that cover the load's memory range
4. Replaces the load with the stored value if no aliasing store exists in between

## Test Cases

- **basic.ll** - Simple store-to-load forwarding
- **offset.ll** - Forwarding with GEP offsets
- **dominance.ll** - Verifies dominance requirements
- **intermediate_store.ll** - Handles intermediate stores correctly
- **no_noalias.ll** - Rejects optimization when noalias is missing
- **call_use.ll** - Rejects when argument has non-memory uses
- **struct_field.ll** - Handles struct field accesses
- **type_conversion.ll** - Tests byte-level extraction
- **comprehensive.ll** - Multiple loads/stores at different offsets

## Running the Tests

Using opt with the new pass manager:
```bash
opt -load-pass-plugin=LLVMEnzyme-18.so -passes="simple-gvn" -S < test.ll
```

Using opt with the legacy pass manager (LLVM < 16):
```bash
opt -load LLVMEnzyme-18.so -simple-gvn -S < test.ll
```

## Example

Input:
```llvm
define i32 @foo(i32* noalias nocapture %ptr) {
  store i32 42, i32* %ptr
  %v = load i32, i32* %ptr
  ret i32 %v
}
```

Output after SimpleGVN:
```llvm
define i32 @foo(i32* noalias nocapture %ptr) {
  store i32 42, i32* %ptr
  ret i32 42
}
```
