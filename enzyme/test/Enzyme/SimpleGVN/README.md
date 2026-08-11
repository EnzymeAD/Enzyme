# SimpleGVN Pass Tests

This directory contains tests for the SimpleGVN optimization pass.

## What is SimpleGVN?

SimpleGVN is a GVN-like (Global Value Numbering) optimization pass that forwards loads from `noalias` and `nocapture` function arguments to their corresponding stores. Unlike LLVM's built-in GVN pass, SimpleGVN does not have a limit on the number of instructions or memory offsets it will analyze.

## How It Works

The pass operates in two phases:

### Phase 1: Store-to-Load Forwarding
1. Identifies function arguments with both `noalias` and `nocapture` attributes, and allocas
2. Verifies all uses are exclusively loads, stores, GEP instructions, or casts
3. For each load, finds dominating stores that cover the load's memory range
4. Replaces the load with the stored value if no aliasing store exists in between

### Phase 2: Load-to-Load Forwarding
1. Re-collects loads and stores, this time allowing nocapture function calls
2. For each load, finds dominating loads that cover the same memory range
3. Replaces the load with the value from the dominating load if:
   - No aliasing store exists between the two loads
   - No nocapture function call exists between the two loads (as they may modify memory)

## Test Cases

### Store-to-Load Forwarding Tests
- **basic.ll** - Simple store-to-load forwarding
- **offset.ll** - Forwarding with GEP offsets
- **dominance.ll** - Verifies dominance requirements
- **intermediate_store.ll** - Handles intermediate stores correctly
- **no_noalias.ll** - Rejects optimization when noalias is missing
- **call_use.ll** - Rejects when argument has non-memory uses (non-nocapture calls)
- **struct_field.ll** - Handles struct field accesses
- **type_conversion.ll** - Tests byte-level extraction
- **comprehensive.ll** - Multiple loads/stores at different offsets

### Load-to-Load Forwarding Tests
- **load_load_basic.ll** - Simple load-to-load forwarding
- **load_load_offset.ll** - Load-to-load forwarding with GEP offsets
- **load_load_nocapture_call.ll** - No forwarding when nocapture call exists between loads
- **load_load_no_nocapture_call.ll** - Optimization disabled when call lacks nocapture attribute
- **load_load_with_store_between.ll** - No load-to-load forwarding when store exists between loads

## Running the Tests

Using opt with the new pass manager:
```bash
opt -load-pass-plugin=LLVMEnzyme-18.so -passes="simple-gvn" -S < test.ll
```

Using opt with the legacy pass manager (LLVM < 16):
```bash
opt -load LLVMEnzyme-18.so -simple-gvn -S < test.ll
```

## Examples

### Store-to-Load Forwarding Example

Input:
```llvm
define i32 @foo(ptr noalias nocapture %ptr) {
  store i32 42, ptr %ptr
  %v = load i32, ptr %ptr
  ret i32 %v
}
```

Output after SimpleGVN:
```llvm
define i32 @foo(ptr noalias nocapture %ptr) {
  store i32 42, ptr %ptr
  ret i32 42
}
```

### Load-to-Load Forwarding Example

Input:
```llvm
define i32 @bar(ptr noalias nocapture %ptr) {
  %v1 = load i32, ptr %ptr
  %v2 = load i32, ptr %ptr
  %sum = add i32 %v1, %v2
  ret i32 %sum
}
```

Output after SimpleGVN:
```llvm
define i32 @bar(ptr noalias nocapture %ptr) {
  %v1 = load i32, ptr %ptr
  %sum = add i32 %v1, %v1
  ret i32 %sum
}
```
