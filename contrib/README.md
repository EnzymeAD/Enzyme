# Shipping a Julia-compatible Enzyme library
Since Julia has LLVM as a core dependency and uses a heavily patched version
we want to ship enzyme build against the same LLVM version in a way that is compatible
ABI compatible with the LLVM version used by Julia.

## Instructions

```
julia --project=. build_tarballs.jl --debug x86_64-linux-gnu
```
