---
title: "FAQ"
date: "2019-11-29"
menu: "main"
weight: 40
---

## Enzyme builds successfully but won't run tests

Double check that Enzyme's build can find lit, LLVM's testing framework. This can be done by explicitly passing lit as an argument to CMake as described [here](/Installation).

## Enzyme is Crashing

LLVM's plugin infrastructure is broken in many versions. Empirically LLVM 8 and up will often incorrectly disallow memory from being passed between LLVM and a plugin. If you see one of these errors and want to use the same version of LLVM try passing the flag `enzyme_preopt=0` described [here](/getting_started/UsingEnzyme). The flag disables preprocessing optimizations that Enzyme runs and tends to reduce these errors. If this doesn't work, check the following.

### Illegal TypeAnalysis on LLVM 10+

There is a [known bug](https://bugs.llvm.org/show_bug.cgi?id=47612) in an existing LLVM optimization pass (SROA) that will incorrectly generate type information from a memcpy. This bug has been fixed in LLVM 13

### opt can't find -enzyme option

When you use opt command like below:
`
opt input.ll -load=/path/to/LLVMEnzyme-VERSION.so -enzyme -o output.ll -S
`
opt reports a error:`opt: unknown pass name 'enzyme'`

it is beacause LLVM 13 and newer have switched to the new pass manager pipeline and we haven't yet turned that on within enzyme.
For now you can just tell llvm to use the old pass manager by adding this flag to opt --enable-new-pm=0 or this flag to clang -fno-experimental-new-pass-manager

### UNREACHABLE executed (GVN error)

Until June 2020, LLVM's exisitng GVN pass had a bug handling invariant.load's that would cause it to crash. These tend to be generated a lot by Enzyme for better optimization. This was reported [here](https://bugs.llvm.org/show_bug.cgi?id=46054) and resolved in master. Options for resolving include updating to later verison of LLVM with the fix, or disabling creation of invariant.load's.

## Other

If you have an issue not resolved here, please make an issue on [Github](https://github.com/wsmoses/Enzyme) and consider making a pull request to this FAQ!
