---
title: "Using Enzyme"
date: 2019-11-29T15:26:15Z
draft: false
weight: 20
---

## Generating LLVM

To begin, let's create a simple code `test.c` we want to differentiate. Enzyme will replace any calls to functions whose names contain "\_\_enzyme\_autodiff" with calls to the corresponding For now, let's ignore the details of Enzyme's calling convention/ABI which are described in detail [here](/getting_started/CallingConvention)

```c
// test.c
#include <stdio.h>
extern double __enzyme_autodiff(void*, double);
double square(double x) {
    return x * x;
}
double dsquare(double x) {
    // This returns the derivative of square or 2 * x
    return __enzyme_autodiff(square, x);
}
int main() {
    for(double i=1; i<5; i++)
        printf("square(%f)=%f, dsquare(%f)=%f", i, square(i), i, dsquare(i));
}
```

We can generate LLVM from this code by calling clang as follows. Note that `clang` should be the path to whatever clang you built Enzyme against.
```sh
clang test.c -S -emit-llvm -o input.ll -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops
```

The arguments `-S -emit-llvm` specify that we want to emit LLVM bitcode rather than an executable. The arguments `-o input.ll` specify that we want the output to be in a file `input.ll`. The argument `-O2 -ffast-math` runs optimizations (with fast-math) before we run Enzyme's AD process, which is often beneficial for performance. The argument `-fno-vectorize -fno-slp-vectorize -fno-unroll-loops` specifies that we don't want to run vectorization or loop unrolling. In practice, it is better for performance to only run these scheduling optimizations after AD.

The generated LLVM IR should look something like the following
```llvm
; input.ll
...
define double @square(double %x) #0 {
entry:
  %mul = fmul double %x, %x
  ret double %mul
}

define double @dsquare(double %x) local_unnamed_addr #1 {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (double (double)* @square to i8*), double %x) #4
  ret double %call
}
...
```

## Performing AD Enzyme
We can now run Enzyme to differentiate our LLVM IR. The following command will load Enzyme and run the differentiation transformation pass. Note that `opt` should be the path to whatever opt was creating by the LLVM you built Enzyme against. If you see a segfault when trying to run opt, this is likely an issue in LLVM's plugin infrasture. Please see [the installation guide](/getting_started/Installation) for more information on how to resolve this.

```sh
opt input.ll -load=/path/to/Enzyme/enzyme/build/Enzyme/LLVMEnzyme-<VERSION>.so -enzyme -o output.ll -S
```

Taking a look at `output.ll`, we find the following:

```llvm
; output.ll
define internal { double } @diffesquare(double %x, double %differeturn) #0 {
entry:
  %"mul'de" = alloca double
  store double 0.000000e+00, double* %"mul'de"
  %"x'de" = alloca double
  store double 0.000000e+00, double* %"x'de"
  br label %invertentry

invertentry:                                      ; preds = %entry
  store double %differeturn, double* %"mul'de"
  %0 = load double, double* %"mul'de"
  %m0diffex = fmul fast double %0, %x
  %m1diffex = fmul fast double %0, %x
  store double 0.000000e+00, double* %"mul'de"
  %1 = load double, double* %"x'de"
  %2 = fadd fast double %1, %m0diffex
  store double %2, double* %"x'de"
  %3 = load double, double* %"x'de"
  %4 = fadd fast double %3, %m1diffex
  store double %4, double* %"x'de"
  %5 = load double, double* %"x'de"
  %6 = insertvalue { double } undef, double %5, 0
  ret { double } %6
}

define double @dsquare(double %x) local_unnamed_addr #1 {
entry:
  %0 = call { double } @diffesquare(double %x, double 1.000000e+00)
  %1 = extractvalue { double } %0, 0
  ret double %1
}
```

Enzyme has created a new gradient function and replaced the corresponding call to `__enzyme_autodiff`. Note that newly-created gradient function isn't yet optimized. Enzyme assumes that various post-processing will occur after creating the gradient.

For example, suppose we run `-O2` after Enzyme as shown below:
```sh
opt output.ll -O2 -o output_opt.ll -S
```

Taking a look at `output_opt.ll`, we see the following:

```llvm
; output_opt.ll
define double @dsquare(double %x) local_unnamed_addr #0 {
entry:
  %factor.i = fmul fast double %x, 2.000000e+00
  ret double %factor.i
}
```

The generated gradient has been inlined and entirely simplified to return the input times two.

## Advanced options

Enzyme has several advanced options that may be of interest.

### Performance options

#### Disabling Preprocessing

The `enzyme-preopt` option disables the preprocessing optimizations run by the Enzyme pass, except for the absolute minimum neccessary.

```sh
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-preopt=1
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-preopt=0
```

#### Forced Inlining

The `enzyme-inline` option forcibly inlines all subfunction calls. The `enzyme-inline-count` option limits the number of calls inlined by this utility.

```sh
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-inline=1
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-inline=1 -enzyme-inline-count=100
```

#### Compressed Bool Cache

The `enzyme-smallbool` option allows Enzyme's cache to store 8 boolean (i1) values inside a single byte rather than one value per byte.

```sh
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-smallbool=1
```

### Semantic options

#### Loose type analysis

The `enzyme-loose-types` option tells Enzyme to make an educated guess about the type of a value it cannot prove, rather than emit a compile-time error and fail. This can be helpful for starting to bootstrap code with Enzyme but shouldn't be used in production as Enzyme may make an incorrect guess and create an incorrect gradient.

```sh
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-loose-types=1
```


#### Assume inactivity of undefined functions

The `enzyme-emptyfn-inactive` option tells activity analysis to assume that all calls to functions whose definitions aren't available and aren't explicitly given a custom gradient via metadata are assumed to be inactive. This can be useful for assuming printing functions don't impact derivative computations and provide a performance benefit, as well as getting around a compile-time error where the derivative of a foreign function is not known. However, this option should be used carefully as it may result in incorrect behavior if it is used to incorrectly assume a call to a foreign function doesn't impact  the derivative computation. As a result, the recommended way to remedy this is to mark the function as inactive explicitly, or provide a custom gradient via metadata.

```sh
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-emptyfn-inactive=1
```

#### Assume inactivity of unmarked globals

The `enzyme-globals-default-inactive` option tells activity analysis to assume that global variables without an explicitly defined shadow global are assumed to be inactive. Like `enzyme_emptyfnconst`, this option should be used carefully as it may result in incorrect behavior if it is used to incorrectly assume that a global variable doesn't contain data used in a derivative computation.

```sh
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-globals-default-inactive=1
```

#### Cache behavior

The `enzyme-cache-never` option tells the cache to recompute all load values, even if alias analysis isn't able to prove the legality of such a recomputation. This may improve performance but is likely to result in incorrect derivatives being produced as this is not generally true.

```sh
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-cache-never=1
```

In contrast, the `enzyme-cache-always` option tells the cache to still cache values that alias analysis and differential use analysis say are not needed to be cached (perhaps being legal to recompute instead). This will usually decrease performance and is intended for developers in order to catch caching bugs.

```sh
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-cache-always=1
```

### Debugging options for developers

#### enzyme-print

This option prints out functions being differentiated before preprocessing optimizations, after preprocessing optimizations, and after being synthesized by Enzyme. It is mostly use to debug the AD process.

```sh
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-print
prefn:

; Function Attrs: norecurse nounwind readnone uwtable
define double @square(double %x) #0 {
entry:
  %mul = fmul double %x, %x
  ret double %mul
}
```

#### enzyme-print-activity

This option prints out the results of activity analysis as they are being derived. The output is somewaht specific to the analysis pass and is only intended for developers.

```sh
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-print-activity
in new function diffesquare nonconstant arg double %0
 VALUE nonconst from arg nonconst double %x
checking if is constant[3]   %mul = fmul double %x, %x
 < UPSEARCH3>  %mul = fmul double %x, %x
 VALUE nonconst from arg nonconst double %x
nonconstant(3)  up-inst   %mul = fmul double %x, %x op double %x
 </UPSEARCH3>  %mul = fmul double %x, %x
couldnt decide nonconstants(3):  %mul = fmul double %x, %x
 Value nonconstant (couldn't disprove)[3]  %mul = fmul double %x, %x
```

#### enzyme-print-type

This option prints out the results of type analysis as they are being derived. The output is somewaht specific to the analysis pass and is only intended for developers.

```sh
$ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-print-type
analyzing function square
 + knowndata: double %x : {[-1]:Float@double} - {}
 + retdata: {}
updating analysis of val: double %x current: {} new {[-1]:Float@double}
updating analysis of val: double %x current: {[-1]:Float@double} new {[-1]:Float@double} from double %x
updating analysis of val:   %mul = fmul double %x, %x current: {} new {}
updating analysis of val: double %x current: {[-1]:Float@double} new {[-1]:Float@double} from   %mul = fmul double %x, %x
updating analysis of val: double %x current: {[-1]:Float@double} new {[-1]:Float@double} from   %mul = fmul double %x, %x
updating analysis of val:   %mul = fmul double %x, %x current: {} new {[-1]:Float@double} from   %mul = fmul double %x, %x
```
