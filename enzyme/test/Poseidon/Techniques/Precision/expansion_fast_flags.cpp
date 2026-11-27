// DEFINE: %{expand} = %opt %loadPoseidon -load-pass-plugin=%poseidon_test_programs/poseidon-test-expansion-match%shlibext -passes=apply-expansion -S
// DEFINE: %{harness} = %clang++ -std=c++17 -O2 -ffp-contract=off -I%S/../../Inputs/expansion_match %t/harness.cpp
// RUN: rm -rf %t && split-file %s %t
// RUN: sed -e 's/^\(define .*)\) {$/\1 #0 {/' -e '$a attributes #0 = { "unsafe-fp-math"="true" "no-nans-fp-math"="true" "no-infs-fp-math"="true" "approx-func-fp-math"="true" "no-signed-zeros-fp-math"="true" }' %t/fast.ll > %t/fast_attributes.ll
// RUN: %{expand} %t/fast.ll -o %t/expanded.ll
// RUN: FileCheck --check-prefix=IR --implicit-check-not='{{ (fast|contract|reassoc|nnan|ninf|nsz|arcp|afn) }}' %s < %t/expanded.ll
// RUN: %clang++ -O2 -c %t/expanded.ll -o %t/expanded.o
// RUN: %{harness} %t/expanded.o -o %t/run
// RUN: %t/run | FileCheck --check-prefix=OUT %s
// RUN: %if !global-fp-contract %{ %{expand} %t/fast_attributes.ll -o %t/attributes.ll %}
// RUN: %if !global-fp-contract %{ %clang++ -O2 -c %t/attributes.ll -o %t/attributes.o %}
// RUN: %if !global-fp-contract %{ %{harness} %t/attributes.o -o %t/run_attributes %}
// RUN: %if !global-fp-contract %{ %t/run_attributes | FileCheck --check-prefix=ATTR %s %}

// FP64 operations carrying every fast-math flag (fast.ll) are materialized for
// n = 2, 3, 4 as operations with none (IR); compiled at -O2 the two-component
// sum and product equal MultiFloats.jl's on the operands' splits (OUT). The
// same holds under every fast-math function attribute (ATTR), which from
// LLVM 24 reaches no flagless operation.

// IR-LABEL: define double @ex2_add(
// IR: fptrunc double %a to float
// IR-LABEL: define double @ex2_mul(
// IR: call float @llvm.fma.f32(
// IR-LABEL: define double @ex2_div(
// IR: fdiv float
// IR-LABEL: define double @ex2_fma(
// IR: call float @llvm.fma.f32(
// IR-LABEL: define double @ex3_mul(
// IR: call float @llvm.fma.f32(
// IR-LABEL: define double @ex4_add(
// IR: fadd float

// OUT: ex2_add(0.1, 0.2) = 0x1.333333333333p-2
// OUT: sum 100000 of 100000, product 100000 of 100000 equal to the reference

// ATTR: ex2_add(0.1, 0.2) = 0x1.333333333333p-2
// ATTR-NEXT: sum 100000 of 100000, product 100000 of 100000 equal to the reference
// ATTR-NOT: {{.}}

//--- fast.ll
define double @ex2_add(double %a, double %b) {
  %r = fadd fast double %a, %b
  ret double %r
}

define double @ex2_mul(double %a, double %b) {
  %r = fmul fast double %a, %b
  ret double %r
}

define double @ex2_div(double %a, double %b) {
  %r = fdiv fast double %a, %b
  ret double %r
}

define double @ex2_fma(double %a, double %b, double %c) {
  %r = call fast double @llvm.fmuladd.f64(double %a, double %b, double %c)
  ret double %r
}

define double @ex3_mul(double %a, double %b) {
  %r = fmul fast double %a, %b
  ret double %r
}

define double @ex4_add(double %a, double %b) {
  %r = fadd fast double %a, %b
  ret double %r
}

declare double @llvm.fmuladd.f64(double, double, double)

//--- harness.cpp
// The expanded ex2_add and ex2_mul against MultiFloats.jl's two-limb addition
// and multiplication on the operands' splits.
#include "multifloats_subset.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" double ex2_add(double, double);
extern "C" double ex2_mul(double, double);

namespace {

struct Pair {
  float x[2];
};

Pair split(double x) {
  float hi = (float)x;
  return {{hi, (float)(x - (double)hi)}};
}

double join(Pair p) { return (double)p.x[0] + (double)p.x[1]; }

Pair mfadd(Pair x, Pair y) {
  Pair r;
  mfref::mfadd(x.x, y.x, r.x);
  return r;
}

Pair mfmul(Pair x, Pair y) {
  Pair r;
  mfref::mfmul(x.x, y.x, r.x);
  return r;
}

uint64_t bits(double d) {
  uint64_t b;
  memcpy(&b, &d, 8);
  return b;
}

uint64_t g_state = 0x9e3779b97f4a7c15ull;
double operand() {
  g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
  double m = 1.0 + (double)(g_state >> 11) * 0x1.0p-53;
  int e = (int)((g_state >> 3) % 61) - 30;
  return (g_state & 1 ? -1.0 : 1.0) * std::ldexp(m, e);
}

} // namespace

int main() {
  const int n = 100000;
  long add = 0, mul = 0;
  for (int i = 0; i < n; ++i) {
    double a = i ? operand() : 0.1, b = i ? operand() : 0.2;
    add += bits(ex2_add(a, b)) == bits(join(mfadd(split(a), split(b))));
    mul += bits(ex2_mul(a, b)) == bits(join(mfmul(split(a), split(b))));
  }
  printf("ex2_add(0.1, 0.2) = %a\n", ex2_add(0.1, 0.2));
  printf("sum %ld of %d, product %ld of %d equal to the reference\n", add, n,
         mul, n);
  return add != n || mul != n;
}
