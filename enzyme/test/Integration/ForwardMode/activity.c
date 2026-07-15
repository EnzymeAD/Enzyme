// RUN: echo "mysin" > %t.inactive_sin.list; echo "mycos" > %t.inactive_cos.list
// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - | FileCheck %s --check-prefix=CHECK-CORRECT
// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme --enzyme-load-inactive-file="%t.inactive_sin.list" --enzyme-load-inactive-file="%t.inactive_cos.list" -S | %lli - | FileCheck %s --check-prefix=CHECK-INACTIVE-INCORRECT
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme --enzyme-load-inactive-file="%t.inactive_sin.list" --enzyme-load-inactive-file="%t.inactive_cos.list" -S | %lli - | FileCheck %s --check-prefix=CHECK-INACTIVE-INCORRECT
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme --enzyme-load-inactive-file="%t.inactive_sin.list" --enzyme-load-inactive-file="%t.inactive_cos.list" --enzyme-print-activity -S 2>&1 | grep -F "[activity]" | FileCheck %s --check-prefix=CHECK-PRINT

#include "stdio.h"
#include "../test_utils.h"
#include "math.h"

__attribute__((noinline)) double mysin(double x) { return sin(x); };
__attribute__((noinline)) double mycos(double x) { return cos(x); };

extern double __enzyme_fwddiff(void*, ...);

double f(double x) {
  return mysin(x) * mycos(x);
}

int main() {
  double xs[] = {0.0, 0.5, 1.0, 2.0, 3.0};

 for (int i = 0; i < 5; i++) {
    double d_ret = __enzyme_fwddiff((void*)f, xs[i], 1.0);
    double expected = cos(xs[i]) * cos(xs[i]) - sin(xs[i]) * sin(xs[i]);
    printf("i=%d d_ret=%f expected=%f\n", i, d_ret, expected);
  }
}

// CHECK-CORRECT: i=0 d_ret=1.000000 expected=1.000000
// CHECK-CORRECT-NEXT: i=1 d_ret=0.540302 expected=0.540302
// CHECK-CORRECT-NEXT: i=2 d_ret=-0.416147 expected=-0.416147
// CHECK-CORRECT-NEXT: i=3 d_ret=-0.653644 expected=-0.653644
// CHECK-CORRECT-NEXT: i=4 d_ret=0.960170 expected=0.960170
//
// CHECK-INACTIVE-INCORRECT: i=0 d_ret=0.000000 expected=1.000000
// CHECK-INACTIVE-INCORRECT-NEXT: i=1 d_ret=0.000000 expected=0.540302
// CHECK-INACTIVE-INCORRECT-NEXT: i=2 d_ret=0.000000 expected=-0.416147
// CHECK-INACTIVE-INCORRECT-NEXT: i=3 d_ret=0.000000 expected=-0.653644
// CHECK-INACTIVE-INCORRECT-NEXT: i=4 d_ret=0.000000 expected=0.960170
//
// CHECK-PRINT: [activity] loaded file forced instruction to be inactive: mysin
// CHECK-PRINT: [activity] loaded file forced instruction to be inactive: mycos
