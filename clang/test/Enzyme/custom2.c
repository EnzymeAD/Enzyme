// RUN: %clang_cc1 -emit-llvm %s -O3 -o - | opt -lower-autodiff -O3 -S | FileCheck %s

double gradient_add2(double x, double differet) {
    return differet;
}

double augment_add2(double x) {
    double add = x + 2;
    return add;
}

//__attribute__((noinline))
__attribute__((enzyme("augment", augment_add2)))
__attribute__((enzyme("gradient", gradient_add2)))
double add2(double x);


//{
//    return 2 + x;
//}

double add4(double x) {
  return add2(x) + 2;
}


double test_derivative(double x) {
  return __builtin_autodiff(add4, x);
}

// CHECK: define double @test_derivative(double %x)
// CHECK-NEXT: entry:
// CHECK-NEXT:  ret double 1.000000e+00
// CHECK-NEXT: }
