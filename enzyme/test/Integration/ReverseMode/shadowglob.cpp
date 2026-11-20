// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++14 -fsyntax-only -DTEST_ERRORS %s %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++14 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++14 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++14 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++14 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++14 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++14 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 | %lli - ; fi

#ifdef TEST_ERRORS

double valid_shadow = 0.0;
__attribute__((enzyme_shadow)) double missing_arg = 1.0; // expected-error {{'enzyme_shadow' attribute requires the global variable holding the shadow as its argument}}
int wrong_type_shadow = 0;
__attribute__((enzyme_shadow(wrong_type_shadow))) double wrong_type = 1.0; // expected-error {{the shadow given to 'enzyme_shadow' must have the type of the global, or be an array of that type}}
__attribute__((enzyme_shadow(valid_shadow))) void on_func(); // expected-warning {{'enzyme_shadow' attribute only applies to global variables}}
__attribute__((enzyme_shadow(on_func))) double non_global_arg = 1.0; // expected-error {{the argument of 'enzyme_shadow' must be a global variable}}

void local_scope() {
  double local_shadow = 0.0;
  __attribute__((enzyme_shadow(valid_shadow))) double local_var = 1.0; // expected-warning {{'enzyme_shadow' attribute only applies to global variables}}
  (void)local_shadow;
  (void)local_var;
}

template <typename T>
__attribute__((enzyme_shadow(valid_shadow))) T templated_global = 0; // expected-error {{'enzyme_shadow' does not support templated globals}}

#else

#include "../test_utils.h"

struct Double2 {
  double v[2];
};

int enzyme_width;
double __enzyme_autodiff(void *, ...);
double __enzyme_fwddiff(void *, ...);
Double2 __enzyme_fwddiff2(void *, int, int, double, double, double)
    asm("__enzyme_fwddiff");

extern double scale_shadow;
extern __attribute__((enzyme_shadow(scale_shadow))) double scale;
double scale_shadow = 0.0;
__attribute__((enzyme_shadow(scale_shadow))) double scale = 2.0;

struct Params {
  static double offset_shadow;
  __attribute__((enzyme_shadow(offset_shadow))) static double offset;
};
double Params::offset_shadow = 0.0;
double Params::offset = 1.0;

double vec_scale_shadow[2] = {0.0, 0.0};
__attribute__((enzyme_shadow(vec_scale_shadow))) double vec_scale = 3.0;

__attribute__((noinline)) double eval_polynomial(double x) {
  return scale * x * x + Params::offset * x;
}

__attribute__((noinline)) double eval_vec_scale(double x) {
  return vec_scale * x;
}

int main() {
  double dx = __enzyme_autodiff(reinterpret_cast<void *>(eval_polynomial), 3.0);
  APPROX_EQ(dx, 2 * 2.0 * 3.0 + 1.0, 1e-10);
  APPROX_EQ(scale_shadow, 3.0 * 3.0, 1e-10);
  APPROX_EQ(Params::offset_shadow, 3.0, 1e-10);
  APPROX_EQ(scale, 2.0, 1e-10);

  scale_shadow = 1.0;
  Params::offset_shadow = 0.0;
  double tangent =
      __enzyme_fwddiff(reinterpret_cast<void *>(eval_polynomial), 3.0, 0.0);
  APPROX_EQ(tangent, 9.0, 1e-10);

  vec_scale_shadow[0] = 2.0;
  vec_scale_shadow[1] = 5.0;
  Double2 vec_tangent = __enzyme_fwddiff2(
      reinterpret_cast<void *>(eval_vec_scale), enzyme_width, 2, 4.0, 0.0, 0.0);
  APPROX_EQ(vec_tangent.v[0], 8.0, 1e-10);
  APPROX_EQ(vec_tangent.v[1], 20.0, 1e-10);
  return 0;
}

#endif
