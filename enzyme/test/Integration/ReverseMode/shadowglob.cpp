// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 | %lli - ; fi

// Globals whose shadows the program declares with enzyme_shadow: the
// adjoints of the globals accumulate into them in reverse mode, and forward
// mode reads the tangents seeded in them.

#include "../test_utils.h"

double __enzyme_autodiff(void *, ...);
double __enzyme_fwddiff(void *, ...);

double scale_shadow = 0;
__attribute__((enzyme_shadow(scale_shadow))) double scale = 2.0;

struct Params {
  static double offset_shadow;
  __attribute__((enzyme_shadow(offset_shadow))) static double offset;
};
double Params::offset_shadow = 0;
double Params::offset = 1.0;

__attribute__((noinline)) double f(double x) {
  return scale * x * x + Params::offset * x;
}

int main() {
  double dx = __enzyme_autodiff((void *)f, 3.0);
  APPROX_EQ(dx, 2 * 2.0 * 3.0 + 1.0, 1e-10);
  APPROX_EQ(scale_shadow, 3.0 * 3.0, 1e-10);
  APPROX_EQ(Params::offset_shadow, 3.0, 1e-10);
  APPROX_EQ(scale, 2.0, 1e-10);

  scale_shadow = 1.0;
  Params::offset_shadow = 0.0;
  double t = __enzyme_fwddiff((void *)f, 3.0, 0.0);
  APPROX_EQ(t, 9.0, 1e-10);
  return 0;
}
