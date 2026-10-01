// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

// The shadows of globals through the C++ interface: enzyme::context makes a
// set of shadows, enzyme::shadow returns the shadow of a global in it, and
// enzyme::autodiff given the context differentiates with those shadows.

#include "../test_utils.h"

#include <enzyme/enzyme>

double scale = 2.0;

struct Params {
  static double offset;
};
double Params::offset = 1.0;

double f(double x) { return scale * x * x + Params::offset * x; }

int main() {
  auto ctx = enzyme::context();
  double *dscale = enzyme::shadow(ctx, scale);
  double *doffset = enzyme::shadow(ctx, Params::offset);

  // Reverse mode accumulates the adjoints of the globals in the context.
  *dscale = 0.0;
  *doffset = 0.0;
  auto res = enzyme::autodiff<enzyme::Reverse>(ctx, f, enzyme::Active(3.0));
  APPROX_EQ(enzyme::get<0>(enzyme::get<0>(res)), 2 * 2.0 * 3.0 + 1.0, 1e-10);
  APPROX_EQ(*dscale, 3.0 * 3.0, 1e-10);
  APPROX_EQ(*doffset, 3.0, 1e-10);

  // Forward mode reads the tangents seeded in the context.
  *dscale = 1.0;
  *doffset = 0.0;
  auto t = enzyme::autodiff<enzyme::Forward>(
      ctx, f, enzyme::Duplicated<double>(3.0, 0.0));
  APPROX_EQ(enzyme::get<0>(t), 9.0, 1e-10);

  // Another context has shadows of its own: its seeds are zero, and its
  // adjoints do not reach the first context.
  auto other = enzyme::context();
  auto res2 =
      enzyme::autodiff<enzyme::Reverse>(other, f, enzyme::Active(3.0));
  APPROX_EQ(enzyme::get<0>(enzyme::get<0>(res2)), 13.0, 1e-10);
  APPROX_EQ(*enzyme::shadow(other, scale), 9.0, 1e-10);
  APPROX_EQ(*dscale, 1.0, 1e-10);
  return 0;
}
