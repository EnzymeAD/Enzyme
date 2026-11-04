// clang-format off
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// clang-format on

// Without an explicit return activity, enzyme::autodiff<enzyme::Forward> must
// return the tangent of an aggregate-valued function, exactly as it does for a
// scalar one, rather than the primal value.

#include "../test_utils.h"

#include <array>
#include <cmath>
#include <enzyme/enzyme>

struct pair {
  double x;
  double y;
};

std::array<double, 2> trig(const double &x) {
  return std::array<double, 2>{std::cos(x), 2 * std::sin(x)};
}

pair scaled(double x) { return pair{3 * x, x * x}; }

double scalar(double x) { return x * x; }

int unchanged(double x) { return (int)x; }

int main() {
  const double x = 1.0;
  const double dx = 1.0;

  {
    auto res = enzyme::autodiff<enzyme::Forward>(
        trig, enzyme::Duplicated<const double &>{x, dx});
    std::array<double, 2> dfdx = enzyme::get<0>(res);
    printf("dfdx = %f %f\n", dfdx[0], dfdx[1]);
    APPROX_EQ(dfdx[0], -std::sin(1.0), 1e-10);
    APPROX_EQ(dfdx[1], 2 * std::cos(1.0), 1e-10);
  }

  {
    auto res = enzyme::autodiff<enzyme::Forward>(
        scaled, enzyme::Duplicated<double>{2.0, 1.0});
    pair dp = enzyme::get<0>(res);
    printf("dp = %f %f\n", dp.x, dp.y);
    APPROX_EQ(dp.x, 3.0, 1e-10);
    APPROX_EQ(dp.y, 4.0, 1e-10);
  }

  {
    auto res = enzyme::autodiff<enzyme::Forward>(
        scalar, enzyme::Duplicated<double>{3.0, 1.0});
    double d = enzyme::get<0>(res);
    printf("d = %f\n", d);
    APPROX_EQ(d, 6.0, 1e-10);
  }

  {
    auto res = enzyme::autodiff<enzyme::Forward>(
        unchanged, enzyme::Duplicated<double>{3.5, 1.0});
    int v = enzyme::get<0>(res);
    printf("v = %d\n", v);
    TEST_EQ(v, 3);
  }
  return 0;
}
