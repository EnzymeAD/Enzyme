// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

#include "../test_utils.h"

#include <enzyme/enzyme>

struct PolyFunctor {
  double a;
  double b;

  double operator()(double x) const { return a * x * x + b * x; }
};

int main() {
  PolyFunctor fn{3.0, 2.0};
  PolyFunctor dfn{0.0, 0.0};

  // Reverse mode differentiating both functor captures and Active scalar arg
  auto d_x_tup = enzyme::autodiff<enzyme::Reverse>(
      enzyme::Duplicated<PolyFunctor &>{fn, dfn}, enzyme::Active<double>{4.0});
  double dx = enzyme::get<0>(enzyme::get<0>(d_x_tup));
  APPROX_EQ(dx, 2.0 * 3.0 * 4.0 + 2.0, 1e-10);
  APPROX_EQ(dfn.a, 16.0, 1e-10);
  APPROX_EQ(dfn.b, 4.0, 1e-10);

  // Capturing lambda passed via Duplicated<Lambda>
  double buf[3] = {1.0, 2.0, 3.0};
  double dbuf[3] = {0.0, 0.0, 0.0};
  auto make_lam = [](double *p) {
    return [p](int n) {
      double s = 0.0;
      for (int i = 0; i < n; ++i)
        s += 3.0 * p[i] * p[i];
      return s;
    };
  };
  auto lam = make_lam(buf);
  auto dlam = make_lam(dbuf);
  enzyme::autodiff<enzyme::Reverse>(
      enzyme::Duplicated<decltype(lam)>{lam, dlam}, enzyme::Const<int>{3});
  for (int i = 0; i < 3; ++i)
    APPROX_EQ(dbuf[i], 6.0 * buf[i], 1e-10);

  // Const-wrapped callable
  auto cx_tup = enzyme::autodiff<enzyme::Reverse>(
      enzyme::Const<const PolyFunctor &>{fn}, enzyme::Active<double>{5.0});
  APPROX_EQ(enzyme::get<0>(enzyme::get<0>(cx_tup)), 2.0 * 3.0 * 5.0 + 2.0,
            1e-10);
  return 0;
}
