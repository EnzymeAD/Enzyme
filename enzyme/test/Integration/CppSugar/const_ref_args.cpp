// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

#include "../test_utils.h"

#include <enzyme/enzyme>
#include <vector>

struct Pair {
  double a;
  double b;
};

double square_cref(const double &x) { return x * x; }
double square_cref_noexcept(const double &x) noexcept { return x * x; }
double mul_crefs(const double &x, const double &y) { return x * y; }
double mul_val_cref(double x, const double &y) { return x * y; }
double accum_mixed(const double &x, double &y) {
  y += x;
  return x * y;
}
double square_cptr(const double *x) { return (*x) * (*x); }
double pair_dot(const Pair &p, const Pair &q) { return p.a * q.a + p.b * q.b; }
double sum_vec(const std::vector<double> &v) {
  double s = 0.0;
  for (double x : v)
    s += x;
  return s;
}
double nullary_fn() { return 42.0; }

int main() {
  // Non-const Duplicated<double&> / DuplicatedNoNeed<double&> for const double&
  {
    double x = 3.0, dx = 0.0;
    enzyme::autodiff<enzyme::Reverse>(square_cref,
                                      enzyme::Duplicated<double &>{x, dx});
    APPROX_EQ(dx, 6.0, 1e-10);

    dx = 0.0;
    auto rev_prim = enzyme::autodiff<enzyme::ReverseWithPrimal>(
        square_cref, enzyme::DuplicatedNoNeed<double &>{x, dx});
    APPROX_EQ(dx, 6.0, 1e-10);
    APPROX_EQ(enzyme::get<1>(rev_prim), 9.0, 1e-10);

    double (*fp)(const double &) = square_cref;
    dx = 0.0;
    enzyme::autodiff<enzyme::Reverse>(fp, enzyme::Duplicated<double &>{x, dx});
    APPROX_EQ(dx, 6.0, 1e-10);

    dx = 0.0;
    enzyme::autodiff<enzyme::Reverse>(square_cref_noexcept,
                                      enzyme::Duplicated<double &>{x, dx});
    APPROX_EQ(dx, 6.0, 1e-10);

    double vx = 1.0;
    double fwd = enzyme::get<0>(enzyme::autodiff<enzyme::Forward>(
        square_cref, enzyme::Duplicated<double &>{x, vx}));
    APPROX_EQ(fwd, 6.0, 1e-10);
  }

  // Non-const Const<double&> and mixed const/non-const reference wrappers
  {
    double x = 4.0, y = 5.0;
    auto g = enzyme::autodiff<enzyme::Reverse>(
        mul_val_cref, enzyme::Active<double>{x}, enzyme::Const<double &>{y});
    APPROX_EQ(enzyme::get<0>(enzyme::get<0>(g)), 5.0, 1e-10);

    const double cx = 4.0;
    double dx = 0.0, dy = 0.0;
    enzyme::autodiff<enzyme::Reverse>(
        mul_crefs, enzyme::Duplicated<const double &>{cx, dx},
        enzyme::Duplicated<double &>{y, dy});
    APPROX_EQ(dx, 5.0, 1e-10);
    APPROX_EQ(dy, 4.0, 1e-10);

    double xm = 2.0, ym = 3.0, dxm = 0.0, dym = 0.0;
    enzyme::autodiff<enzyme::Reverse>(
        accum_mixed, enzyme::Duplicated<double &>{xm, dxm},
        enzyme::Duplicated<double &>{ym, dym});
    // f(x, y) = x * (y + x) = x*y + x^2 => df/dx = y + 2x = 7, df/dy = x = 2
    APPROX_EQ(dxm, 7.0, 1e-10);
    APPROX_EQ(dym, 2.0, 1e-10);
  }

  // Non-const pointer wrappers (explicit and CTAD) for const T* parameter
  {
    double x = 5.0, dx = 0.0;
    enzyme::autodiff<enzyme::Reverse>(square_cptr,
                                      enzyme::Duplicated<double *>{&x, &dx});
    APPROX_EQ(dx, 10.0, 1e-10);

    dx = 0.0;
    enzyme::autodiff<enzyme::Reverse>(square_cptr, enzyme::Duplicated{&x, &dx});
    APPROX_EQ(dx, 10.0, 1e-10);

    const double cx = 5.0;
    dx = 0.0;
    enzyme::autodiff<enzyme::Reverse>(square_cptr,
                                      enzyme::Duplicated{&cx, &dx});
    APPROX_EQ(dx, 10.0, 1e-10);

    dx = 0.0;
    enzyme::autodiff<enzyme::Reverse>(square_cptr,
                                      enzyme::DuplicatedNoNeed{&cx, &dx});
    APPROX_EQ(dx, 10.0, 1e-10);

    auto c_res = enzyme::autodiff<enzyme::ReverseWithPrimal>(
        square_cptr, enzyme::Const{&x});
    APPROX_EQ(enzyme::get<1>(c_res), 25.0, 1e-10);
  }

  // Struct and std::vector const references with non-const wrappers
  {
    Pair p{2.0, 3.0}, dp{0.0, 0.0}, q{7.0, 11.0};
    enzyme::autodiff<enzyme::Reverse>(
        pair_dot, enzyme::Duplicated<Pair &>{p, dp}, enzyme::Const<Pair &>{q});
    APPROX_EQ(dp.a, 7.0, 1e-10);
    APPROX_EQ(dp.b, 11.0, 1e-10);

    std::vector<double> v{1.0, 2.0, 3.0}, dv(3, 0.0);
    enzyme::autodiff<enzyme::Reverse>(
        sum_vec, enzyme::Duplicated<std::vector<double> &>{v, dv});
    for (double val : dv)
      APPROX_EQ(val, 1.0, 1e-10);
  }

  // const-qualified wrapper lvalues (verifies enzyme::remove_cvref)
  {
    double x = 3.0, y = 7.0;
    const enzyme::Active<double> ax{x};
    const enzyme::Const<double &> cy{y};
    auto res = enzyme::autodiff<enzyme::Reverse>(mul_val_cref, ax, cy);
    APPROX_EQ(enzyme::get<0>(enzyme::get<0>(res)), 7.0, 1e-10);
  }

  // primal_call with reference and const-reference wrappers
  {
    const double cx = 6.0;
    double x = 2.0, y = 3.0, dx = 0.0, dy = 0.0;
    APPROX_EQ(enzyme::primal_call(square_cref,
                                  enzyme::Duplicated<const double &>{cx, dx}),
              36.0, 1e-10);
    APPROX_EQ(enzyme::primal_call(square_cref, enzyme::Const<const double &>{cx}),
              36.0, 1e-10);
    APPROX_EQ(enzyme::primal_call(accum_mixed,
                                  enzyme::Duplicated<double &>{x, dx},
                                  enzyme::Duplicated<double &>{y, dy}),
              10.0, 1e-10);
    APPROX_EQ(y, 5.0, 1e-10);
    APPROX_EQ(enzyme::primal_call(nullary_fn), 42.0, 1e-10);
  }

  return 0;
}
