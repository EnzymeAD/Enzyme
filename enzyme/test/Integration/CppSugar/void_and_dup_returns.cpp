// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

#include "../test_utils.h"

#include <enzyme/enzyme>

struct Pair {
  double a;
  double b;
};

void scale_pair_into(const Pair &in, Pair &out, double scale) {
  out.a = scale * in.a * in.a;
  out.b = scale * in.b;
}

void scale_pair_two_active(const Pair &in, Pair &out, double alpha,
                           double beta) {
  out.a = alpha * in.a * in.a;
  out.b = beta * in.b;
}

void copy_scaled_pair_into(const Pair &in, Pair &out) {
  out.a = in.a * in.a;
  out.b = 3.0 * in.b;
}

double mixed_dup_and_active(const double *x, double alpha) {
  return alpha * x[0] * x[0] + alpha * x[1];
}

int main() {
  // Reverse mode on void-returning function mixing Duplicated<Struct&> and Active<double>
  Pair in{2.0, 3.0}, din{0.0, 0.0};
  Pair out{0.0, 0.0}, dout{1.0, 1.0};
  auto rev = enzyme::autodiff<enzyme::Reverse>(
      scale_pair_into, enzyme::Duplicated<const Pair &>{in, din},
      enzyme::DuplicatedNoNeed<Pair &>{out, dout}, enzyme::Active<double>{4.0});
  double dscale = enzyme::get<0>(enzyme::get<0>(rev));
  APPROX_EQ(dscale, 4.0 + 3.0, 1e-10);
  APPROX_EQ(din.a, 16.0, 1e-10);
  APPROX_EQ(din.b, 4.0, 1e-10);

  // Forward mode on void-returning function (Const<void> default return activity)
  Pair fin{2.0, 3.0}, dfin{0.5, 1.0};
  Pair fout{0.0, 0.0}, dfout{0.0, 0.0};
  enzyme::autodiff<enzyme::Forward>(
      scale_pair_into, enzyme::Duplicated<const Pair &>{fin, dfin},
      enzyme::Duplicated<Pair &>{fout, dfout},
      enzyme::Duplicated<double>{4.0, 1.0});
  APPROX_EQ(fout.a, 16.0, 1e-10);
  APPROX_EQ(dfout.a, 4.0 + 4.0 * 2.0 * 2.0 * 0.5, 1e-10);
  APPROX_EQ(dfout.b, 3.0 + 4.0 * 1.0, 1e-10);

  // ReverseWithPrimal with Duplicated<const double*> preceding Active<double>:
  // grad_out_type filters out Duplicated/Const slots so only Active gradients
  // appear in the returned gradient tuple.
  double x[2] = {3.0, 5.0}, dx[2] = {0.0, 0.0};
  auto rp = enzyme::autodiff<enzyme::ReverseWithPrimal>(
      mixed_dup_and_active, enzyme::Duplicated<const double *>{x, dx},
      enzyme::Active<double>{2.0});
  APPROX_EQ(enzyme::get<1>(rp), 2.0 * (9.0 + 5.0), 1e-10);
  APPROX_EQ(enzyme::get<0>(enzyme::get<0>(rp)), 14.0, 1e-10);
  APPROX_EQ(dx[0], 12.0, 1e-10);
  APPROX_EQ(dx[1], 2.0, 1e-10);

  // ReverseWithPrimal on void-returning function with only Duplicated args
  {
    Pair vin{2.0, 3.0}, dvin{0.0, 0.0};
    Pair vout{0.0, 0.0}, dvout{1.0, 1.0};
    enzyme::autodiff<enzyme::ReverseWithPrimal>(
        copy_scaled_pair_into, enzyme::Duplicated<const Pair &>{vin, dvin},
        enzyme::DuplicatedNoNeed<Pair &>{vout, dvout});
    APPROX_EQ(dvin.a, 4.0, 1e-10);
    APPROX_EQ(dvin.b, 3.0, 1e-10);
  }

  // ReverseWithPrimal on void-returning function with 1 Active arg
  {
    Pair vin{2.0, 3.0}, dvin{0.0, 0.0};
    Pair vout{0.0, 0.0}, dvout{1.0, 1.0};
    auto rwp1 = enzyme::autodiff<enzyme::ReverseWithPrimal>(
        scale_pair_into, enzyme::Duplicated<const Pair &>{vin, dvin},
        enzyme::DuplicatedNoNeed<Pair &>{vout, dvout},
        enzyme::Active<double>{4.0});
    APPROX_EQ(enzyme::get<0>(enzyme::get<0>(rwp1)), 4.0 + 3.0, 1e-10);
    APPROX_EQ(dvin.a, 16.0, 1e-10);
    APPROX_EQ(dvin.b, 4.0, 1e-10);
  }

  // ReverseWithPrimal on void-returning function with 2 Active args
  // (verifies return tuple omits nodiff and matches sret ABI layout)
  {
    Pair vin{2.0, 3.0}, dvin{0.0, 0.0};
    Pair vout{0.0, 0.0}, dvout{1.0, 2.0};
    auto rwp2 = enzyme::autodiff<enzyme::ReverseWithPrimal>(
        scale_pair_two_active, enzyme::Duplicated<const Pair &>{vin, dvin},
        enzyme::DuplicatedNoNeed<Pair &>{vout, dvout},
        enzyme::Active<double>{4.0}, enzyme::Active<double>{5.0});
    APPROX_EQ(enzyme::get<0>(enzyme::get<0>(rwp2)), 4.0, 1e-10);
    APPROX_EQ(enzyme::get<1>(enzyme::get<0>(rwp2)), 6.0, 1e-10);
    APPROX_EQ(dvin.a, 16.0, 1e-10);
    APPROX_EQ(dvin.b, 10.0, 1e-10);
  }
  return 0;
}
