// RUN: %clang++ -std=c++11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include "../test_utils.h"

#include <cmath>
#include <string>
#include <unordered_map>

extern double __enzyme_autodiff(void *, double);

std::unordered_map<std::string, double> fill_umap(double x, double y) {
  std::unordered_map<std::string, double> data;

  data["alpha_x"] = x * x + y;
  data["alpha_y"] = y * y + 2.;

  return data;
}

double compute_umap_op(double r) {
  double x = std::sqrt(r);
  double y = std::log(r);

  std::unordered_map<std::string, double> data = fill_umap(x, y);

  return data["alpha_x"];
}

double compute_umap_at(double r) {
  double x = std::sqrt(r);
  double y = std::log(r);

  std::unordered_map<std::string, double> const data = fill_umap(x, y);

  return data.at("alpha_x");
}

double compute_umap_reserve(double r) {
  std::unordered_map<std::string, double> data;
  data.reserve(4);
  data["alpha_x"] = r * r;
  data["alpha_y"] = 2.0;

  return std::log(data.at("alpha_x")) + data.at("alpha_y") * r;
}

int main() {
  double r0 = 3.0;

  // operator[]
  {
    double const alpha_x = compute_umap_op(r0);
    APPROX_EQ(alpha_x, 3.0 + std::log(3.0), 1e-6);

    double ddx = __enzyme_autodiff((void *)compute_umap_op, r0);
    APPROX_EQ(ddx, 1.0 + 1.0 / 3.0, 1e-6);
  }

  // at()
  {
    double const alpha_x = compute_umap_at(r0);
    APPROX_EQ(alpha_x, 3.0 + std::log(3.0), 1e-6);

    double ddx = __enzyme_autodiff((void *)compute_umap_at, r0);
    APPROX_EQ(ddx, 1.0 + 1.0 / 3.0, 1e-6);
  }

  // reserve() + at()
  {
    double const val = compute_umap_reserve(r0);
    APPROX_EQ(val, std::log(9.0) + 6.0, 1e-6);

    double ddx = __enzyme_autodiff((void *)compute_umap_reserve, r0);
    APPROX_EQ(ddx, 2.0 / 3.0 + 2.0, 1e-6);
  }

  return 0;
}
