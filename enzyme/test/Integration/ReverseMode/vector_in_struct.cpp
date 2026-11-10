// clang-format off
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math %O0TBAA %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// clang-format on

#include <stdio.h>
#include <vector>

#include "../test_utils.h"

struct VectorState {
  double velocity = 0.0;
  double density = 0.0;
  double pressure = 0.0;
  std::vector<double> history;
};

template <typename T>
void ComputeWithVector(const T &v, const T &rho, VectorState &state) {
  state.velocity = v;
  state.density = rho;
  state.pressure = 0.5 * rho * v * v;
  state.history.clear();
  state.history.push_back(v);
  state.history.push_back(state.pressure);
}

template void ComputeWithVector<double>(const double &, const double &,
                                        VectorState &);

extern "C" {
extern int enzyme_dup;
extern int enzyme_const;
extern void __enzyme_autodiff(void *, ...);
}

void adjoint_vector(const double *pv, double *d_pv, const double *prho,
                    VectorState *pState, VectorState *d_pState) {
  __enzyme_autodiff((void *)ComputeWithVector<double>, enzyme_dup, (double *)pv,
                    d_pv, enzyme_const, prho, enzyme_dup, pState, d_pState);
}

int main() {
  double v = 25.0, rho = 1.225;
  double d_v = 0.0;
  VectorState v_state = {};
  VectorState v_d_state = {};
  v_d_state.pressure = 1.0;
  adjoint_vector(&v, &d_v, &rho, &v_state, &v_d_state);
  printf("d_v = %.6f\n", d_v);
  APPROX_EQ(d_v, 30.625, 1e-7);
  return 0;
}
