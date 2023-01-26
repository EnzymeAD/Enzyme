// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-vectorize-at-leaf-nodes -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-vectorize-at-leaf-nodes -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-vectorize-at-leaf-nodes -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-vectorize-at-leaf-nodes -S | %lli -
// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-vectorize-at-leaf-nodes -enzyme-inline=1 -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-vectorize-at-leaf-nodes -enzyme-inline=1 -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-vectorize-at-leaf-nodes -enzyme-inline=1 -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-vectorize-at-leaf-nodes -enzyme-inline=1 -S | %lli -

#include <stdio.h>
#include <vector>
#include <array>
#include <stddef.h>
#include <algorithm>
#include <random>

using namespace std;

extern int enzyme_width;
extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_const;

// template <typename T, size_t N>
// using VT = T __attribute__((ext_vector_type(N)));

extern void __enzyme_fwddiff(void (*)(double const *, double *, size_t),
                             /*enzyme_width*/ int, size_t,
                             /*enzyme_dup*/ int, double const *, double*,
                             /*enzyme_dup*/ int, double*, double*,
                             /*enzyme_const*/ int, size_t);

void square_sum(double const* in, double *out, size_t size) {
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += in[i] * in[i];
    }
    *out = result;
}

template<size_t N>
void dsquare_sum(double const* in, double *derivatives, size_t size) {
  alignas(64) vector<double> din = vector<double>(N*size, 0.0);
  double out = 0.0;
  alignas(64) vector<double> dout = vector<double>(N, 0.0);

  for (size_t i = 0; i < size; i += N) {
    for (size_t j = 0; j < N; ++j) {
      din[i*N + j * N + j] = 1.0;
    }

    __enzyme_fwddiff(square_sum, 
                     enzyme_width, N, 
                     enzyme_dupnoneed, in, din.data(), 
                     enzyme_dupnoneed, &out, dout.data(),
                     enzyme_const, size);

    for (size_t j = 0; j < N; ++j) {
      derivatives[i + j] = dout[j];
      din[i*N + j * N + j] = 0.0;
    }
  }
}


int main() {
  alignas(64) array<double, 1024> control;
  alignas(64) array<double, 1024> derivatives;

  uniform_real_distribution<double> unif(0.0, 1.0);
  default_random_engine re;

  generate(control.begin(), control.end(), [&]() { return unif(re); });

   dsquare_sum<8>(control.data(), derivatives.data(), control.size());

  for (size_t i = 0; i < control.size(); ++i) {
    printf("[%f : %f]\n", control[i], derivatives[i]);
  }
}

