// RUN: rm -rf %t && mkdir -p %t
// RUN: %clangxx -std=c++17 -O2 -fno-fast-math -ffp-contract=off -I%S/../../Poseidon %S/../../Poseidon/Core/UniformReal.cpp %s -o %t/uniform_real
// RUN: %t/uniform_real | FileCheck %s
// REQUIRES: poseidon

// The first uniform reals of each engine the accuracy models sample with, at
// the default seed, as the standard library of the first build drew them:
// Core/UniformReal.cpp writes that library's mapping out, so a build against
// any other library draws these same numbers.

#include "Core/UniformReal.h"

#include <cstdio>

int main() {
  std::minstd_rand0 minstd(239778888);
  for (int i = 0; i < 3; ++i)
    std::printf("minstd [0,1) %a\n", poseidon::uniformReal(minstd, 0.0, 1.0));
  for (int i = 0; i < 3; ++i)
    std::printf("minstd [-3,2) %a\n",
                poseidon::uniformReal(minstd, -3.0, 2.0));
  std::mt19937_64 mt(239778888);
  for (int i = 0; i < 3; ++i)
    std::printf("mt64 [0,1) %a\n", poseidon::uniformReal(mt, 0.0, 1.0));
  for (int i = 0; i < 3; ++i)
    std::printf("mt64 [-40,1.5) %a\n", poseidon::uniformReal(mt, -40.0, 1.5));
}

// CHECK: minstd [0,1) 0x1.24534b62f5c64p-1
// CHECK-NEXT: minstd [0,1) 0x1.3a3160ac99b9cp-1
// CHECK-NEXT: minstd [0,1) 0x1.b89855e2e06c1p-3
// CHECK-NEXT: minstd [-3,2) 0x1.90cb47a3e2a78p+0
// CHECK-NEXT: minstd [-3,2) -0x1.f2f47ea7a83ccp+0
// CHECK-NEXT: minstd [-3,2) -0x1.3dcb036088d35p+1
// CHECK-NEXT: mt64 [0,1) 0x1.ce0d3960295fdp-3
// CHECK-NEXT: mt64 [0,1) 0x1.8f3c1d7f8e5e8p-1
// CHECK-NEXT: mt64 [0,1) 0x1.4650013460fbap-11
// CHECK-NEXT: mt64 [-40,1.5) -0x1.366e2c6bc088p-1
// CHECK-NEXT: mt64 [-40,1.5) -0x1.1da18b23aa3e8p+4
// CHECK-NEXT: mt64 [-40,1.5) -0x1.da232e70612ap+0
