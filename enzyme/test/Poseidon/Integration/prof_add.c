// RUN: %clang -O0 %s -S -emit-llvm -o %t.ll
// RUN: %opt %t.ll %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -enzyme-preopt=false -poseidon-profile-generate -S -o %t.opt.ll
// RUN: %clang -O0 %t.opt.ll -c -o %t.o
// RUN: %clang++ %t.o %FPProfileLib -lstdc++ -lm -o %t.exe
// RUN: rm -rf %t.profiles && POSEIDON_PROFILE_DIR=%t.profiles %t.exe
// RUN: cat %t.profiles/preprocess_tester.fpprofile | FileCheck %s
// REQUIRES: poseidon, enzyme

#include <stdio.h>

extern double __poseidon_fp_optimize(void *, ...);

double tester(double x, double y) {
  return x + y;
}

int main() {
  double res = __poseidon_fp_optimize((void *)tester, 3.0, 4.0);
  printf("result = %f\n", res);

  res = __poseidon_fp_optimize((void *)tester, 1.0, 2.0);
  printf("result = %f\n", res);

  return 0;
}

// CHECK: MinRes = 3.{{[0-9e+]+}}
// CHECK: MaxRes = 7.{{[0-9e+]+}}
// CHECK: SumValue = 1.00000000000000000e+01
// CHECK: SumSens = 1.00000000000000000e+01
// CHECK: SumGrad = 2.00000000000000000e+00
// CHECK: Exec = 2
// CHECK: NumOperands = 2
// CHECK: Operand[0] = [1.{{[0-9e+]+}}, 3.{{[0-9e+]+}}]
// CHECK: Operand[1] = [2.{{[0-9e+]+}}, 4.{{[0-9e+]+}}]
