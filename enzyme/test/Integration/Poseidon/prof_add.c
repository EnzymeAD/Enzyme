// RUN: %clang -O0 %s -S -emit-llvm -o %t.ll
// RUN: %opt %t.ll %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,simplifycfg)" -enzyme-preopt=false -fpprofile-generate -S -o %t.opt.ll
// RUN: %clang -O0 %t.opt.ll -c -o %t.o
// RUN: %clang++ %t.o %FPProfileLib -lstdc++ -lm -o %t.exe
// RUN: rm -rf %t.profiles && ENZYME_FPPROFILE_DIR=%t.profiles %t.exe
// RUN: cat %t.profiles/preprocess_tester.fpprofile | FileCheck %s

#include <stdio.h>

extern double __enzyme_fp_optimize(void *, ...);

double tester(double x, double y) {
  return x + y;
}

int main() {
  double res = __enzyme_fp_optimize((void *)tester, 3.0, 4.0);
  printf("result = %f\n", res);

  res = __enzyme_fp_optimize((void *)tester, 1.0, 2.0);
  printf("result = %f\n", res);

  return 0;
}

// CHECK: MinRes = 3.{{[0-9e+]+}}
// CHECK: MaxRes = 7.{{[0-9e+]+}}
// CHECK: Exec = 2
// CHECK: NumOperands = 2
// CHECK: Operand[0] = [1.{{[0-9e+]+}}, 3.{{[0-9e+]+}}]
// CHECK: Operand[1] = [2.{{[0-9e+]+}}, 4.{{[0-9e+]+}}]
