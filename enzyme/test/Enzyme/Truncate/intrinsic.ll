; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

declare double @llvm.pow.f64(double %Val, double %Power)
declare double @llvm.powi.f64.i16(double %Val, i16 %power)
declare void @llvm.nvvm.barrier0()

define double @f(double %x, double %y) {
  %res1 = call double @llvm.pow.f64(double %x, double %y)
  %res2 = call double @llvm.powi.f64.i16(double %x, i16 2)
  %res = fadd double %res1, %res2
  call void @llvm.nvvm.barrier0()
  ret double %res
}

declare double (double, double)* @__enzyme_truncate(...)

define double @tester(double %x, double %y) {
entry:
  %ptr = call double (double, double)* (...) @__enzyme_truncate(double (double, double)* @f, i64 64, i64 32)
  %res = call double %ptr(double %x, double %y)
  ret double %res
}
