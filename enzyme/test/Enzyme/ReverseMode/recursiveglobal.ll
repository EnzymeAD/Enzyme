; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; A constant global whose initializer refers back to itself, as emitted for a
; relative-pointer jump table when lowering a switch over several constant
; tables. Type analysis must not recurse infinitely analyzing the initializer.

@tab = constant [1 x i64] [i64 sub (i64 0, i64 ptrtoint ([1 x i64]* @tab to i64))]

define double @f(double %x) {
  %v = load double, double* bitcast ([1 x i64]* @tab to double*), align 8
  ret double %v
}

define double @d(double %x) {
  %r = call double (...) @__enzyme_autodiff(double (double)* @f, double %x)
  ret double %r
}

declare double @__enzyme_autodiff(...)

; The pointer-typed shadow setup in between differs across LLVM versions, so
; only the signature and the (zero) derivative are checked here.
; CHECK: define internal { double } @diffef(double %x, double %differeturn)
; CHECK-NEXT: invert:
; CHECK: ret { double } zeroinitializer
