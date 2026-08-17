; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Same as gsl_sf_legendre_array_e.ll, but reached the way Julia reaches a
; ccall: through a lazily bound declaration named "ejlstr$<function>$<library>"
; that carries the real entry point in enzyme_math. The two GSL entry points
; the derivative needs have to be declared under the same convention, or they
; will not resolve when the module is JIT linked.

declare dso_local i32 @"ejlstr$gsl_sf_legendre_array_e$libgsl.so"(i32, i32, double, double, double*) local_unnamed_addr "enzyme_math"="gsl_sf_legendre_array_e"

define dso_local void @tester(i32 %a0, i32 %a1, double %x, double %a3, double* %a4) {
entry:
  %c = call i32 @"ejlstr$gsl_sf_legendre_array_e$libgsl.so"(i32 %a0, i32 %a1, double %x, double %a3, double* %a4)
  ret void
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (...) @__enzyme_autodiff(void (i32, i32, double, double, double*)* @tester, i32 0, i32 10, double %x, metadata !"enzyme_const", double %y, double* null, double* null)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffetester(i32 %a0, i32 %a1, double %x, double %a3, double* %a4, double* %"a4'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %c = call i32 @"ejlstr$gsl_sf_legendre_array_e$libgsl.so"(i32 %a0, i32 %a1, double %x, double %a3, double* %a4)
; CHECK-NEXT:   %[[as:.+]] = call i32 @"ejlstr$gsl_sf_legendre_array_n$libgsl.so"(i32 %a1)
; CHECK: call i32 @"ejlstr$gsl_sf_legendre_deriv_array_e$libgsl.so"(i32 %a0, i32 %a1, double %x, double %a3, double* %{{.+}}, double* %{{.+}})
