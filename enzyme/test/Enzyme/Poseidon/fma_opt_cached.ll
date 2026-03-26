; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -fpprofile-use=%S/Inputs/fma_opt_profiles -fpopt-enable-herbie=true -fpopt-enable-pt=false -fpopt-enable-solver=false -fpopt-cache-path=%S/Inputs/fma_opt_cache -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -fpprofile-use=%S/Inputs/fma_opt_profiles -fpopt-enable-herbie=true -fpopt-enable-pt=false -fpopt-enable-solver=false -fpopt-cache-path=%S/Inputs/fma_opt_cache -S | FileCheck %s

; Cached Herbie rewrite: (x + y) * z  -->  fma(x, z, y * z)

define double @tester(double %x, double %y, double %z) {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @test_opt(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__enzyme_fp_optimize(double (double, double, double)* nonnull @tester, double %x, double %y, double %z, metadata !"enzyme_err_tol", double 0.5)
  ret double %0
}

declare double @__enzyme_fp_optimize(double (double, double, double)*, ...)

; CHECK: define double @test_opt(double %x, double %y, double %z)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[call:.+]] = call double @preprocess_tester(double %x, double %y, double %z)
; CHECK-NEXT:   ret double %[[call]]

; CHECK: define double @preprocess_tester(double %x, double %y, double %z)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[mul:.+]] = fmul fast double %z, %y
; CHECK-NEXT:   %[[fma:.+]] = tail call fast double @llvm.fma.f64(double %x, double %z, double %[[mul]])
; CHECK-NEXT:   ret double %[[fma]]
