; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -fpprofile-use=%S/Inputs/expm1div_profiles -fpopt-enable-herbie=true -fpopt-enable-pt=false -fpopt-enable-solver=false -fpopt-cache-path= -fpopt-print -S 2>&1 | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -fpprofile-use=%S/Inputs/expm1div_profiles -fpopt-enable-herbie=true -fpopt-enable-pt=false -fpopt-enable-solver=false -fpopt-cache-path= -fpopt-print -S 2>&1 | FileCheck %s
; REQUIRES: poseidon

; Live Herbie on (exp(x) - 1) / x: verifies expm1 candidate is produced.

declare double @llvm.exp.f64(double)

define double @tester(double %x) {
entry:
  %exp = call fast double @llvm.exp.f64(double %x)
  %sub = fsub fast double %exp, 1.0
  %div = fdiv fast double %sub, %x
  ret double %div
}

define double @test_opt(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fp_optimize(double (double)* nonnull @tester, double %x, metadata !"enzyme_err_tol", double 0.5)
  ret double %0
}

declare double @__enzyme_fp_optimize(double (double)*, ...)

; CHECK: Candidates:
; CHECK: expm1
; CHECK: Finished
; CHECK: define double @preprocess_tester(double %x)
; CHECK: ret double
