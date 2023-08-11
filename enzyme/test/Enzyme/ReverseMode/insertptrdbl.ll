; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg)" -S | FileCheck %s

declare double @__enzyme_autodiff(double (double)*, ...) 

define { double*, double } @sub(double* %x, double %y) {
entry:
  %i1 = insertvalue { double*, double } undef, double* %x, 0
  %i2 = insertvalue { double*, double } %i1, double %y, 1
  ret { double*, double } %i2
}

define double @square(double %x) {
entry:
  %m = alloca double, align 8
  store double %x, double* %m, align 8
  %v = call { double*, double } @sub(double* %m, double %x)
  %e = extractvalue { double*, double } %v, 0
  %ld = load double, double* %e
  ret double %ld
}

define double @dsquare(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @square, double %x)
  ret double %0
}

; CHECK: define internal { double*, double } @augmented_sub(double* %x, double* %"x'", double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"i1'ipiv" = insertvalue { double*, double } zeroinitializer, double* %"x'", 0
; CHECK-NEXT:   %"i2'ipiv" = insertvalue { double*, double } %"i1'ipiv", double 0.000000e+00, 1
; CHECK-NEXT:   ret { double*, double } %"i2'ipiv"
; CHECK-NEXT: }

; CHECK: define internal { double } @diffesub(double* %x, double* %"x'", double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret { double } zeroinitializer
; CHECK-NEXT: }
