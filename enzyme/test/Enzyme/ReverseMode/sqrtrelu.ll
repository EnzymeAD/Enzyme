; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; #include <math.h>
; 
; double sqrelu(double x) {
;     return (x > 0) ? sqrt(x * sin(x)) : 0;
; }
; 
; double dsqrelu(double x) {
;     return __builtin_autodiff(sqrelu, x);
; }

; Function Attrs: nounwind readnone uwtable
define dso_local double @sqrelu(double %x) #0 {
entry:
  %cmp = fcmp fast ogt double %x, 0.000000e+00
  br i1 %cmp, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  %0 = tail call fast double @llvm.sin.f64(double %x)
  %mul = fmul fast double %0, %x
  %1 = tail call fast double @llvm.sqrt.f64(double %mul)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi double [ %1, %cond.true ], [ 0.000000e+00, %entry ]
  ret double %cond
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double) #1

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double) #1

; Function Attrs: nounwind uwtable
define dso_local double @dsqrelu(double %x) local_unnamed_addr #2 {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @sqrelu, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...) #3

attributes #0 = { nounwind readnone uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind uwtable }
attributes #3 = { nounwind }


; CHECK: define internal { double } @diffesqrelu(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = fcmp fast ogt double %x, 0.000000e+00
; CHECK-NEXT:   %0 = select{{( fast)?}} i1 %cmp, double %differeturn, double 0.000000e+00
; CHECK-NEXT:   br i1 %cmp, label %invertcond.true, label %invertentry

; CHECK: invertentry: 
; CHECK-NEXT:   %"x'de.0" = phi double [ %[[res:.+]], %invertcond.true ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   %[[rv:.+]] = insertvalue { double } {{(undef|poison)}}, double %"x'de.0", 0
; CHECK-NEXT:   ret { double } %[[rv]]

; CHECK: invertcond.true:
; CHECK-NEXT:   %[[dsin:.+]] = tail call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:   %[[mul:.+]] = fmul fast double %[[dsin]], %x
; CHECK-NEXT:   %[[sqrtzero:.+]] = fcmp fast ueq double %[[mul]], 0.000000e+00
; CHECK-NEXT:   %[[sqrt:.+]] = call fast double @llvm.sqrt.f64(double %[[mul]])
; CHECK-NEXT:   %[[tsq:.+]] = fmul fast double 2.000000e+00, %[[sqrt]]
; CHECK-NEXT:   %[[div:.+]] = fdiv fast double %0, %[[tsq]]
; CHECK-NEXT:   %[[dsqrt:.+]] = select{{( fast)?}} i1 %[[sqrtzero]], double 0.000000e+00, double %[[div]]
; CHECK-NEXT:   %[[dmul0:.+]] = fmul fast double %[[dsqrt]], %x
; CHECK-NEXT:   %[[dmul1:.+]] = fmul fast double %[[dsqrt]], %[[dsin]]
; CHECK-NEXT:   %[[dcos:.+]] = call fast double @llvm.cos.f64(double %x)
; CHECK-NEXT:   %[[fmul:.+]] = fmul fast double %[[dmul0]], %[[dcos]]
; CHECK-NEXT:   %[[res]] = fadd fast double %[[dmul1]], %[[fmul]]
; CHECK-NEXT:   br label %invertentry
