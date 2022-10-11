; RUN:  %opt < %s %newLoadEnzyme -passes="enzyme,inline,mem2reg,instcombine,early-cse,adce"  -enzyme-preopt=false -S | FileCheck %s

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

; CHECK: define dso_local double @dsqrelu(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp.i = fcmp fast ogt double %x, 0.000000e+00
; CHECK-NEXT:   br label %cond.end.i

; CHECK: cond.true.i:                                      ; No predecessors!
; CHECK-NEXT:  br label %cond.end.i

; CHECK: cond.end.i:                                       ; preds = %entry, %cond.true.i
; CHECK-NEXT:  br i1 %cmp.i, label %invertcond.true.i, label %diffesqrelu.exit

; CHECK: invertcond.true.i:                                ; preds = %cond.end.i
; CHECK-NEXT:  %0 = call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:  %mul_unwrap.i = fmul fast double %0, %x
; CHECK-NEXT:  %1 = call fast double @llvm.sqrt.f64(double %mul_unwrap.i)
; CHECK-NEXT:  %2 = fdiv fast double 5.000000e-01, %1
; CHECK-NEXT:  %3 = fcmp fast oeq double %mul_unwrap.i, 0.000000e+00
; CHECK-NEXT:  %4 = select fast i1 %3, double 0.000000e+00, double %2
; CHECK-NEXT:  %m0diffe.i = fmul fast double %4, %x
; CHECK-NEXT:  %m1diffex.i = fmul fast double %4, %0
; CHECK-NEXT:  %5 = call fast double @llvm.cos.f64(double %x)
; CHECK-NEXT:  %6 = fmul fast double %m0diffe.i, %5
; CHECK-NEXT:  %7 = fadd fast double %m1diffex.i, %6
; CHECK-NEXT:  br label %diffesqrelu.exit

; CHECK: diffesqrelu.exit:                                 ; preds = %invertcond.true.i, %cond.end.i
; CHECK-NEXT:  %"x'de.i.0" = phi double [ %7, %invertcond.true.i ], [ 0.000000e+00, %cond.end.i ]
; CHECK-NEXT:  ret double %"x'de.i.0"
; CHECK-NEXT:}