; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; __attribute__((noinline))
; double f(double x) {
;     return x;
; }
; 
; double relu(double x) {
;     return (x > 0) ? f(x) : 0;
; }
; 
; double drelu(double x) {
;     return __builtin_autodiff(relu, x);
; }

define dso_local double @f(double %x) #1 {
entry:
  ret double %x
}

define dso_local double @relu(double %x) {
entry:
  %cmp = fcmp fast ogt double %x, 0.000000e+00
  br i1 %cmp, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  %call = tail call fast double @f(double %x)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi double [ %call, %cond.true ], [ 0.000000e+00, %entry ]
  ret double %cond
}

define dso_local double @drelu(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @relu, double %x)
  ret double %0
}

declare double @__enzyme_autodiff(double (double)*, ...) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone noinline }

; CHECK: define internal { double } @differelu(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = fcmp fast ogt double %x, 0.000000e+00
; CHECK-NEXT:   %0 = select{{( fast)?}} i1 %cmp, double %differeturn, double 0.000000e+00
; CHECK-NEXT:   br i1 %cmp, label %invertcond.true, label %invertentry

; CHECK: invertentry: 
; CHECK-NEXT:   %"x'de.0" = phi double [ %3, %invertcond.true ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   %1 = insertvalue { double } undef, double %"x'de.0", 0
; CHECK-NEXT:   ret { double } %1

; CHECK: invertcond.true: 
; CHECK-NEXT:   %2 = call { double } @diffef(double %x, double %0)
; CHECK-NEXT:   %3 = extractvalue { double } %2, 0
; CHECK-NEXT:   br label %invertentry

; CHECK: define internal {{(dso_local )?}}{ double } @diffef(double %x, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[result:.+]] = insertvalue { double } undef, double %[[differet]], 0
; CHECK-NEXT:   ret { double } %[[result]]
; CHECK-NEXT: }
