; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -early-cse -instcombine -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instcombine,%simplifycfg)" -S | FileCheck %s

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

define dso_local <4 x double> @f(<4 x double> %x) #1 {
entry:
  ret <4 x double> %x
}

define dso_local i1 @cmp() #1 {
entry:
  ret i1 true
}

define dso_local <4 x double> @relu(<4 x double> %x) {
entry:
  %cmp = call i1 @cmp()
  br i1 %cmp, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  %call = tail call fast <4 x double> @f(<4 x double> %x)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi <4 x double> [ %call, %cond.true ], [ zeroinitializer, %entry ]
  ret <4 x double> %cond
}

define dso_local <4 x double> @drelu(<4 x double> %x) {
entry:
  %0 = tail call <4 x double> (<4 x double> (<4 x double>)*, ...) @__enzyme_autodiff(<4 x double> (<4 x double>)* nonnull @relu, <4 x double> %x)
  ret <4 x double> %0
}

declare <4 x double> @__enzyme_autodiff(<4 x double> (<4 x double>)*, ...) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone noinline }

; CHECK: define internal { <4 x double> } @differelu(<4 x double> %x, <4 x double> %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = call i1 @cmp()
; CHECK-NEXT:   br i1 %cmp, label %invertcond.true, label %invertentry

; CHECK: invertentry: 
; CHECK-NEXT:   %"x'de.0" = phi <4 x double> [ %[[result:.+]], %invertcond.true ], [ zeroinitializer, %entry ]
; CHECK-NEXT:   %[[res:.+]] = insertvalue { <4 x double> } undef, <4 x double> %"x'de.0", 0
; CHECK-NEXT:   ret { <4 x double> } %[[res]]

; CHECK: invertcond.true:                                ; preds = %entry
; CHECK-NEXT:   %[[diffef:.+]] = call { <4 x double> } @diffef(<4 x double> %x, <4 x double> %differeturn)
; CHECK-NEXT:   %[[result]] = extractvalue { <4 x double> } %[[diffef]], 0
; CHECK-NEXT:   br label %invertentry

; CHECK: define internal {{(dso_local )?}}{ <4 x double> } @diffef(<4 x double> %x, <4 x double> %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[result:.+]] = insertvalue { <4 x double> } undef, <4 x double> %[[differet]], 0
; CHECK-NEXT:   ret { <4 x double> } %[[result]]
; CHECK-NEXT: }
