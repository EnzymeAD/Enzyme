; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=0 -enzyme -mem2reg -early-cse -simplifycfg -instsimplify -correlated-propagation -simplifycfg -adce -S -enzyme-detect-readthrow=0 | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=0 -passes="enzyme,function(mem2reg,early-cse,%simplifycfg,instsimplify,correlated-propagation,%simplifycfg,adce)" -S -enzyme-detect-readthrow=0 | FileCheck %s

; Regression test for issue #2629:
; An outer branch guarded by a constant (enzyme_const) bool `fan` contains an
; inner branch guarded by an active predicate `cond` (computed from `%a`).
; When fan=false, the inner predicate is never computed, so its tape cache is
; never written. The reverse pass must not branch on an uninitialized/undef
; taped predicate in that case.
;
; The fix ensures that:
; 1. The inner predicate cache (`_cache.0`) is initialized to `false` (not
;    undef) for the path where the outer guard is not taken.
; 2. The `cond_unwrap` lookup in the reverse pass is deferred to a `staging`
;    block that is only reached when `fan` is true.

declare double @__enzyme_autodiff(i8*, ...)

define double @f(double* %a, i1 %fan) {
entry:
  br i1 %fan, label %if.fan, label %merge

if.fan:
  %a0 = load double, double* %a, align 8
  %cond = fcmp ogt double %a0, 0.000000e+00
  br i1 %cond, label %inner, label %merge

inner:
  %gp = getelementptr inbounds double, double* %a, i32 1
  %a1 = load double, double* %gp, align 8
  br label %merge

merge:
  %res = phi double [0.000000e+00, %entry], [1.000000e+00, %if.fan], [%a1, %inner]
  ret double %res
}

define void @caller(double* %a, double* %da, i1 %fan) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(
    i8* bitcast (double (double*, i1)* @f to i8*),
    double* nonnull %a, double* nonnull %da,
    i1 %fan)
  ret void
}

; CHECK: define internal void @diffef(double* %a, double* %"a'", i1 %fan, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br i1 %fan, label %if.fan, label %invertmerge

; CHECK: if.fan:
; CHECK-NEXT:   %a0 = load double, double* %a
; CHECK-NEXT:   %cond = fcmp ogt double %a0, 0.000000e+00
; CHECK-NEXT:   br i1 %cond, label %inner, label %invertmerge

; CHECK: inner:
; CHECK-NEXT:   br label %invertmerge

; CHECK: invertentry:
; CHECK-NEXT:   ret void

; CHECK: invertinner:
; CHECK-NEXT:   %"gp'ipg_unwrap" = getelementptr inbounds double, double* %"a'", i32 1
; CHECK-NEXT:   %[[a0:.+]] = load double, double* %"gp'ipg_unwrap"
; CHECK-NEXT:   %[[a1:.+]] = fadd fast double %[[a0]], %[[sel:.+]]
; CHECK-NEXT:   store double %[[a1]], double* %"gp'ipg_unwrap"
; CHECK-NEXT:   br label %invertentry

; CHECK: invertmerge:
; NOTE: the `false` entry for %entry predecessor is critical - without the fix
; this would be `undef`, causing the reverse to take the inner path spuriously.
; CHECK-NEXT:   %_cache.0 = phi i1 [ true, %inner ], [ false, %if.fan ], [ false, %entry ]
; CHECK-NEXT:   %[[sel]] = select {{(fast )?}}i1 %_cache.0, double %differeturn, double 0.000000e+00
; NOTE: cond2 is only evaluated inside %staging, which is only reachable when
; %fan is true, preventing the uninitialized-predicate crash from issue #2629.
; CHECK-NEXT:   br i1 %fan, label %staging, label %invertentry

; CHECK: staging:
; CHECK-NEXT:   %a0_unwrap = load double, double* %a
; CHECK-NEXT:   %cond_unwrap = fcmp ogt double %a0_unwrap, 0.000000e+00
; CHECK-NEXT:   br i1 %cond_unwrap, label %invertinner, label %invertentry
; CHECK-NEXT: }
