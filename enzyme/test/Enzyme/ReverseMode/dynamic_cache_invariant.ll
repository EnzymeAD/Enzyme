; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; The number of iterations of the loop is not known before it executes, so the
; cache of %v is reallocated as the loop executes. Within the forward pass, the
; pointer to the cache therefore changes between iterations, and loads of it
; must not be marked with !invariant.group. Otherwise, LLVM is free to replace
; such a load in one iteration with the pointer loaded in a prior iteration,
; which may have since been reallocated.

declare void @__enzyme_autodiff(i8*, ...)

define double @f(double %x) {
entry:
  br label %loop

loop:
  %v = phi double [ %x, %entry ], [ %sq, %loop ]
  %sq = fmul double %v, %v
  %cmp = fcmp olt double %sq, 1.000000e+02
  br i1 %cmp, label %loop, label %exit

exit:
  ret double %sq
}

define void @caller(double %x) {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @f to i8*), double %x)
  ret void
}

; CHECK: define internal {{.*}} @diffef(

; The forward pass load of the reallocated cache is not marked invariant.
; CHECK: store {{double\*|ptr}} %{{.*}}, {{double\*\*|ptr}} %v_cache, align 8
; CHECK: %{{.*}} = load {{double\*|ptr}}, {{double\*\*|ptr}} %v_cache, align 8, !dereferenceable ![[deref:[0-9]+]]{{$}}
; CHECK: store double %v, {{double\*|ptr}} %{{.*}}, align 8, !invariant.group

; The reverse pass loads of the now fully allocated cache are.
; CHECK: invertloop:
; CHECK: %{{.*}} = load {{double\*|ptr}}, {{double\*\*|ptr}} %v_cache, align 8, !dereferenceable ![[deref]], !invariant.group !{{[0-9]+}}
