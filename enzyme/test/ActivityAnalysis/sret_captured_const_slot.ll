; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=f -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f -S -o /dev/null | FileCheck %s

; make_env writes only its sret parameter and so is marked as writing no
; memory the caller can see (preprocessing derives the attribute; here it is
; stated directly), yet it stores %cslot into that active sret, from where
; the closure loads through it. The slot must be active so it gets a shadow.

define internal double @closure(ptr %env, ptr %xp) #1 {
  %cp = load ptr, ptr %env, align 8
  %c = load double, ptr %cp, align 8
  %x = load double, ptr %xp, align 8
  %cx = fmul double %c, %x
  ret double %cx
}

define internal void @make_env(ptr sret([16 x i8]) %out, ptr %x, ptr %cp) #0 {
  store ptr %x, ptr %out, align 8
  %slot = getelementptr inbounds i8, ptr %out, i64 8
  store ptr %cp, ptr %slot, align 8
  ret void
}

define internal double @sum(ptr %env) #1 {
  %x = load ptr, ptr %env, align 8
  %envc = getelementptr inbounds i8, ptr %env, i64 8
  %r = call double @closure(ptr %envc, ptr %x)
  ret double %r
}

define double @f(ptr %x) {
  %env = alloca [16 x i8], align 8
  %cslot = alloca double, align 8
  store double 2.000000e+00, ptr %cslot, align 8
  call void @make_env(ptr sret([16 x i8]) %env, ptr %x, ptr %cslot)
  %r = call double @sum(ptr %env)
  ret double %r
}

attributes #0 = { "enzyme_LocalReadOnlyOrThrow" }
attributes #1 = { readonly nounwind }

; CHECK: ptr %x: icv:0
; CHECK: %env = alloca [16 x i8], align 8: icv:0 ici:1
; CHECK-NEXT: %cslot = alloca double, align 8: icv:0 ici:1
; CHECK-NEXT: store double 2.000000e+00, ptr %cslot, align 8: icv:1 ici:1
