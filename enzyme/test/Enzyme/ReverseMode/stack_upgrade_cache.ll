; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

define float @sub(float %this, i32 %cond) {
entry:
  %tobool = icmp ne i32 %cond, 0
  br i1 %tobool, label %then, label %else

then:
  %malloccall = call noalias ptr @malloc(i64 4)
  store float %this, ptr %malloccall, align 4
  %val = load float, ptr %malloccall, align 4
  %res = fmul float %val, %val
  call void @free(ptr %malloccall)
  br label %else

else:
  %phi = phi float [ %res, %then ], [ %this, %entry ]
  ret float %phi
}

define float @g(float %t, i32 %cond) {
entry:
  %0 = tail call float (float (float, i32)*, ...) @__enzyme_autodiff(float (float, i32)* @sub, float %t, i32 %cond)
  ret float %0
}

declare float @__enzyme_autodiff(float (float, i32)*, ...)
declare noalias ptr @malloc(i64)
declare void @free(ptr)

; CHECK: define internal { float } @diffesub(float %this, i32 %cond, float %differeturn) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"phi'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, ptr %"phi'de", align 4
; CHECK-NEXT:   %"this'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, ptr %"this'de", align 4
; CHECK-NEXT:   %"res'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, ptr %"res'de", align 4
; CHECK-NEXT:   %malloccall_cache = alloca ptr, align 8
; CHECK-NEXT:   %"val'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, ptr %"val'de", align 4
; CHECK-NEXT:   %"malloccall'mi_cache" = alloca ptr, align 8
; CHECK-NEXT:   %tobool = icmp ne i32 %cond, 0
; CHECK-NEXT:   br i1 %tobool, label %then, label %else

; CHECK: then:{{.*}}
; CHECK-NEXT:   %"malloccall'mi" = call noalias nonnull dereferenceable(4) dereferenceable_or_null(4) ptr @malloc(i64 4) #2, !enzyme_cache_alloc ![[MD_0:[0-9]+]]
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr nonnull dereferenceable(4) dereferenceable_or_null(4) %"malloccall'mi", i8 0, i64 4, i1 false)
; CHECK-NEXT:   %malloccall = call noalias nonnull dereferenceable(4) dereferenceable_or_null(4) ptr @malloc(i64 4) #2, !enzyme_cache_alloc ![[MD_2:[0-9]+]]
; CHECK-NEXT:   store ptr %malloccall, ptr %malloccall_cache, align 8, !invariant.group ![[INV_4:[0-9]+]]
; CHECK-NEXT:   store ptr %"malloccall'mi", ptr %"malloccall'mi_cache", align 8, !invariant.group ![[INV_5:[0-9]+]]
; CHECK-NEXT:   store float %this, ptr %malloccall, align 4, !alias.scope ![[SCOPE_6:[0-9]+]], !noalias ![[NOALIAS_9:[0-9]+]]
; CHECK-NEXT:   %val = load float, ptr %malloccall, align 4, !alias.scope ![[SCOPE_6]], !noalias ![[NOALIAS_9]]
; CHECK-NEXT:   br label %else

; CHECK: else:{{.*}}
; CHECK-NEXT:   br label %invertelse

; CHECK: invertentry:{{.*}}
; CHECK-NEXT:   %0 = load float, ptr %"this'de", align 4
; CHECK-NEXT:   %1 = insertvalue { float } undef, float %0, 0
; CHECK-NEXT:   ret { float } %1

; CHECK: invertthen:{{.*}}
; CHECK-NEXT:   %2 = load float, ptr %"res'de", align 4
; CHECK-NEXT:   store float 0.000000e+00, ptr %"res'de", align 4
; CHECK-NEXT:   %3 = load ptr, ptr %malloccall_cache, align 8, !invariant.group ![[INV_4]]
; CHECK-NEXT:   %val_unwrap = load float, ptr %3, align 4, !alias.scope ![[SCOPE_6]], !noalias ![[NOALIAS_9]]
; CHECK-NEXT:   %4 = fmul fast float %2, %val_unwrap
; CHECK-NEXT:   %5 = load float, ptr %"val'de", align 4
; CHECK-NEXT:   %6 = fadd fast float %5, %4
; CHECK-NEXT:   store float %6, ptr %"val'de", align 4
; CHECK-NEXT:   %7 = fmul fast float %2, %val_unwrap
; CHECK-NEXT:   %8 = load float, ptr %"val'de", align 4
; CHECK-NEXT:   %9 = fadd fast float %8, %7
; CHECK-NEXT:   store float %9, ptr %"val'de", align 4
; CHECK-NEXT:   %10 = load float, ptr %"val'de", align 4
; CHECK-NEXT:   store float 0.000000e+00, ptr %"val'de", align 4
; CHECK-NEXT:   %11 = load ptr, ptr %"malloccall'mi_cache", align 8, !invariant.group ![[INV_5]]
; CHECK-NEXT:   %12 = load float, ptr %11, align 4, !alias.scope ![[NOALIAS_9]], !noalias ![[SCOPE_6]]
; CHECK-NEXT:   %13 = fadd fast float %12, %10
; CHECK-NEXT:   store float %13, ptr %11, align 4, !alias.scope ![[NOALIAS_9]], !noalias ![[SCOPE_6]]
; CHECK-NEXT:   %14 = load float, ptr %11, align 4, !alias.scope ![[NOALIAS_9]], !noalias ![[SCOPE_6]]
; CHECK-NEXT:   store float 0.000000e+00, ptr %11, align 4, !alias.scope ![[NOALIAS_9]], !noalias ![[SCOPE_6]]
; CHECK-NEXT:   %15 = load float, ptr %"this'de", align 4
; CHECK-NEXT:   %16 = fadd fast float %15, %14
; CHECK-NEXT:   store float %16, ptr %"this'de", align 4
; CHECK-NEXT:   call void @free(ptr nonnull %11), !enzyme_cache_free ![[MD_0]]
; CHECK-NEXT:   call void @free(ptr %3), !enzyme_cache_free ![[MD_2]]
; CHECK-NEXT:   br label %invertentry

; CHECK: invertelse:{{.*}}
; CHECK-NEXT:   store float %differeturn, ptr %"phi'de", align 4
; CHECK-NEXT:   %17 = load float, ptr %"phi'de", align 4
; CHECK-NEXT:   store float 0.000000e+00, ptr %"phi'de", align 4
; CHECK-NEXT:   %18 = xor i1 %tobool, true
; CHECK-NEXT:   %19 = select fast i1 %tobool, float %17, float 0.000000e+00
; CHECK-NEXT:   %20 = load float, ptr %"res'de", align 4
; CHECK-NEXT:   %21 = fadd fast float %20, %17
; CHECK-NEXT:   %22 = select fast i1 %tobool, float %21, float %20
; CHECK-NEXT:   store float %22, ptr %"res'de", align 4
; CHECK-NEXT:   %23 = select fast i1 %18, float %17, float 0.000000e+00
; CHECK-NEXT:   %24 = load float, ptr %"this'de", align 4
; CHECK-NEXT:   %25 = fadd fast float %24, %17
; CHECK-NEXT:   %26 = select fast i1 %tobool, float %24, float %25
; CHECK-NEXT:   store float %26, ptr %"this'de", align 4
; CHECK-NEXT:   br i1 %tobool, label %invertthen, label %invertentry
; CHECK-NEXT: }

; CHECK-DAG: ![[MD_0]] = !{![[MD_0_INNER:[0-9]+]]}
; CHECK-DAG: ![[MD_0_INNER]] = distinct !{i1 true}
; CHECK-DAG: ![[MD_2]] = !{![[MD_2_INNER:[0-9]+]]}
; CHECK-DAG: ![[MD_2_INNER]] = distinct !{i1 true}

