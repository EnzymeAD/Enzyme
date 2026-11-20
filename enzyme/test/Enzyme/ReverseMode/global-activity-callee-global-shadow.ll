; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -enzyme-global-activity=1 -S | FileCheck %s; fi

; Under global activity, @f writes @g and its callee @leaf reads it. Both
; derivatives must use the same shadow global: a zeroed local shadow in
; @diffeleaf would drop the adjoint that @diffef reads from @g_shadow.

@g = internal global [2 x double] zeroinitializer, align 8, !enzyme_type !0

define internal double @leaf() noinline {
entry:
  %l = load double, ptr getelementptr inbounds ([2 x double], ptr @g, i64 0, i64 1), align 8
  %m = fmul double %l, %l
  ret double %m
}

define double @f(double %x) {
entry:
  store double %x, ptr getelementptr inbounds ([2 x double], ptr @g, i64 0, i64 1), align 8
  %r = call double @leaf()
  ret double %r
}

declare double @__enzyme_autodiff(...)

define double @test(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x)
  ret double %r
}

!0 = !{!"Unknown", i32 -1, !1}
!1 = !{!"Pointer", i32 -1, !2}
!2 = !{!"Float@double"}

; CHECK: define internal { double } @diffef(double %x, double %differeturn)
; CHECK: invertentry:
; CHECK-NEXT:   call void @diffeleaf(double %differeturn)
; CHECK-NEXT:   %[[DG:.+]] = load double, ptr getelementptr inbounds ([2 x double], ptr @g_shadow, i64 0, i64 1), align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr getelementptr inbounds ([2 x double], ptr @g_shadow, i64 0, i64 1), align 8
; CHECK-NEXT:   %[[RET:.+]] = insertvalue { double } undef, double %[[DG]], 0
; CHECK-NEXT:   ret { double } %[[RET]]

; CHECK: define internal void @diffeleaf(double %differeturn)
; CHECK-NOT: alloca
; CHECK: invertentry:
; CHECK-NEXT:   %[[M0:.+]] = fmul fast double %differeturn, %l
; CHECK-NEXT:   %[[M1:.+]] = fmul fast double %differeturn, %l
; CHECK-NEXT:   %[[SUM:.+]] = fadd fast double %[[M0]], %[[M1]]
; CHECK-NEXT:   %[[OLD:.+]] = load double, ptr getelementptr inbounds ([2 x double], ptr @g_shadow, i64 0, i64 1), align 8
; CHECK-NEXT:   %[[NEW:.+]] = fadd fast double %[[OLD]], %[[SUM]]
; CHECK-NEXT:   store double %[[NEW]], ptr getelementptr inbounds ([2 x double], ptr @g_shadow, i64 0, i64 1), align 8
; CHECK-NEXT:   ret void
