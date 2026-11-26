; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; The shadow of a global that is only used in the differentiated function is an
; alloca, so a constant GEP into the global needs a shadow GEP instruction.

@g = internal global [2 x double] zeroinitializer, align 8, !enzyme_type !0

define double @f(double %x) {
entry:
  store double %x, ptr getelementptr inbounds ([2 x double], ptr @g, i64 0, i64 1), align 8
  %l = load double, ptr getelementptr inbounds ([2 x double], ptr @g, i64 0, i64 1), align 8
  %m = fmul double %l, %l
  ret double %m
}

declare double @__enzyme_autodiff(...)

!0 = !{!"Unknown", i32 -1, !1}
!1 = !{!"Pointer", i32 -1, !2}
!2 = !{!"Float@double"}

define double @test(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x)
  ret double %r
}

; CHECK: define internal { double } @diffef(double %x, double %differeturn)
; CHECK: %"g'ipa" = alloca [2 x double], align 8
; CHECK: %"'ipg" = getelementptr [2 x double], ptr %"g'ipa", i64 0, i64 1
; CHECK: load double, ptr %"'ipg"
