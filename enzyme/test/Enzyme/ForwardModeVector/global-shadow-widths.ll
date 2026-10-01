; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; One global differentiated at width 1 and in two contexts of width 3 gets
; one shadow for each: a double, and a [3 x double] per context that holds
; the lanes.

%struct.Tangents = type { double, double, double }

@g = global double 2.000000e+00, align 8

define double @f(double %x) {
entry:
  %v = load double, ptr @g, align 8
  %m = fmul double %v, %x
  store double %m, ptr @g, align 8
  ret double %m
}

@enzyme_context = external global i32

declare double @__enzyme_fwddiff(...)
declare ptr @__enzyme_context(i32)
declare ptr @__enzyme_shadow(ptr, ptr, i32)

define double @test1(double %x) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @f, double %x, double 1.0)
  ret double %r
}

define double @test3(double %x) {
entry:
  %ctx = call ptr @__enzyme_context(i32 3)
  %p = call ptr @__enzyme_shadow(ptr %ctx, ptr @g, i32 1)
  store double 1.0, ptr %p, align 8
  %r = call %struct.Tangents (...) @__enzyme_fwddiff(ptr @f, ptr @enzyme_context, ptr %ctx, double %x, double 1.0, double 2.0, double 3.0)
  %d = load double, ptr %p, align 8
  ret double %d
}

; Each context has shadows of its own, even at the same width.
define double @other3(double %x) {
entry:
  %ctx = call ptr @__enzyme_context(i32 3)
  %r = call %struct.Tangents (...) @__enzyme_fwddiff(ptr @f, ptr @enzyme_context, ptr %ctx, double %x, double 1.0, double 2.0, double 3.0)
  %p = call ptr @__enzyme_shadow(ptr %ctx, ptr @g, i32 2)
  %d = load double, ptr %p, align 8
  ret double %d
}

; CHECK-DAG: @g.ad.enzyme.context = private global [3 x double] zeroinitializer, align 8
; CHECK-DAG: @g.ad.enzyme.context.1 = private global [3 x double] zeroinitializer, align 8
; CHECK-DAG: @g.ad.w1 = common global double 0.000000e+00, align 8
; CHECK-DAG: @g = global double 2.000000e+00, align 8, !enzyme_shadows ![[shadows:[0-9]+]]

; CHECK: define double @test3(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store double 1.000000e+00, ptr getelementptr inbounds ([3 x double], ptr @g.ad.enzyme.context, i64 0, i64 1), align 8
; CHECK-NEXT:   %0 = call fast [3 x double] @fwddiffe3f(double %x, [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00])
; CHECK-NEXT:   %d = load double, ptr getelementptr inbounds ([3 x double], ptr @g.ad.enzyme.context, i64 0, i64 1), align 8

; CHECK: define double @other3(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast [3 x double] @fwddiffe3f.2(double %x, [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00])
; CHECK-NEXT:   %d = load double, ptr getelementptr inbounds ([3 x double], ptr @g.ad.enzyme.context.1, i64 0, i64 2), align 8

; CHECK: define internal double @fwddiffef(double %x, double %"x'")
; CHECK:   %"v'ipl" = load double, ptr @g.ad.w1, align 8
; CHECK:   store double %{{.*}}, ptr @g.ad.w1, align 8

; CHECK: define internal [3 x double] @fwddiffe3f(double %x, [3 x double] %"x'")
; CHECK:   load double, ptr @g.ad.enzyme.context, align 8
; CHECK:   load double, ptr getelementptr inbounds ([3 x double], ptr @g.ad.enzyme.context, i32 0, i32 1), align 8
; CHECK:   load double, ptr getelementptr inbounds ([3 x double], ptr @g.ad.enzyme.context, i32 0, i32 2), align 8
; CHECK:   store double %{{.*}}, ptr @g.ad.enzyme.context, align 8
; CHECK:   store double %{{.*}}, ptr getelementptr inbounds ([3 x double], ptr @g.ad.enzyme.context, i32 0, i32 1), align 8
; CHECK:   store double %{{.*}}, ptr getelementptr inbounds ([3 x double], ptr @g.ad.enzyme.context, i32 0, i32 2), align 8

; CHECK: define internal [3 x double] @fwddiffe3f.2(double %x, [3 x double] %"x'")
; CHECK:   load double, ptr @g.ad.enzyme.context.1, align 8
; CHECK:   store double %{{.*}}, ptr getelementptr inbounds ([3 x double], ptr @g.ad.enzyme.context.1, i32 0, i32 2), align 8

; CHECK: ![[shadows]] = !{![[c0:[0-9]+]], ![[c1:[0-9]+]], ![[w1:[0-9]+]]}
; CHECK-NEXT: ![[c0]] = !{ptr @enzyme.context, ptr @g.ad.enzyme.context}
; CHECK-NEXT: ![[c1]] = !{ptr @enzyme.context.1, ptr @g.ad.enzyme.context.1}
; CHECK-NEXT: ![[w1]] = !{i32 1, ptr @g.ad.w1}
