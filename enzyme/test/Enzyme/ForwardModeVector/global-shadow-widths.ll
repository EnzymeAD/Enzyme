; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; One global differentiated at width 1 and at width 3 gets one shadow per
; width: a double, and a [3 x double] that holds the lanes. The program
; queries a shadow, so neither derivative may use a local shadow instead.

%struct.Tangents = type { double, double, double }

@g = global double 2.000000e+00, align 8

define double @f(double %x) {
entry:
  %v = load double, ptr @g, align 8
  %m = fmul double %v, %x
  store double %m, ptr @g, align 8
  ret double %m
}

declare double @__enzyme_fwddiff(...)

declare ptr @__enzyme_shadow(ptr, i32, i32)

define ptr @seed() {
entry:
  %p = call ptr @__enzyme_shadow(ptr @g, i32 3, i32 1)
  ret ptr %p
}

define double @test1(double %x) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @f, double %x, double 1.0)
  ret double %r
}

define %struct.Tangents @test3(double %x) {
entry:
  %r = call %struct.Tangents (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Tangents %r
}

; CHECK-DAG: @g.ad.l1.w1 = common global double 0.000000e+00, align 8
; CHECK-DAG: @g.ad.l1.w3 = common global [3 x double] zeroinitializer, align 8
; CHECK-DAG: @g = global double 2.000000e+00, align 8, !enzyme_shadows ![[shadows:[0-9]+]]

; CHECK: define ptr @seed()
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret ptr getelementptr inbounds ([3 x double], ptr @g.ad.l1.w3, i64 0, i64 1)

; CHECK: define internal double @fwddiffef(double %x, double %"x'")
; CHECK:   %"v'ipl" = load double, ptr @g.ad.l1.w1, align 8
; CHECK:   store double %{{.*}}, ptr @g.ad.l1.w1, align 8

; CHECK: define internal [3 x double] @fwddiffe3f(double %x, [3 x double] %"x'")
; CHECK:   load double, ptr @g.ad.l1.w3, align 8
; CHECK:   load double, ptr getelementptr inbounds ([3 x double], ptr @g.ad.l1.w3, i32 0, i32 1), align 8
; CHECK:   load double, ptr getelementptr inbounds ([3 x double], ptr @g.ad.l1.w3, i32 0, i32 2), align 8
; CHECK:   store double %{{.*}}, ptr @g.ad.l1.w3, align 8
; CHECK:   store double %{{.*}}, ptr getelementptr inbounds ([3 x double], ptr @g.ad.l1.w3, i32 0, i32 1), align 8
; CHECK:   store double %{{.*}}, ptr getelementptr inbounds ([3 x double], ptr @g.ad.l1.w3, i32 0, i32 2), align 8

; CHECK-DAG: ![[shadows]] = !{![[w3:[0-9]+]], ![[w1:[0-9]+]]}
; CHECK-DAG: ![[w1]] = !{i32 1, i32 1, ptr @g.ad.l1.w1}
; CHECK-DAG: ![[w3]] = !{i32 1, i32 3, ptr @g.ad.l1.w3}
