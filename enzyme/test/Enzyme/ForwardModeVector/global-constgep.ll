; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

%struct.Gradients = type { double, double }

@global = external dso_local local_unnamed_addr global [4 x double], align 8, !enzyme_shadow !{ptr @dglobal}
@dglobal = external dso_local local_unnamed_addr global [2 x [4 x double]], align 8

declare %struct.Gradients @__enzyme_fwddiff(ptr, ...)

define double @mulglobal(double %x) {
entry:
  %0 = load double, ptr getelementptr inbounds ([4 x double], ptr @global, i64 0, i64 2), align 8
  %mul = fmul double %0, %x
  ret double %mul
}

define %struct.Gradients @derivative(double %x) {
entry:
  %r = call %struct.Gradients (ptr, ...) @__enzyme_fwddiff(ptr @mulglobal, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 2.0)
  ret %struct.Gradients %r
}

; Each lane's shadow load must index that lane's slot of @dglobal, not the
; [2 x ptr] aggregate of the per-lane shadows.

; CHECK: define internal [2 x double] @fwddiffe2mulglobal(double %x, [2 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'ipl" = load double, ptr getelementptr inbounds ([4 x double], ptr @dglobal, i64 0, i64 2), align 8
; CHECK-NEXT:   %"'ipl1" = load double, ptr getelementptr inbounds ({{(\[4 x double\], ptr getelementptr inbounds \()?}}[2 x [4 x double]], ptr @dglobal, i32 0, i32 1{{(\), i64 0)?}}, i64 2), align 8
; CHECK-NEXT:   %[[a0:.+]] = load double, ptr getelementptr inbounds ([4 x double], ptr @global, i64 0, i64 2), align 8
