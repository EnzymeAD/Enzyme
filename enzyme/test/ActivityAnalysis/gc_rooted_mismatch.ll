; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f -S | FileCheck %s

define ptr @get_tmp(ptr readnone %primary, ptr %roots) {
entry:
  %val = load ptr, ptr %roots, align 8
  ret ptr %val
}

define void @f(double %x, ptr %prim_val, ptr %root_val) {
entry:
  %primary = alloca ptr, align 8
  store ptr %prim_val, ptr %primary, align 8
  
  %roots = alloca ptr, align 8
  store ptr %root_val, ptr %roots, align 8
  
  %buf = call ptr @get_tmp(ptr nocapture readnone %primary, ptr %roots)
  
  store double %x, ptr %buf, align 8
  ret void
}

; CHECK: double %x: icv:0
; CHECK-NEXT: ptr %prim_val: icv:0
; CHECK-NEXT: ptr %root_val: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %primary = alloca ptr, align 8: icv:1
; CHECK:   %roots = alloca ptr, align 8: icv:0
