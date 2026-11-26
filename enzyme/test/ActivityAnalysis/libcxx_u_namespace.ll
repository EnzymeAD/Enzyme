; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=foo -S | FileCheck %s

@_ZNSt3__u4cerrE = external global [16 x i8], align 8

declare i64 @_ZNKSt3__u4hashIiEclEi(ptr, i32) readnone nounwind
declare void @use_cerr(ptr readonly nocapture) nounwind

define i64 @foo(ptr %h, i32 %x) {
entry:
  call void @use_cerr(ptr @_ZNSt3__u4cerrE)
  %hv = call i64 @_ZNKSt3__u4hashIiEclEi(ptr %h, i32 %x)
  ret i64 %hv
}

; CHECK: entry
; CHECK-NEXT:   call void @use_cerr(ptr @_ZNSt3__u4cerrE): icv:1 ici:1
; CHECK-NEXT:   %hv = call i64 @_ZNKSt3__u4hashIiEclEi(ptr %h, i32 %x): icv:1 ici:1
; CHECK-NEXT:   ret i64 %hv: icv:1 ici:1
