; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; The shadow copies are emitted per run of like-typed bytes. Each must carry the
; part of `enzyme_truetype` for its own bytes, rebased to where it starts, so
; that differentiating the derivative again can still type them.

define void @f(ptr %dst, ptr %src) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 100000, i1 false), !enzyme_truetype !0
  call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 100000, i1 false), !enzyme_truetype !0
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

declare void @__enzyme_fwddiff(...)

define void @df(ptr %dst, ptr %ddst, ptr %src) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_dup", ptr %dst, ptr %ddst, metadata !"enzyme_const", ptr %src)
  ret void
}

!0 = !{!"Float@double", i64 0, !"Integer", i64 8, !"Float@double", i64 50000, !"Integer", i64 50008}

; CHECK: define internal void @fwddiffef(ptr nocapture writeonly %dst, ptr nocapture %"dst'", ptr nocapture readonly %src)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 1 %"dst'", i8 0, i64 8, i1 {{(true|false)}}), !enzyme_truetype ![[FLT:[0-9]+]]
; CHECK-NEXT:   %[[a:.+]] = getelementptr inbounds i8, ptr %"dst'", i64 8
; CHECK-NEXT:   %[[b:.+]] = getelementptr inbounds i8, ptr %src, i64 8
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %[[a]], ptr %[[b]], i64 49992, i1 false) #{{[0-9]+}}, !enzyme_truetype ![[INT:[0-9]+]]
; CHECK-NEXT:   %[[c:.+]] = getelementptr inbounds i8, ptr %"dst'", i64 50000
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 1 %[[c]], i8 0, i64 8, i1 {{(true|false)}}), !enzyme_truetype ![[FLT]]
; CHECK-NEXT:   %[[d:.+]] = getelementptr inbounds i8, ptr %"dst'", i64 50008
; CHECK-NEXT:   %[[e:.+]] = getelementptr inbounds i8, ptr %src, i64 50008
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %[[d]], ptr %[[e]], i64 49992, i1 false) #{{[0-9]+}}, !enzyme_truetype ![[INT]]
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 100000, i1 false) #{{[0-9]+}}, !enzyme_truetype ![[FULL:[0-9]+]]
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 100000, i1 false) #{{[0-9]+}}, !enzyme_truetype ![[FULL]]
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr %"dst'", i8 0, i64 100000, i1 false) #{{[0-9]+}}, !enzyme_truetype ![[FULL]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK-DAG: ![[FULL]] = !{!"Float@double", i64 0, !"Integer", i64 8, !"Float@double", i64 50000, !"Integer", i64 50008}
; CHECK-DAG: ![[FLT]] = !{!"Float@double", i64 0}
; CHECK-DAG: ![[INT]] = !{!"Integer", i64 0}
