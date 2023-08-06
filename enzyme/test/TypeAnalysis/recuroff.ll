; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=r -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=r -S -o /dev/null | FileCheck %s

@ptr = private unnamed_addr global [5000 x i64] zeroinitializer, align 1

define void @callee(i64* %x, i64 %off) {
entry:
  %gep = getelementptr inbounds i64, i64* %x, i64 %off
  %ld = load i64, i64* %gep, align 8, !tbaa !8
  %add = add i64 %off, 1
  call void @callee(i64* %x, i64 %add)
  ret void
}

define void @r(i64* %x) {
entry:
  call void @callee(i64* %x, i64 23)
  ret void
}

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"double", !5, i64 0}
!8 = !{!7, !7, i64 0}


; CHECK: callee - {} |{[-1]:Pointer}:{} {[-1]:Integer}:{23,} 
; CHECK-NEXT: i64* %x: {[-1]:Pointer, [-1,184]:Float@double}
; CHECK-NEXT: i64 %off: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %gep = getelementptr inbounds i64, i64* %x, i64 %off: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %ld = load i64, i64* %gep, align 8, !tbaa !0: {[-1]:Float@double}
; CHECK-NEXT:   %add = add i64 %off, 1: {[-1]:Integer}
; CHECK-NEXT:   call void @callee(i64* %x, i64 %add): {}
; CHECK-NEXT:   ret void: {}
; CHECK-NEXT: callee - {} |{[-1]:Pointer, [-1,184]:Float@double}:{} {[-1]:Integer}:{}
; CHECK-NEXT: i64* %x: {[-1]:Pointer, [-1,184]:Float@double}
; CHECK-NEXT: i64 %off: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %gep = getelementptr inbounds i64, i64* %x, i64 %off: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %ld = load i64, i64* %gep, align 8, !tbaa !0: {[-1]:Float@double}
; CHECK-NEXT:   %add = add i64 %off, 1: {[-1]:Integer}
; CHECK-NEXT:   call void @callee(i64* %x, i64 %add): {}
; CHECK-NEXT:   ret void: {}
