; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S | FileCheck %s

define void @callee() {
entry:
  %i = alloca { float, float, float, i1 }, align 4
  %f0 = getelementptr inbounds { float, float, float, i1 }, { float, float, float, i1 }* %i, i32 0, i32 0
  store float 1.000000e+00, float* %f0, align 4, !tbaa !8

  %f1 = getelementptr inbounds { float, float, float, i1 }, { float, float, float, i1 }* %i, i32 0, i32 1
  store float 1.000000e+00, float* %f1, align 4, !tbaa !8

  %f2 = getelementptr inbounds { float, float, float, i1 }, { float, float, float, i1 }* %i, i32 0, i32 2
  store float 1.000000e+00, float* %f2, align 4, !tbaa !8

  %int = getelementptr inbounds { float, float, float, i1 }, { float, float, float, i1 }* %i, i32 0, i32 3
  store i1 true, i1* %int, align 4, !tbaa !10

  %res = load { float, float, float, i1 }, { float, float, float, i1 }* %i, align 4

  ret void
}

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"float", !5, i64 0}
!8 = !{!7, !7, i64 0}
!9 = !{!"long", !5, i64 0}
!10 = !{!9, !9, i64 0}


; CHECK: callee - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %i = alloca { float, float, float, float, i64 }, align 4: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}
; CHECK-NEXT:   %f0 = getelementptr inbounds { float, float, float, float, i64 }, { float, float, float, float, i64 }* %i, i32 0, i32 0: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   store float 1.000000e+00, float* %f0, align 4, !tbaa !0: {}
; CHECK-NEXT:   %f1 = getelementptr inbounds { float, float, float, float, i64 }, { float, float, float, float, i64 }* %i, i32 0, i32 1: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   store float 1.000000e+00, float* %f1, align 4, !tbaa !0: {}
; CHECK-NEXT:   %f2 = getelementptr inbounds { float, float, float, float, i64 }, { float, float, float, float, i64 }* %i, i32 0, i32 2: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   store float 1.000000e+00, float* %f2, align 4, !tbaa !0: {}
; CHECK-NEXT:   %f3 = getelementptr inbounds { float, float, float, float, i64 }, { float, float, float, float, i64 }* %i, i32 0, i32 3: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   store float 1.000000e+00, float* %f3, align 4, !tbaa !0: {}
; CHECK-NEXT:   %int = getelementptr inbounds { float, float, float, float, i64 }, { float, float, float, float, i64 }* %i, i32 0, i32 4: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT:   store i64 314, i64* %int, align 4, !tbaa !4: {}
; CHECK-NEXT:   %res = load { float, float, float, float, i64 }, { float, float, float, float, i64 }* %i, align 4: {[0]:Float@float, [4]:Float@float, [8]:Float@float, [12]:Float@float, [16]:Integer, [17]:Integer, [18]:Integer, [19]:Integer, [20]:Integer, [21]:Integer, [22]:Integer, [23]:Integer}
; CHECK-NEXT:   ret void: {}
