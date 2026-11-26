; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

; A COMMON block of a REAL*8 buffer followed by a REAL*4 one (MITgcm's
; /EE_BUFFERS_LOCAL/). With [-1,-1] for both dummies the two types collide at
; offset 32768; with runs of the dummies' sizes they lie side by side, and the
; REAL*4 one is kept although it starts beyond -enzyme-max-type-offset.

@ee_buffers_local_ = global [49152 x i8] zeroinitializer

declare void @r8(ptr "enzyme_type"="{[-1]:Pointer, [-1,0+1*32768]:Float@double}")
declare void @r4(ptr "enzyme_type"="{[-1]:Pointer, [-1,0+1*16384]:Float@float}")

define void @caller() {
entry:
  call void @r8(ptr @ee_buffers_local_)
  %r4buf = getelementptr inbounds i8, ptr @ee_buffers_local_, i64 32768
  call void @r4(ptr %r4buf)
  %first8 = load i64, ptr @ee_buffers_local_
  %last8p = getelementptr inbounds i8, ptr @ee_buffers_local_, i64 32760
  %last8 = load i64, ptr %last8p
  %first4 = load i32, ptr %r4buf
  %last4p = getelementptr inbounds i8, ptr @ee_buffers_local_, i64 49148
  %last4 = load i32, ptr %last4p
  ret void
}

; CHECK: caller - {} |
; CHECK: %r4buf = getelementptr inbounds i8, ptr @ee_buffers_local_, i64 32768: {[-1]:Pointer, [-1,0+1*16384]:Float@float}
; CHECK: %first8 = load i64, ptr @ee_buffers_local_, align 4: {[-1]:Float@double}
; CHECK: %last8p = getelementptr inbounds i8, ptr @ee_buffers_local_, i64 32760: {[-1]:Pointer, [-1,0+1*8]:Float@double, [-1,8+1*16384]:Float@float}
; CHECK: %last8 = load i64, ptr %last8p, align 4: {[-1]:Float@double}
; CHECK: %first4 = load i32, ptr %r4buf, align 4: {[-1]:Float@float}
; CHECK: %last4p = getelementptr inbounds i8, ptr @ee_buffers_local_, i64 49148: {[-1]:Pointer, [-1,0+1*4]:Float@float}
; CHECK: %last4 = load i32, ptr %last4p, align 4: {[-1]:Float@float}
