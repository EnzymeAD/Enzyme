; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=get -S -o /dev/null | FileCheck %s

; An array of 100 elements of type t { real(8) :: a(3); integer(8) :: n }:
; one entry per component, repeated with the element size as stride.

define void @get(ptr "enzyme_type"="{[-1]:Pointer, [-1,0+32*100+1*24]:Float@double, [-1,24+32*100]:Integer}" %x, i64 %i, ptr %out) {
entry:
  %elt = getelementptr inbounds [32 x i8], ptr %x, i64 %i
  %a2p = getelementptr inbounds i8, ptr %elt, i64 16
  %a2 = load i64, ptr %a2p
  store i64 %a2, ptr %out
  %np = getelementptr inbounds i8, ptr %elt, i64 24
  %n = load i64, ptr %np
  %last = getelementptr inbounds i8, ptr %x, i64 3192
  %lastn = load i64, ptr %last
  ret void
}

; CHECK: ptr %x: {[-1]:Pointer, [-1,0+32*100+1*24]:Float@double, [-1,24+32*100]:Integer}
; CHECK: %elt = getelementptr inbounds [32 x i8], ptr %x, i64 %i: {[-1]:Pointer, [-1,0+1*24]:Float@double, [-1,24]:Integer}
; CHECK: %a2 = load i64, ptr %a2p, align 4: {[-1]:Float@double}
; CHECK: %n = load i64, ptr %np, align 4: {[0]:Integer}
; CHECK: %lastn = load i64, ptr %last, align 4: {[0]:Integer}
