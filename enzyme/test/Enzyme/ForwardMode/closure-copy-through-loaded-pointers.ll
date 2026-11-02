; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; A lambda capturing two arrays by value keeps their data pointers in a local
; object, and the body copies *x into *y through pointers loaded from it. The
; stored value is read through a pointer loaded from %c, so it comes from x's
; memory, not from %c's, and %c must still be active.

declare void @__enzyme_fwddiff(...)

@enzyme_dup = global i32 0

define void @f(ptr %x, ptr %y) {
entry:
  %c = alloca [2 x ptr], align 8
  store ptr %y, ptr %c, align 8
  %c1 = getelementptr inbounds i8, ptr %c, i64 8
  store ptr %x, ptr %c1, align 8
  %xp = load ptr, ptr %c1, align 8
  %v = load double, ptr %xp, align 8
  %yp = load ptr, ptr %c, align 8
  store double %v, ptr %yp, align 8
  ret void
}
define void @df(ptr %x, ptr %dx, ptr %y, ptr %dy) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f, ptr @enzyme_dup, ptr %x, ptr %dx, ptr @enzyme_dup, ptr %y, ptr %dy)
  ret void
}

; CHECK: define internal void @fwddiffef(ptr %x, ptr %"x'", ptr %y, ptr %"y'")
; CHECK:   %"c'ipa" = alloca [2 x ptr], align 8
; CHECK:   %"v'ipl" = load double, ptr %"xp'ipl", align 8
; CHECK:   %"yp'ipl" = load ptr, ptr %"c'ipa", align 8
; CHECK:   store double %"v'ipl", ptr %"yp'ipl", align 8
