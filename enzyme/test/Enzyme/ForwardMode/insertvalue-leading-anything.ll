; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

; The inactive aggregate %base never has its first element written, so its
; type tree is Anything at bytes 0 to 7 and Float@double at bytes 8 to 15, and
; so is the tree of the active insertvalue %p over it. Nulling the shadow of
; %base with that tree must classify the aggregate as all float from its
; later floating entry rather than dereference a null leading type.

define double @f(double %x, double %y, double %z, i1 %c) {
  %by = insertvalue { double, double } undef, double %y, 1
  %bz = insertvalue { double, double } undef, double %z, 1
  %base = select i1 %c, { double, double } %by, { double, double } %bz
  %xx = fmul double %x, %x
  %p = insertvalue { double, double } %base, double %xx, 1
  %e1 = extractvalue { double, double } %p, 1
  ret double %e1
}

define double @df(double %x, double %y, double %z, i1 %c) {
  %r = call double (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_dup", double %x, double 1.0, metadata !"enzyme_const", double %y, metadata !"enzyme_const", double %z, metadata !"enzyme_const", i1 %c)
  ret double %r
}

declare double @__enzyme_fwddiff(...)

; The shadow of %base is null, and the tangent of the returned element is
; that of %xx alone.

; CHECK: define internal double @fwddiffef(double %x, double %"x'", double %y, double %z, i1 %c)
; CHECK-NEXT:   %[[a:.+]] = fmul fast double %"x'", %x
; CHECK-NEXT:   %[[b:.+]] = fmul fast double %"x'", %x
; CHECK-NEXT:   %[[s:.+]] = fadd fast double %[[a]], %[[b]]
; CHECK-NEXT:   ret double %[[s]]
