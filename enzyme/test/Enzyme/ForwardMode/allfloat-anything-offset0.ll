; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Storing a byte extracted from an aggregate whose type tree is
; {[0]:Anything, [1..6]:Integer} (no [-1] entry) asks TypeTree::allFloat with
; anythingIsFloat for the shadow. Offset 0 being Anything left the float type
; null, and allFloat then took the size of a null type (segfault). This is
; e.g. the padding after a 1-byte field of a Julia struct, which Julia 1.13
; copies as [7 x i8].

declare void @__enzyme_fwddiff(...)

@enzyme_const = external global i32

define void @f(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Anything, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer}" %src, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Anything}" %dst, ptr %fdst, double %x) {
entry:
  %agg = load [7 x i8], ptr %src, align 1
  %e = extractvalue [7 x i8] %agg, 0
  store i8 %e, ptr %dst, align 1
  store double %x, ptr %fdst, align 8
  ret void
}

define void @test(ptr %src, ptr %dst, ptr %ddst, ptr %fdst, ptr %dfdst, double %x) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f, ptr @enzyme_const, ptr %src, ptr %dst, ptr %ddst, ptr %fdst, ptr %dfdst, double %x, double 1.0)
  ret void
}

; CHECK: define internal void @fwddiffef(
; CHECK:   %agg = load [7 x i8], ptr %src
; CHECK:   %e = extractvalue [7 x i8] %agg, 0
; CHECK:   %"e'ipev" = extractvalue [7 x i8] %{{.*}}, 0
; CHECK:   store i8 %"e'ipev", ptr %"dst'"
; CHECK:   store i8 %e, ptr %dst
; CHECK:   store double %"x'", ptr %"fdst'"
; CHECK:   store double %x, ptr %fdst
; CHECK:   ret void
