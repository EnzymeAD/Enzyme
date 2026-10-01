; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; A function whose derivative comes from a custom rule (enzyme_math =
; enzyme_custom) may have that rule read a parameter its own body never
; reads, so the body alone must not make such a parameter writeonly or
; readnone. A parameter that only carries the result (sret, also one still
; marked enzyme_sret for Enzyme.jl's calling-convention fixup, sret union,
; returnRoots) is still annotated, as is every parameter of an ordinary
; function.

define void @custom(ptr noalias sret(double) %out, ptr %x, ptr %unused) #0 {
  store double 1.0, ptr %x
  store double 2.0, ptr %out
  ret void
}

define void @custom_enzyme_sret(ptr noalias "enzyme_sret"="test_type4" %out, ptr %x) #0 {
  store double 1.0, ptr %x
  store double 2.0, ptr %out
  ret void
}

; What the function was already marked with is kept, on parameters and on the
; function itself: those markings were stated deliberately.
define void @custom_marked(ptr noalias sret(double) %out, ptr writeonly %x, ptr readnone %unused) #1 {
  store double 1.0, ptr %x
  store double 2.0, ptr %out
  ret void
}

define double @custom_scalar(double %x) #2 {
  %r = fmul double %x, %x
  ret double %r
}

; enzyme_custom_full_attributes states that the rule accesses no more than the
; body does (Enzyme.jl sets it for @easy_rule), so such a function is
; annotated like an ordinary one.
define void @custom_easy(ptr noalias sret(double) %out, ptr %x, ptr %unused) #3 {
  store double 1.0, ptr %x
  store double 2.0, ptr %out
  ret void
}

define void @plain(ptr noalias sret(double) %out, ptr %x, ptr %unused) {
  store double 1.0, ptr %x
  store double 2.0, ptr %out
  ret void
}

attributes #0 = { "enzyme_math"="enzyme_custom" }
attributes #1 = { memory(argmem: write) "enzyme_math"="enzyme_custom" }
attributes #2 = { memory(none) "enzyme_math"="enzyme_custom" }
attributes #3 = { memory(argmem: write) "enzyme_math"="enzyme_custom" "enzyme_custom_full_attributes" }

; CHECK: define void @custom(ptr noalias {{.*}}writeonly{{.*}} %out, ptr {{(nocapture|captures\(none\))}} %x, ptr {{.*}}readonly{{.*}} %unused)
; CHECK: define void @custom_enzyme_sret(ptr noalias {{.*}}writeonly{{.*}} %out, ptr {{(nocapture|captures\(none\))}} %x)
; CHECK: define void @custom_marked(ptr noalias {{.*}}writeonly{{.*}} %out, ptr {{.*}}writeonly{{.*}} %x, ptr {{.*}}readnone{{.*}} %unused) #[[MARKED:[0-9]+]]
; CHECK: define double @custom_scalar(double %x) #[[SCALAR:[0-9]+]]
; CHECK: define void @custom_easy(ptr noalias {{.*}}writeonly{{.*}} %out, ptr {{.*}}writeonly{{.*}} %x, ptr {{.*}}readnone{{.*}} %unused) #[[EASY:[0-9]+]]
; CHECK: define void @plain(ptr noalias {{.*}}writeonly{{.*}} %out, ptr {{.*}}writeonly{{.*}} %x, ptr {{.*}}readnone{{.*}} %unused)
; CHECK: attributes #[[MARKED]] = { {{.*}}memory(argmem: write){{.*}} }
; CHECK: attributes #[[SCALAR]] = { {{.*}}memory(none){{.*}} }
; CHECK: attributes #[[EASY]] = { {{.*}}memory(argmem: write){{.*}}"enzyme_custom_full_attributes"{{.*}} }
