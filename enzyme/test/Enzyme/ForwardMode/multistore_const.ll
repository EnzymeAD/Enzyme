; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,%simplifycfg)" -S | FileCheck %s

define void @f(i64 addrspace(11)* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}" %ptr) {
entry:
  store i64 0, i64 addrspace(11)* %ptr, align 8
  ret void
}

define void @f_vec(<2 x double>* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %ptr) {
entry:
  store <2 x double> <double 1.000000e+01, double 0.000000e+00>, <2 x double>* %ptr, align 8
  ret void
}

define void @test(i64 addrspace(11)* %ptr, i64 addrspace(11)* %dptr, <2 x double>* %vptr, <2 x double>* %dvptr) {
entry:
  call void (...) @__enzyme_fwddiff(void (i64 addrspace(11)*)* @f, metadata !"enzyme_dup", i64 addrspace(11)* %ptr, i64 addrspace(11)* %dptr)
  call void (...) @__enzyme_fwddiff(void (<2 x double>*)* @f_vec, metadata !"enzyme_dup", <2 x double>* %vptr, <2 x double>* %dvptr)
  ret void
}

declare void @__enzyme_fwddiff(...)

; CHECK: define internal void @fwddiffef({{(i64 addrspace\(11\)\*|ptr addrspace\(11\))}} "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}" %ptr, {{(i64 addrspace\(11\)\*|ptr addrspace\(11\))}} "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}" %"ptr'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i64 0, {{(i64 addrspace\(11\)\*|ptr addrspace\(11\))}} %"ptr'", align 8
; CHECK-NEXT:   store i64 0, {{(i64 addrspace\(11\)\*|ptr addrspace\(11\))}} %ptr, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @fwddiffef_vec({{(<2 x double>\*|ptr)}} "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %ptr, {{(<2 x double>\*|ptr)}} "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %"ptr'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   store <2 x double> zeroinitializer, {{(<2 x double>\*|ptr)}} %"ptr'", align 8
; CHECK-NEXT:   store <2 x double> <double 1.000000e+01, double 0.000000e+00>, {{(<2 x double>\*|ptr)}} %ptr, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

