; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,%simplifycfg)" -S | FileCheck %s

define void @f(i64 addrspace(11)* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}" %ptr) {
entry:
  store i64 0, i64 addrspace(11)* %ptr, align 8
  ret void
}

define void @test(i64 addrspace(11)* %ptr, i64 addrspace(11)* %dptr) {
entry:
  call void (...) @__enzyme_fwddiff(void (i64 addrspace(11)*)* @f, metadata !"enzyme_dup", i64 addrspace(11)* %ptr, i64 addrspace(11)* %dptr)
  ret void
}

declare void @__enzyme_fwddiff(...)

; CHECK: define internal void @fwddiffef({{(i64 addrspace\(11\)\*|ptr addrspace\(11\))}} "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}" %ptr, {{(i64 addrspace\(11\)\*|ptr addrspace\(11\))}} "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}" %"ptr'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i64 0, {{(i64 addrspace\(11\)\*|ptr addrspace\(11\))}} %"ptr'", align 8
; CHECK-NEXT:   store i64 0, {{(i64 addrspace\(11\)\*|ptr addrspace\(11\))}} %ptr, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
