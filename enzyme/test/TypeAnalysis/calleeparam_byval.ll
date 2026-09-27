; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

; The callee annotates a 16-byte by-value parameter. That annotation has to be
; checked against the parameter's size, not against the size of the call's
; (here void) result, which used to abort with "Canonicalization failed"
; (EnzymeAD/Enzyme.jl#3604).

define void @callee({ double, i64 } "enzyme_type"="{[0]:Float@double, [8]:Integer}" %p) {
  ret void
}

define void @caller({ double, i64 } %x) {
entry:
  call void @callee({ double, i64 } %x)
  ret void
}

; CHECK: caller - {} |{}:{}
; CHECK-NEXT: { double, i64 } %x: {[0]:Float@double, [8]:Integer}
