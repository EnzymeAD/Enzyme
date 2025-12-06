; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=foo -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=foo -S -o /dev/null | FileCheck %s

; Test that GEP with null pointer does not cause "Illegal updateAnalysis" error
; This test is based on the issue: https://fwd.gymni.ch/IpbFyC
; The fix prevents propagating pointer type to GEP result when input is null

define void @foo(i64 %offset) {
entry:
  %gep = getelementptr i8, ptr null, i64 %offset
  ret void
}

; CHECK: foo - {} |{}:{}
; CHECK-NEXT: i64 %offset: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %gep = getelementptr i8, ptr null, i64 %offset: {[-1]:Pointer, [-1,-1]:Anything}
; CHECK-NEXT:   ret void: {}
