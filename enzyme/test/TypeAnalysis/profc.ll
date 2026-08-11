; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | %FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | %FileCheck %s

@__profc_foo = internal global [1 x i64] zeroinitializer, align 8

define void @callee() {
entry:
  %ld = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_foo, i64 0, i64 0), align 8
  ret void
}

; CHECK: callee - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %ld = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_foo, i64 0, i64 0), align 8: {[-1]:Integer}
; CHECK-NEXT:   ret void: {}
