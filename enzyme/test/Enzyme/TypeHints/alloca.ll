; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="type-hints-to-values" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="type-hints-to-values,sroa" -S | FileCheck %s --check-prefix=SROA; fi

; A by-value closure {ptr data, i64 stride} with an integer hint on its stride.
; The hint is the only use that keeps SROA from promoting the closure.

declare void @__enzyme_integer(ptr, i64)

define double @read(ptr %data, i64 %stride, i64 %i) {
entry:
  %c = alloca { ptr, i64 }, align 8
  store ptr %data, ptr %c, align 8
  %sp = getelementptr inbounds i8, ptr %c, i64 8
  store i64 %stride, ptr %sp, align 8
  call void @__enzyme_integer(ptr %sp, i64 8)
  %d = load ptr, ptr %c, align 8
  %s = load i64, ptr %sp, align 8
  %idx = mul i64 %i, %s
  %p = getelementptr double, ptr %d, i64 %idx
  %v = load double, ptr %p, align 8
  ret double %v
}

; CHECK-LABEL: define double @read(
; CHECK-NOT: @__enzyme_integer(
; CHECK:      %d = load ptr, ptr %c, align 8
; CHECK-NEXT: %s = load i64, ptr %sp, align 8
; CHECK-NEXT: %[[T:.+]] = call i64 @__enzyme_typed_value.i64(i64 %s), !enzyme_type ![[INT:[0-9]+]]
; CHECK-NEXT: %idx = mul i64 %i, %[[T]]

; CHECK: declare i64 @__enzyme_typed_value.i64(i64) #[[ATTR:[0-9]+]]
; CHECK: attributes #[[ATTR]] = { nofree nosync nounwind speculatable willreturn memory(none) }
; CHECK: ![[INT]] = !{!"Unknown", i32 -1, ![[INT1:[0-9]+]]}
; CHECK: ![[INT1]] = !{!"Integer"}

; SROA-LABEL: define double @read(
; SROA-NOT: alloca
; SROA:      %[[T:.+]] = call i64 @__enzyme_typed_value.i64(i64 %stride), !enzyme_type
; SROA-NEXT: %idx = mul i64 %i, %[[T]]
