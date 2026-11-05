; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="type-hints-to-values" -S | FileCheck %s; fi

; A load gets the hinted types of the bytes it reads, at its own offsets.

declare void @__enzyme_double(ptr, i64)
declare void @__enzyme_float(ptr, i64)

; The load starts 8 bytes before the hinted double.
define i128 @before(i128 %x) {
entry:
  %a = alloca [3 x i64], align 8
  store i128 %x, ptr %a, align 8
  %b = getelementptr inbounds i8, ptr %a, i64 8
  call void @__enzyme_double(ptr %b, i64 8)
  %v = load i128, ptr %a, align 8
  ret i128 %v
}

; CHECK-LABEL: define i128 @before(
; CHECK: call i128 @__enzyme_typed_value.i128(i128 %v), !enzyme_type ![[BEFORE:[0-9]+]]

; The hint covers two floats and the load reads the second.
define float @inside(<2 x float> %x) {
entry:
  %a = alloca <2 x float>, align 8
  store <2 x float> %x, ptr %a, align 8
  call void @__enzyme_float(ptr %a, i64 8)
  %b = getelementptr inbounds i8, ptr %a, i64 4
  %v = load float, ptr %b, align 4
  ret float %v
}

; CHECK-LABEL: define float @inside(
; CHECK: call float @__enzyme_typed_value.float(float %v), !enzyme_type ![[FLOAT:[0-9]+]]

; CHECK: ![[BEFORE]] = !{!"Unknown", i32 8, ![[DOUBLE:[0-9]+]]}
; CHECK: ![[DOUBLE]] = !{!"Float@double"}
; CHECK: ![[FLOAT]] = !{!"Unknown", i32 -1, ![[FLOAT1:[0-9]+]]}
; CHECK: ![[FLOAT1]] = !{!"Float@float"}
