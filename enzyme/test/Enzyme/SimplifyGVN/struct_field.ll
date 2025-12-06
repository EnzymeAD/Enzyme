; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -simplify-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="simplify-gvn" -S | FileCheck %s

; Test store-to-load forwarding with struct field access

%struct.Point = type { i32, i32 }

define i32 @test_struct_field(%struct.Point* noalias nocapture %point) {
entry:
  %x_ptr = getelementptr %struct.Point, %struct.Point* %point, i64 0, i32 0
  %y_ptr = getelementptr %struct.Point, %struct.Point* %point, i64 0, i32 1
  store i32 10, i32* %x_ptr, align 4
  store i32 20, i32* %y_ptr, align 4
  %x_val = load i32, i32* %x_ptr, align 4
  %y_val = load i32, i32* %y_ptr, align 4
  %sum = add i32 %x_val, %y_val
  ret i32 %sum
}

; CHECK: define i32 @test_struct_field(%struct.Point* noalias nocapture %point)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %x_ptr = getelementptr %struct.Point, %struct.Point* %point, i64 0, i32 0
; CHECK-NEXT:   %y_ptr = getelementptr %struct.Point, %struct.Point* %point, i64 0, i32 1
; CHECK-NEXT:   store i32 10, i32* %x_ptr, align 4
; CHECK-NEXT:   store i32 20, i32* %y_ptr, align 4
; CHECK-NEXT:   %sum = add i32 10, 20
; CHECK-NEXT:   ret i32 %sum
