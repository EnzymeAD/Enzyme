; RUN: %opt %s %loadPoseidon -load-pass-plugin=%poseidon_test_programs/poseidon-test-model%shlibext -passes=precision-change-types -disable-output | FileCheck %s

; getPrecisionChangeType is the inverse of getLLVMFPType on the formats that
; are LLVM types.

; CHECK: BF16: bfloat -> BF16
; CHECK-NEXT: FP16: half -> FP16
; CHECK-NEXT: FP32: float -> FP32
; CHECK-NEXT: FP64: double -> FP64
; CHECK-NEXT: FP80: x86_fp80 -> FP80
; CHECK-NEXT: FP128: fp128 -> FP128
