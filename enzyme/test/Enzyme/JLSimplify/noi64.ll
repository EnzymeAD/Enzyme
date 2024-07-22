; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -jl-inst-simplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="jl-inst-simplify" -S | FileCheck %s

declare i64* @malloc(i64)

define fastcc i1 @augmented_julia__affine_normalize_1484() {
  %i5 = call noalias i64* @malloc(i64 16)
  %i29 = load i64, i64* %i5, align 8
  %i31 = call noalias nonnull i64 addrspace(10)* inttoptr (i64 137352001798896 to i64 addrspace(10)* ({} addrspace(10)*, i64, i64)*)({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 137351863426640 to {}*) to {} addrspace(10)*), i64 10, i64 10) 
  %i35 = load i64, i64 addrspace(10)* %i31, align 8
  %i39 = icmp ne i64 %i35, %i29
  ret i1 %i39
}

; CHECK: define fastcc i1 @augmented_julia__affine_normalize_1484() {
; CHECK-NEXT:   %i5 = call noalias i64* @malloc(i64 16)
; CHECK-NEXT:   %i29 = load i64, i64* %i5, align 8
; CHECK-NEXT:   %i31 = call noalias nonnull i64 addrspace(10)* inttoptr (i64 137352001798896 to i64 addrspace(10)* ({} addrspace(10)*, i64, i64)*)({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 137351863426640 to {}*) to {} addrspace(10)*), i64 10, i64 10)
; CHECK-NEXT:   %i35 = load i64, i64 addrspace(10)* %i31, align 8
; CHECK-NEXT:   %i39 = icmp ne i64 %i35, %i29
; CHECK-NEXT:   ret i1 %i39
; CHECK-NEXT: }
