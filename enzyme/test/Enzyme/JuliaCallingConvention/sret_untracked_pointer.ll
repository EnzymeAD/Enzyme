; RUN: if [ %llvmver -ge 15 ]; then %opt %OPnewLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s; fi

; The sret type has a pointer that the GC does not track, as the data pointer
; of a GenericMemoryRef of Julia. The caller does not give an alloca as sret.
; Thus the pass must copy the pointer into the merged sret before the call and
; back into the memory of the caller after the call.

define void @caller(double %arg, ptr %sret_box) {
entry:
  %res = call ptr @pointer_callee(ptr writeonly "enzyme_sret"="test_type7" %sret_box, double %arg)
  ret void
}

define internal fastcc ptr @pointer_callee(ptr noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(16) "enzyme_sret"="test_type7" %0, double %1) {
top:
  %mem = call ptr @malloc(i64 8)
  store double %1, ptr %mem, align 8
  store ptr %mem, ptr %0, align 8
  ret ptr null
}

declare ptr @malloc(i64)

; CHECK-LABEL: define void @caller(double %arg, ptr %sret_box) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %stack_sret = alloca { ptr, { ptr, i64 } }, align 8
; CHECK-NEXT:   %0 = getelementptr inbounds { ptr, { ptr, i64 } }, ptr %stack_sret, i32 0, i32 1
; CHECK-NEXT:   %1 = getelementptr inbounds { ptr, i64 }, ptr %0, i64 0, i32 0
; CHECK-NEXT:   %2 = getelementptr inbounds { ptr, i64 }, ptr %sret_box, i64 0, i32 0
; CHECK-NEXT:   %3 = load ptr, ptr %2, align 8
; CHECK-NEXT:   store ptr %3, ptr %1, align 8
; CHECK-NEXT:   %4 = getelementptr inbounds { ptr, i64 }, ptr %0, i64 0, i32 1
; CHECK-NEXT:   %5 = getelementptr inbounds { ptr, i64 }, ptr %sret_box, i64 0, i32 1
; CHECK-NEXT:   %6 = load i64, ptr %5, align 4
; CHECK-NEXT:   store i64 %6, ptr %4, align 4
; CHECK-NEXT:   call void @pointer_callee(ptr sret({ ptr, { ptr, i64 } }) %stack_sret, double %arg)
; CHECK-NEXT:   %7 = getelementptr inbounds { ptr, { ptr, i64 } }, ptr %stack_sret, i32 0, i32 0
; CHECK-NEXT:   %res = load ptr, ptr %7, align 8
; CHECK-NEXT:   %8 = getelementptr inbounds { ptr, i64 }, ptr %sret_box, i64 0, i32 0
; CHECK-NEXT:   %9 = getelementptr inbounds { ptr, i64 }, ptr %0, i64 0, i32 0
; CHECK-NEXT:   %10 = load ptr, ptr %9, align 8
; CHECK-NEXT:   store ptr %10, ptr %8, align 8
; CHECK-NEXT:   %11 = getelementptr inbounds { ptr, i64 }, ptr %sret_box, i64 0, i32 1
; CHECK-NEXT:   %12 = getelementptr inbounds { ptr, i64 }, ptr %0, i64 0, i32 1
; CHECK-NEXT:   %13 = load i64, ptr %12, align 4
; CHECK-NEXT:   store i64 %13, ptr %11, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
