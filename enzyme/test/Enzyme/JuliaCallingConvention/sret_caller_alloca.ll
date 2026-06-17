; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

define void @caller(double %arg, { double, i64 }* %sret_box, [1 x {} addrspace(10)*]* %rroots) {
entry:
  call void @callee({ double, i64 }* sret({ double, i64 }) %sret_box, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %rroots, double %arg)
  ret void
}

define internal void @callee({ double, i64 }* sret({ double, i64 }) %0, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %1, double %2) {
top:
  %gep = getelementptr inbounds { double, i64 }, { double, i64 }* %0, i32 0, i32 0
  store double %2, double* %gep, align 8
  ret void
}

; CHECK-LABEL: define void @caller(double %arg, { double, i64 }* %sret_box, [1 x {} addrspace(10)*]* %rroots) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %stack_sret = alloca { double, i64 }, align 8
; CHECK-NEXT:   %stack_roots_AT = alloca [1 x {} addrspace(10)*], align 8
; CHECK-NEXT:   %0 = getelementptr inbounds { double, i64 }, { double, i64 }* %sret_box, i64 0, i32 0
; CHECK-NEXT:   %1 = getelementptr inbounds { double, i64 }, { double, i64 }* %stack_sret, i64 0, i32 0
; CHECK-NEXT:   %2 = load double, double* %0, align 8
; CHECK-NEXT:   store double %2, double* %1, align 8
; CHECK-NEXT:   %3 = getelementptr inbounds { double, i64 }, { double, i64 }* %sret_box, i64 0, i32 1
; CHECK-NEXT:   %4 = getelementptr inbounds { double, i64 }, { double, i64 }* %stack_sret, i64 0, i32 1
; CHECK-NEXT:   %5 = load i64, i64* %3, align 4
; CHECK-NEXT:   store i64 %5, i64* %4, align 4
; CHECK-NEXT:   call void @callee({ double, i64 }* sret({ double, i64 }) %stack_sret, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %stack_roots_AT, double %arg)
; CHECK-NEXT:   %6 = getelementptr inbounds { double, i64 }, { double, i64 }* %sret_box, i64 0, i32 0
; CHECK-NEXT:   %7 = getelementptr inbounds { double, i64 }, { double, i64 }* %stack_sret, i64 0, i32 0
; CHECK-NEXT:   %8 = load double, double* %7, align 8
; CHECK-NEXT:   store double %8, double* %6, align 8
; CHECK-NEXT:   %9 = getelementptr inbounds { double, i64 }, { double, i64 }* %sret_box, i64 0, i32 1
; CHECK-NEXT:   %10 = getelementptr inbounds { double, i64 }, { double, i64 }* %stack_sret, i64 0, i32 1
; CHECK-NEXT:   %11 = load i64, i64* %10, align 4
; CHECK-NEXT:   store i64 %11, i64* %9, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK-LABEL: define internal void @callee({ double, i64 }* noalias sret({ double, i64 }) %0, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %1, double %2) {
; CHECK-NEXT: top:
; CHECK-NEXT:   %gep = getelementptr inbounds { double, i64 }, { double, i64 }* %0, i32 0, i32 0
; CHECK-NEXT:   store double %2, double* %gep, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
