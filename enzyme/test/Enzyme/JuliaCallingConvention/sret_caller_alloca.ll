; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

define void @caller(double %arg, { double, {} addrspace(10)* }* %sret_box, [1 x {} addrspace(10)*]* %rroots) {
entry:
  call void @callee({ double, {} addrspace(10)* }* sret({ double, {} addrspace(10)* }) %sret_box, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %rroots, double %arg)
  ret void
}

define internal void @callee({ double, {} addrspace(10)* }* sret({ double, {} addrspace(10)* }) %0, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %1, double %2) {
top:
  %gep = getelementptr inbounds { double, {} addrspace(10)* }, { double, {} addrspace(10)* }* %0, i32 0, i32 0
  store double %2, double* %gep, align 8
  ret void
}

; CHECK-LABEL: define void @caller(double %arg, { double, {} addrspace(10)* }* %sret_box, [1 x {} addrspace(10)*]* %rroots) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %stack_sret = alloca { double, {} addrspace(10)* }, align 8
; CHECK-NEXT:   %stack_roots_AT = alloca [1 x {} addrspace(10)*], align 8
; CHECK-NEXT:   %0 = getelementptr inbounds { double, {} addrspace(10)* }, { double, {} addrspace(10)* }* %stack_sret, i64 0, i32 0
; CHECK-NEXT:   %1 = getelementptr inbounds { double, {} addrspace(10)* }, { double, {} addrspace(10)* }* %sret_box, i64 0, i32 0
; CHECK-NEXT:   %2 = load double, double* %1, align 8
; CHECK-NEXT:   store double %2, double* %0, align 8
; CHECK-NEXT:   %3 = getelementptr inbounds { double, {} addrspace(10)* }, { double, {} addrspace(10)* }* %stack_sret, i64 0, i32 1
; CHECK-NEXT:   store {} addrspace(10)* undef, {} addrspace(10)** %3, align 8
; CHECK-NEXT:   %4 = getelementptr inbounds [1 x {} addrspace(10)*], [1 x {} addrspace(10)*]* %stack_roots_AT, i64 0, i32 0
; CHECK-NEXT:   store {} addrspace(10)* undef, {} addrspace(10)** %4, align 8
; CHECK-NEXT:   call void @callee({ double, {} addrspace(10)* }* sret({ double, {} addrspace(10)* }) %stack_sret, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %stack_roots_AT, double %arg)
; CHECK-NEXT:   %5 = getelementptr inbounds { double, {} addrspace(10)* }, { double, {} addrspace(10)* }* %sret_box, i64 0, i32 0
; CHECK-NEXT:   %6 = getelementptr inbounds { double, {} addrspace(10)* }, { double, {} addrspace(10)* }* %stack_sret, i64 0, i32 0
; CHECK-NEXT:   %7 = load double, double* %6, align 8
; CHECK-NEXT:   store double %7, double* %5, align 8
; CHECK-NEXT:   %8 = load [1 x {} addrspace(10)*], [1 x {} addrspace(10)*]* %stack_roots_AT, align 8
; CHECK-NEXT:   store [1 x {} addrspace(10)*] %8, [1 x {} addrspace(10)*]* %rroots, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK-LABEL: define internal void @callee({ double, {} addrspace(10)* }* noalias sret({ double, {} addrspace(10)* }) %0, [1 x {} addrspace(10)*]* noalias writeonly "enzymejl_returnRoots"="1" %1, double %2) {
; CHECK-NEXT: top:
; CHECK-NEXT:   %gep = getelementptr inbounds { double, {} addrspace(10)* }, { double, {} addrspace(10)* }* %0, i32 0, i32 0
; CHECK-NEXT:   store double %2, double* %gep, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
