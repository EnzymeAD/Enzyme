; RUN: %opt < %s %newLoadEnzyme -passes="simple-gvn" -S | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-ni:10:11:12:13"

define {} addrspace(10)* @test_int_to_gc_ptr(i64* noalias %ptr, i64 %val) {
entry:
  store i64 %val, i64* %ptr
  %c = bitcast i64* %ptr to {} addrspace(10)**
  %loaded = load {} addrspace(10)*, {} addrspace(10)** %c
  ret {} addrspace(10)* %loaded
}

; CHECK: define {} addrspace(10)* @test_int_to_gc_ptr(i64* noalias %ptr, i64 %val) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i64 %val, i64* %ptr, align 8
; CHECK-NEXT:   %c = bitcast i64* %ptr to {} addrspace(10)**
; CHECK-NEXT:   %[[INTTOPTR:.+]] = inttoptr i64 %val to {}*
; CHECK-NEXT:   %[[ADDRSPACECAST:.+]] = addrspacecast {}* %[[INTTOPTR]] to {} addrspace(10)*
; CHECK-NEXT:   ret {} addrspace(10)* %[[ADDRSPACECAST]]
; CHECK-NEXT: }
