; RUN: %opt < %s %newLoadEnzyme -passes="simple-gvn" -S | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-ni:10:11:12:13"

define i8* @test_pointer_from_objref(i8** %ptr, {} addrspace(10)* %val) {
entry:
  %c = bitcast i8** %ptr to {} addrspace(10)**
  store {} addrspace(10)* %val, {} addrspace(10)** %c
  %loaded = load i8*, i8** %ptr
  ret i8* %loaded
}

; CHECK: define i8* @test_pointer_from_objref(i8** %ptr, {} addrspace(10)* %val) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %c = bitcast i8** %ptr to {} addrspace(10)**
; CHECK-NEXT:   store {} addrspace(10)* %val, {} addrspace(10)** %c, align 8
; CHECK-NEXT:   [[DERIVED:%.+]] = addrspacecast {} addrspace(10)* %val to {} addrspace(11)*
; CHECK-NEXT:   [[RAW:%.+]] = call {}* @julia.pointer_from_objref({} addrspace(11)* [[DERIVED]])
; CHECK-NEXT:   [[FINAL:%.+]] = bitcast {}* [[RAW]] to i8*
; CHECK-NEXT:   ret i8* [[FINAL]]
; CHECK-NEXT: }
