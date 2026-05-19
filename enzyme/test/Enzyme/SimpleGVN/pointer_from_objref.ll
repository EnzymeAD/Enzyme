; RUN: %opt < %s %newLoadEnzyme -passes="simple-gvn" -S -opaque-pointers | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-ni:10:11:12:13"

define ptr @test_pointer_from_objref(ptr %ptr, ptr addrspace(10) %val) {
entry:
  store ptr addrspace(10) %val, ptr %ptr
  %loaded = load ptr, ptr %ptr
  ret ptr %loaded
}

; CHECK: define ptr @test_pointer_from_objref(ptr %ptr, ptr addrspace(10) %val) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   store ptr addrspace(10) %val, ptr %ptr, align 8
; CHECK-NEXT:   [[DERIVED:%.+]] = addrspacecast ptr addrspace(10) %val to ptr addrspace(11)
; CHECK-NEXT:   [[RAW:%.+]] = call ptr @julia.pointer_from_objref(ptr addrspace(11) [[DERIVED]])
; CHECK-NEXT:   ret ptr [[RAW]]
; CHECK-NEXT: }
