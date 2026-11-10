; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=store_product_ptr -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=store_product_ptr -disable-output | FileCheck %s

%struct.Product = type { ptr, ptr }

define ptr @product_rhs(ptr %this) {
entry:
  %slot = alloca ptr, align 8
  store ptr %this, ptr %slot, align 8
  %loaded_this = load ptr, ptr %slot, align 8
  %m_rhs = getelementptr inbounds %struct.Product, ptr %loaded_this, i32 0, i32 1
  %rhs = load ptr, ptr %m_rhs, align 8
  ret ptr %rhs
}

define ptr @load_implicit_ptr(ptr %slot) {
entry:
  %ptr_val = load ptr, ptr %slot, align 8
  ret ptr %ptr_val
}

define double @store_product_ptr(ptr "enzyme_type"="{[-1]:Pointer}" %lhs_mat, ptr %rhs_mat, ptr %implicit_slot) {
entry:
  %prod = alloca %struct.Product, align 8
  %lhs_slot = getelementptr inbounds %struct.Product, ptr %prod, i32 0, i32 0
  store ptr %lhs_mat, ptr %lhs_slot, align 8
  %rhs_slot = getelementptr inbounds %struct.Product, ptr %prod, i32 0, i32 1
  store ptr %rhs_mat, ptr %rhs_slot, align 8
  %rhs_val = load double, ptr %rhs_mat, align 8
  %got_rhs = call ptr @product_rhs(ptr %prod)
  %got_implicit = call "enzyme_type"="{[-1,0]:Float@double, [-1,8]:Float@double}" ptr @load_implicit_ptr(ptr %implicit_slot)
  ret double %rhs_val
}

; CHECK: product_rhs - {[-1]:Pointer, [-1,0]:Float@double} |{[-1]:Pointer, [-1,-1]:Pointer, [-1,8,0]:Float@double}:{}
; CHECK-NEXT: ptr %this: {[-1]:Pointer, [-1,-1]:Pointer, [-1,8,0]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %slot = alloca ptr, align 8: {[-1]:Pointer, [-1,-1]:Pointer, [-1,-1,0]:Pointer, [-1,-1,8]:Pointer, [-1,-1,8,0]:Float@double}
; CHECK-NEXT:   store ptr %this, ptr %slot, align 8: {}
; CHECK-NEXT:   %loaded_this = load ptr, ptr %slot, align 8: {[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Pointer, [-1,8,0]:Float@double}
; CHECK-NEXT:   %m_rhs = getelementptr inbounds %struct.Product, ptr %loaded_this, i32 0, i32 1: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Float@double}
; CHECK-NEXT:   %rhs = load ptr, ptr %m_rhs, align 8: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   ret ptr %rhs: {}

; CHECK: load_implicit_ptr - {[-1,0]:Float@double, [-1,8]:Float@double} |{[-1]:Pointer}:{}
; CHECK-NEXT: ptr %slot: {[-1]:Pointer, [-1,0,0]:Float@double, [-1,0,8]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %ptr_val = load ptr, ptr %slot, align 8: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}
; CHECK-NEXT:   ret ptr %ptr_val: {}

; CHECK: store_product_ptr - {[-1]:Float@double} |{[-1]:Pointer}:{} {[-1]:Pointer, [-1,0]:Float@double}:{} {[-1]:Pointer, [-1,0,0]:Float@double, [-1,0,8]:Float@double}:{}
; CHECK-NEXT: ptr %lhs_mat: {[-1]:Pointer}
; CHECK-NEXT: ptr %rhs_mat: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT: ptr %implicit_slot: {[-1]:Pointer, [-1,0,0]:Float@double, [-1,0,8]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %prod = alloca %struct.Product, align 8: {[-1]:Pointer, [-1,-1]:Pointer, [-1,8,0]:Float@double}
; CHECK-NEXT:   %lhs_slot = getelementptr inbounds %struct.Product, ptr %prod, i32 0, i32 0: {[-1]:Pointer, [-1,-1]:Pointer}
; CHECK-NEXT:   store ptr %lhs_mat, ptr %lhs_slot, align 8: {}
; CHECK-NEXT:   %rhs_slot = getelementptr inbounds %struct.Product, ptr %prod, i32 0, i32 1: {[-1]:Pointer, [-1,-1]:Pointer, [-1,0,0]:Float@double}
; CHECK-NEXT:   store ptr %rhs_mat, ptr %rhs_slot, align 8: {}
; CHECK-NEXT:   %rhs_val = load double, ptr %rhs_mat, align 8: {[-1]:Float@double}
; CHECK-NEXT:   %got_rhs = call ptr @product_rhs(ptr %prod): {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %got_implicit = call "enzyme_type"="{[-1,0]:Float@double, [-1,8]:Float@double}" ptr @load_implicit_ptr(ptr %implicit_slot): {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}
; CHECK-NEXT:   ret double %rhs_val: {}
