; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"struct.std::__detail::_Hash_node" = type { %"struct.std::__detail::_Hash_node_base", %"struct.std::__detail::_Hash_node_value" }
%"struct.std::__detail::_Hash_node_base" = type { %"struct.std::__detail::_Hash_node_base"* }
%"struct.std::__detail::_Hash_node_value" = type { %"struct.std::__detail::_Hash_node_value_base", %"struct.std::__detail::_Hash_node_code_cache" }
%"struct.std::__detail::_Hash_node_value_base" = type { %"struct.__gnu_cxx::__aligned_buffer" }
%"struct.__gnu_cxx::__aligned_buffer" = type { %"union.std::aligned_storage<40, 8>::type" }
%"union.std::aligned_storage<40, 8>::type" = type { [40 x i8] }
%"struct.std::__detail::_Hash_node_code_cache" = type { i64 }

declare dso_local void @_Z17__enzyme_autodiff(...) local_unnamed_addr

define void @caller(i8* %vals, i8* %dvals) local_unnamed_addr {
  call void (...) @_Z17__enzyme_autodiff(i8* bitcast (void (%"struct.std::__detail::_Hash_node"*, i1)* @fn to i8*), metadata !"enzyme_dup", i8* %vals, i8* %dvals, i1 true)
  ret void
}

define linkonce_odr dso_local void @fn(%"struct.std::__detail::_Hash_node"* %a4, i1 %cmp.i) {
  br i1 %cmp.i, label %land.rhs.i, label %if.end3.i.preheader

land.rhs.i:                                       ; preds = %entry
  %_M_string_length.i36 = getelementptr inbounds %"struct.std::__detail::_Hash_node", %"struct.std::__detail::_Hash_node"* %a4, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i64 8
  br label %if.end3.i.preheader

if.end3.i.preheader:                              ; preds = %entry, %land.rhs.i
  br label %if.end3.i

if.end3.i:                                        ; preds = %if.end3.i.preheader, %if.end3.i
  %__p.023.i = phi %"struct.std::__detail::_Hash_node"* [ %a21, %if.end3.i ], [ %a4, %if.end3.i.preheader ]
  %a20 = bitcast %"struct.std::__detail::_Hash_node"* %__p.023.i to %"struct.std::__detail::_Hash_node"**
  %a21 = load %"struct.std::__detail::_Hash_node"*, %"struct.std::__detail::_Hash_node"** %a20, align 8
  %_M_hash_code.i59 = bitcast %"struct.std::__detail::_Hash_node"* %a21 to i64*
  %a23 = load i64, i64* %_M_hash_code.i59, align 8
  %cmp.i60 = icmp eq i64 %a23, 32
  br i1 %cmp.i60, label %exit, label %if.end3.i

exit:                                             ; preds = %if.end3.i
  ret void
}

; CHECK: define internal void @diffefn(%"struct.std::__detail::_Hash_node"* %a4, %"struct.std::__detail::_Hash_node"* %"a4'", i1 %cmp.i)
; CHECK-NEXT:   %"iv'ac" = alloca i64
; CHECK-NEXT:   br i1 %cmp.i, label %land.rhs.i, label %if.end3.i.preheader

; CHECK: land.rhs.i:                                       ; preds = %0
; CHECK-NEXT:   br label %if.end3.i.preheader

; CHECK: if.end3.i.preheader:                              ; preds = %land.rhs.i, %0
; CHECK-NEXT:   br label %if.end3.i

; CHECK: if.end3.i:                                        ; preds = %if.end3.i, %if.end3.i.preheader
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %if.end3.i ], [ 0, %if.end3.i.preheader ]
; CHECK-NEXT:   %__p.023.i = phi %"struct.std::__detail::_Hash_node"* [ %a21, %if.end3.i ], [ %a4, %if.end3.i.preheader ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %a20 = bitcast %"struct.std::__detail::_Hash_node"* %__p.023.i to %"struct.std::__detail::_Hash_node"**
; CHECK-NEXT:   %a21 = load %"struct.std::__detail::_Hash_node"*, %"struct.std::__detail::_Hash_node"** %a20, align 8
; CHECK-NEXT:   %_M_hash_code.i59 = bitcast %"struct.std::__detail::_Hash_node"* %a21 to i64*
; CHECK-NEXT:   %a23 = load i64, i64* %_M_hash_code.i59, align 8
; CHECK-NEXT:   %cmp.i60 = icmp eq i64 %a23, 32
; CHECK-NEXT:   br i1 %cmp.i60, label %exit, label %if.end3.i

; CHECK: exit:                                             ; preds = %if.end3.i
; CHECK-NEXT:   br label %invertexit

; CHECK: invert:                                           ; preds = %invertif.end3.i.preheader, %invertland.rhs.i
; CHECK-NEXT:   ret void

; CHECK: invertland.rhs.i:                                 ; preds = %invertif.end3.i.preheader
; CHECK-NEXT:   br label %invert

; CHECK: invertif.end3.i.preheader:                        ; preds = %invertif.end3.i
; CHECK-NEXT:   br i1 %cmp.i, label %invertland.rhs.i, label %invert

; CHECK: invertif.end3.i:                                  ; preds = %mergeinvertif.end3.i_exit, %incinvertif.end3.i
; CHECK-NEXT:   %1 = load i64, i64* %"iv'ac"
; CHECK-NEXT:   %2 = icmp eq i64 %1, 0
; CHECK-NEXT:   %3 = xor i1 %2, true
; CHECK-NEXT:   br i1 %2, label %invertif.end3.i.preheader, label %incinvertif.end3.i

; CHECK: incinvertif.end3.i:                               ; preds = %invertif.end3.i
; CHECK-NEXT:   %4 = load i64, i64* %"iv'ac"
; CHECK-NEXT:   %5 = add nsw i64 %4, -1
; CHECK-NEXT:   store i64 %5, i64* %"iv'ac"
; CHECK-NEXT:   br label %invertif.end3.i

; CHECK: invertexit:                                       ; preds = %exit
; CHECK-NEXT:   br label %mergeinvertif.end3.i_exit

; CHECK: mergeinvertif.end3.i_exit:                        ; preds = %invertexit
; CHECK-NEXT:   store i64 0, i64* %"iv'ac"
; CHECK-NEXT:   br label %invertif.end3.i
; CHECK-NEXT: }
