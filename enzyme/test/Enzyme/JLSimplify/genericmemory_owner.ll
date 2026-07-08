; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -jl-inst-simplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="jl-inst-simplify" -S | FileCheck %s

declare noalias nonnull {}* @jl_alloc_genericmemory({}*, i64) "enzyme_allocation"

define i1 @test_genericmemory_owner({}* %type, i64 %len) {
top:
  %mem = call noalias nonnull {}* @jl_alloc_genericmemory({}* %type, i64 %len)
  %mem_ptr_slot = bitcast {}* %mem to {}**
  %mem_ptr = load {}*, {}** %mem_ptr_slot, align 8
  %mem_ptr_cast = bitcast {}* %mem_ptr to i8*
  %standalone_ptr = getelementptr inbounds {}*, {}** %mem_ptr_slot, i64 2
  %standalone_cast = bitcast {}** %standalone_ptr to i8*
  %cmp = icmp eq i8* %mem_ptr_cast, %standalone_cast
  ret i1 %cmp
}

; CHECK: define i1 @test_genericmemory_owner({}* %type, i64 %len)
; CHECK-NEXT: top:
; CHECK-NEXT:   %mem = call noalias nonnull {}* @jl_alloc_genericmemory({}* %type, i64 %len)
; CHECK-NEXT:   %mem_ptr_slot = bitcast {}* %mem to {}**
; CHECK-NEXT:   %mem_ptr = load {}*, {}** %mem_ptr_slot, align 8
; CHECK-NEXT:   %mem_ptr_cast = bitcast {}* %mem_ptr to i8*
; CHECK-NEXT:   %standalone_ptr = getelementptr inbounds {}*, {}** %mem_ptr_slot, i64 2
; CHECK-NEXT:   %standalone_cast = bitcast {}** %standalone_ptr to i8*
; CHECK-NEXT:   %cmp = icmp eq i8* %mem_ptr_cast, %standalone_cast
; CHECK-NEXT:   ret i1 %cmp
