; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -jl-inst-simplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="jl-inst-simplify" -S | FileCheck %s

declare noalias nonnull ptr @jl_alloc_genericmemory(ptr, i64) #0

define i1 @test_genericmemory_owner(ptr %type, i64 %len) {
top:
  %mem = call noalias nonnull ptr @jl_alloc_genericmemory(ptr %type, i64 %len)
  %mem_ptr_slot = bitcast ptr %mem to ptr
  %mem_ptr = load ptr, ptr %mem_ptr_slot, align 8
  %mem_ptr_cast = bitcast ptr %mem_ptr to ptr
  %standalone_ptr = getelementptr inbounds ptr, ptr %mem_ptr_slot, i64 2
  %standalone_cast = bitcast ptr %standalone_ptr to ptr
  %cmp = icmp eq ptr %mem_ptr_cast, %standalone_cast
  ret i1 %cmp
}

attributes #0 = { "enzyme_allocation" }

; CHECK: define i1 @test_genericmemory_owner(ptr %type, i64 %len)
; CHECK-NEXT: top:
; CHECK-NEXT:   %mem = call noalias nonnull ptr @jl_alloc_genericmemory(ptr %type, i64 %len)
; CHECK-NEXT:   %mem_ptr_slot = bitcast ptr %mem to ptr
; CHECK-NEXT:   %mem_ptr = load ptr, ptr %mem_ptr_slot, align 8
; CHECK-NEXT:   %mem_ptr_cast = bitcast ptr %mem_ptr to ptr
; CHECK-NEXT:   %standalone_ptr = getelementptr inbounds ptr, ptr %mem_ptr_slot, i64 2
; CHECK-NEXT:   %standalone_cast = bitcast ptr %standalone_ptr to ptr
; CHECK-NEXT:   %cmp = icmp eq ptr %mem_ptr_cast, %standalone_cast
; CHECK-NEXT:   ret i1 %cmp
