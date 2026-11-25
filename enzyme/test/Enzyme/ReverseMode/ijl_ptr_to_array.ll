; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

declare ptr addrspace(10) @ijl_ptr_to_array_1d(ptr, ptr, i64, i32)
declare void @__enzyme_autodiff(...)

define double @foo(ptr %atype, ptr %data) {
entry:
  %arr = call ptr addrspace(10) @ijl_ptr_to_array_1d(ptr %atype, ptr %data, i64 1, i32 0)
  %val = load double, ptr addrspace(10) %arr, align 8
  ret double %val
}

define void @caller(ptr %atype, ptr %data, ptr %ddata) {
entry:
  call void (...) @__enzyme_autodiff(ptr @foo, metadata !"enzyme_const", ptr %atype, metadata !"enzyme_dup", ptr %data, ptr %ddata)
  ret void
}

; CHECK: define internal void @diffefoo(ptr %atype, ptr %data, ptr %"data'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[arr:.+]] = call ptr addrspace(10) @ijl_ptr_to_array_1d(ptr %atype, ptr %"data'", i64 1, i32 0)
; CHECK-NEXT:   %arr = call ptr addrspace(10) @ijl_ptr_to_array_1d(ptr %atype, ptr %data, i64 1, i32 0)
; CHECK-NEXT:   %[[ld:.+]] = load double, ptr addrspace(10) %[[arr]], align 8
; CHECK-NEXT:   %[[add:.+]] = fadd fast double %[[ld]], %differeturn
; CHECK-NEXT:   store double %[[add]], ptr addrspace(10) %[[arr]], align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
