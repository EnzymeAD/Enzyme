; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Guard _FortranAAssign in forward mode so that an inactive destination
; descriptor (whose shadow is the primal itself, either statically or at
; runtime under enzyme_runtime_activity) is never overwritten by the shadow
; assignment.

declare void @_FortranAAssign(ptr, ptr, ptr, i32)
declare void @__enzyme_fwddiff(...)

define void @assign_fn(ptr %to, ptr %from) {
entry:
  call void @_FortranAAssign(ptr %to, ptr %from, ptr null, i32 1)
  ret void
}

define void @test_active(ptr %to, ptr %dto, ptr %from, ptr %dfrom) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @assign_fn, metadata !"enzyme_dup", ptr %to, ptr %dto, metadata !"enzyme_dup", ptr %from, ptr %dfrom)
  ret void
}

define void @test_const_to(ptr %to, ptr %from, ptr %dfrom) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @assign_fn, metadata !"enzyme_const", ptr %to, metadata !"enzyme_dup", ptr %from, ptr %dfrom)
  ret void
}

define void @test_runtime(ptr %to, ptr %dto, ptr %from, ptr %dfrom) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @assign_fn, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %to, ptr %dto, metadata !"enzyme_dup", ptr %from, ptr %dfrom)
  ret void
}

define void @test_runtime_vec(ptr %to, ptr %dto1, ptr %dto2, ptr %from, ptr %dfrom1, ptr %dfrom2) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @assign_fn, metadata !"enzyme_width", i64 2, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %to, ptr %dto1, ptr %dto2, metadata !"enzyme_dup", ptr %from, ptr %dfrom1, ptr %dfrom2)
  ret void
}

; CHECK: define internal void @fwddiffeassign_fn(ptr %to, ptr %"to'", ptr %from, ptr %"from'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @_FortranAAssign(ptr %to, ptr %from, ptr null, i32 1)
; CHECK-NEXT:   call void @_FortranAAssign(ptr %"to'", ptr %"from'", ptr null, i32 1)
; CHECK-NEXT:   ret void

; CHECK: define internal void @fwddiffeassign_fn.1(ptr %to, ptr %from, ptr %"from'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @_FortranAAssign(ptr %to, ptr %from, ptr null, i32 1)
; CHECK-NEXT:   ret void

; CHECK: define internal void @fwddiffeassign_fn.2(ptr %to, ptr %"to'", ptr %from, ptr %"from'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @_FortranAAssign(ptr %to, ptr %from, ptr null, i32 1)
; CHECK-NEXT:   %[[SAME:.+]] = icmp eq ptr %"to'", %to
; CHECK-NEXT:   %[[NULL:.+]] = icmp eq ptr %"to'", null
; CHECK-NEXT:   %[[SKIP:.+]] = or i1 %[[SAME]], %[[NULL]]
; CHECK-NEXT:   br i1 %[[SKIP]], label %[[EXIT:.+]], label %[[CHECK:.+]]
; CHECK: [[CHECK]]:
; CHECK-NEXT:   %[[BASE:.+]] = load ptr, ptr %to
; CHECK-NEXT:   %[[SBASE:.+]] = load ptr, ptr %"to'"
; CHECK-NEXT:   %[[SBASENULL:.+]] = icmp eq ptr %[[SBASE]], null
; CHECK-NEXT:   %[[DIFFBASE:.+]] = icmp ne ptr %[[SBASE]], %[[BASE]]
; CHECK-NEXT:   %[[DO:.+]] = or i1 %[[SBASENULL]], %[[DIFFBASE]]
; CHECK-NEXT:   br i1 %[[DO]], label %[[ASSIGN:.+]], label %[[EXIT]]
; CHECK: [[ASSIGN]]:
; CHECK-NEXT:   call void @_FortranAAssign(ptr %"to'", ptr %"from'", ptr null, i32 1)
; CHECK-NEXT:   br label %[[EXIT]]
; CHECK: [[EXIT]]:
; CHECK-NEXT:   ret void

; CHECK: define internal void @fwddiffe2assign_fn(ptr %to, [2 x ptr] %"to'", ptr %from, [2 x ptr] %"from'")
; CHECK:   call void @_FortranAAssign(ptr %to, ptr %from, ptr null, i32 1)
; CHECK:   icmp eq ptr %{{.+}}, %to
; CHECK:   call void @_FortranAAssign(ptr %{{.+}}, ptr %{{.+}}, ptr null, i32 1)
; CHECK:   icmp eq ptr %{{.+}}, %to
; CHECK:   call void @_FortranAAssign(ptr %{{.+}}, ptr %{{.+}}, ptr null, i32 1)
; CHECK:   ret void
