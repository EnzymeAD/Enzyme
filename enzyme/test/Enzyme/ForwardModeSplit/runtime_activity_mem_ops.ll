; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

declare ptr @__enzyme_augmentfwd(...)
declare void @__enzyme_fwdsplit(...)

define void @store_float(ptr nocapture writeonly %dst, double %x) noinline {
entry:
  store double %x, ptr %dst, align 8
  ret void
}

define void @atomic_fadd_void(ptr %dst, double %x) noinline {
entry:
  %old = atomicrmw fadd ptr %dst, double %x monotonic, align 8
  ret void
}

define void @atomic_fmax_void(ptr %dst, double %x) noinline {
entry:
  %old = atomicrmw fmax ptr %dst, double %x monotonic, align 8
  ret void
}

define void @test_store_float(ptr %p, ptr %dp, double %x, double %dx) {
entry:
  %t = call ptr (...) @__enzyme_augmentfwd(ptr @store_float, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp, metadata !"enzyme_dup", double %x, double %dx)
  call void (...) @__enzyme_fwdsplit(ptr @store_float, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp, metadata !"enzyme_dup", double %x, double %dx, ptr %t)
  ret void
}

define void @test_atomic_fadd_void(ptr %p, ptr %dp, double %x, double %dx) {
entry:
  %t = call ptr (...) @__enzyme_augmentfwd(ptr @atomic_fadd_void, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp, metadata !"enzyme_dup", double %x, double %dx)
  call void (...) @__enzyme_fwdsplit(ptr @atomic_fadd_void, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp, metadata !"enzyme_dup", double %x, double %dx, ptr %t)
  ret void
}

define void @test_atomic_fmax_void(ptr %p, ptr %dp, double %x, double %dx) {
entry:
  %t = call ptr (...) @__enzyme_augmentfwd(ptr @atomic_fmax_void, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp, metadata !"enzyme_dup", double %x, double %dx)
  call void (...) @__enzyme_fwdsplit(ptr @atomic_fmax_void, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp, metadata !"enzyme_dup", double %x, double %dx, ptr %t)
  ret void
}

; CHECK-LABEL: define internal ptr @augmented_store_float(ptr nocapture writeonly %dst, ptr nocapture %"dst'", double %x, double %"x'")
; CHECK:   store double %x, ptr %dst, align 8
; CHECK-NOT: store double %"x'"
; CHECK:   ret ptr

; CHECK-LABEL: define internal void @fwddiffestore_float(ptr nocapture writeonly %dst, ptr nocapture %"dst'", double %x, double %"x'", ptr %tapeArg)
; CHECK:   %[[CMP:.+]] = icmp ne ptr %dst, %"dst'"
; CHECK:   br i1 %[[CMP]], label %[[THEN:.+]], label %[[CONT:.+]]
; CHECK: [[THEN]]:
; CHECK:   store double %"x'", ptr %"dst'", align 8
; CHECK:   br label %[[CONT]]
; CHECK: [[CONT]]:
; CHECK-NOT: store double %x, ptr %dst
; CHECK:   ret void

; CHECK-LABEL: define internal ptr @augmented_atomic_fadd_void(ptr %dst, ptr %"dst'", double %x, double %"x'")
; CHECK:   %old = atomicrmw fadd ptr %dst, double %x monotonic, align 8
; CHECK-NOT: atomicrmw fadd ptr %"dst'"
; CHECK:   ret ptr

; CHECK-LABEL: define internal void @fwddiffeatomic_fadd_void(ptr %dst, ptr %"dst'", double %x, double %"x'", ptr %tapeArg)
; CHECK-NOT: atomicrmw fadd ptr %dst
; CHECK:   %[[CMP:.+]] = icmp ne ptr %dst, %"dst'"
; CHECK:   br i1 %[[CMP]], label %[[THEN:.+]], label %[[CONT:.+]]
; CHECK: [[THEN]]:
; CHECK:   atomicrmw fadd ptr %"dst'", double %"x'" monotonic, align 8
; CHECK:   br label %[[CONT]]
; CHECK: [[CONT]]:
; CHECK-NOT: atomicrmw fadd ptr %dst
; CHECK:   ret void

; CHECK-LABEL: define internal ptr @augmented_atomic_fmax_void(ptr %dst, ptr %"dst'", double %x, double %"x'")
; CHECK:   cmpxchg ptr %dst
; CHECK:   ret ptr

; CHECK-LABEL: define internal void @fwddiffeatomic_fmax_void(ptr %dst, ptr %"dst'", double %x, double %"x'", ptr %tapeArg)
; CHECK-NOT: cmpxchg ptr %dst
; CHECK:   %[[CMP:.+]] = icmp ne ptr %dst, %"dst'"
; CHECK:   br i1 %[[CMP]], label %[[THEN:.+]], label %[[CONT:.+]]
; CHECK: [[THEN]]:
; CHECK:   %[[L:.+]] = load atomic double, ptr %"dst'" monotonic, align 8
; CHECK:   %[[S:.+]] = select {{.*}}i1 %changed, double %"x'", double %[[L]]
; CHECK:   store atomic double %[[S]], ptr %"dst'" monotonic, align 8
; CHECK:   br label %[[CONT]]
; CHECK: [[CONT]]:
; CHECK:   ret void
