; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

declare double @__enzyme_autodiff(...)

define double @atomic_fadd(ptr %dst, double %x) noinline {
entry:
  %old = atomicrmw fadd ptr %dst, double %x monotonic, align 8
  ret double %old
}

define double @atomic_fmax(ptr %dst, double %x) noinline {
entry:
  %old = atomicrmw fmax ptr %dst, double %x monotonic, align 8
  ret double %old
}

define double @test_atomic_fadd(ptr %p, ptr %dp, double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @atomic_fadd, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp, double %x)
  ret double %r
}

define double @test_atomic_fmax(ptr %p, ptr %dp, double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @atomic_fmax, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp, double %x)
  ret double %r
}

; CHECK-LABEL: define internal { double } @diffeatomic_fadd(ptr %dst, ptr %"dst'", double %x, double %differeturn)
; CHECK: invertentry:
; CHECK:   %[[C1:.+]] = icmp ne ptr %dst, %"dst'"
; CHECK:   br i1 %[[C1]], label %invertentry_active, label %invertentry_amerge
; CHECK: invertentry_active:
; CHECK:   load atomic double, ptr %"dst'" monotonic, align 8
; CHECK:   br label %invertentry_amerge
; CHECK: invertentry_amerge:
; CHECK:   %[[C2:.+]] = icmp ne ptr %dst, %"dst'"
; CHECK:   br i1 %[[C2]], label %invertentry_amerge_active, label %invertentry_amerge_amerge
; CHECK: invertentry_amerge_active:
; CHECK:   atomicrmw fadd ptr %"dst'", double %{{.+}} monotonic, align 8
; CHECK:   br label %invertentry_amerge_amerge
; CHECK: invertentry_amerge_amerge:
; CHECK:   ret { double }

; CHECK-LABEL: define internal { double } @diffeatomic_fmax(ptr %dst, ptr %"dst'", double %x, double %differeturn)
; CHECK: invertentry:
; CHECK:   %[[C:.+]] = icmp ne ptr %dst, %"dst'"
; CHECK:   br i1 %[[C]], label %invertentry_active, label %invertentry_amerge
; CHECK: invertentry_active:
; CHECK:   br i1 %changed, label %invertentry_changed, label %invertentry_merge
; CHECK: invertentry_changed:
; CHECK:   atomicrmw xchg ptr %"dst'", double 0.000000e+00 monotonic, align 8
; CHECK:   br label %invertentry_merge
; CHECK: invertentry_merge:
; CHECK:   atomicrmw fadd ptr %"dst'", double %{{.+}} monotonic, align 8
; CHECK:   br label %invertentry_amerge
; CHECK: invertentry_amerge:
; CHECK:   ret { double }
