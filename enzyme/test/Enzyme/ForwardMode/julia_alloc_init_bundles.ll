; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; Julia 1.14 attaches bundles to its allocations that tell its GC lowering to
; zero their pointer fields; the shadow allocation must carry them too.

declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))

declare ptr @julia.get_pgcstack()

define double @f(double %x) {
top:
  %pgcstack = call ptr @julia.get_pgcstack()
  %obj = call noalias nonnull align 8 dereferenceable(16) ptr addrspace(10) @julia.gc_alloc_obj(ptr %pgcstack, i64 16, ptr addrspace(10) addrspacecast (ptr inttoptr (i64 1234 to ptr) to ptr addrspace(10))) [ "julia.gc_alloc_ptr_offsets"(i64 8) ]
  %fld = addrspacecast ptr addrspace(10) %obj to ptr addrspace(11)
  store double %x, ptr addrspace(11) %fld, align 8
  %ld = load double, ptr addrspace(11) %fld, align 8
  ret double %ld
}

define double @g(double %x) {
top:
  %pgcstack = call ptr @julia.get_pgcstack()
  %obj = call noalias nonnull align 8 dereferenceable(16) ptr addrspace(10) @julia.gc_alloc_obj(ptr %pgcstack, i64 16, ptr addrspace(10) addrspacecast (ptr inttoptr (i64 1234 to ptr) to ptr addrspace(10))) [ "julia.gc_alloc_zeroinit"(i64 8, i64 8) ]
  %fld = addrspacecast ptr addrspace(10) %obj to ptr addrspace(11)
  store double %x, ptr addrspace(11) %fld, align 8
  %ld = load double, ptr addrspace(11) %fld, align 8
  ret double %ld
}

declare double @__enzyme_fwddiff(...)

define double @df(double %x, double %dx) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @f, double %x, double %dx)
  ret double %r
}

define double @dg(double %x, double %dx) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @g, double %x, double %dx)
  ret double %r
}

; The primal allocation is dead, so only the shadow allocation remains.

; CHECK-LABEL: define internal double @fwddiffef(
; CHECK: %0 = call {{.*}}@julia.gc_alloc_obj(ptr %pgcstack, i64 16, {{.*}}) {{.*}}[ "julia.gc_alloc_ptr_offsets"(i64 8) ]
; CHECK-NEXT: %"fld'ipc" = addrspacecast ptr addrspace(10) %0 to ptr addrspace(11)
; CHECK-NEXT: store double %"x'", ptr addrspace(11) %"fld'ipc"

; CHECK-LABEL: define internal double @fwddiffeg(
; CHECK: %0 = call {{.*}}@julia.gc_alloc_obj(ptr %pgcstack, i64 16, {{.*}}) {{.*}}[ "julia.gc_alloc_zeroinit"(i64 8, i64 8) ]
; CHECK-NEXT: %"fld'ipc" = addrspacecast ptr addrspace(10) %0 to ptr addrspace(11)
; CHECK-NEXT: store double %"x'", ptr addrspace(11) %"fld'ipc"
