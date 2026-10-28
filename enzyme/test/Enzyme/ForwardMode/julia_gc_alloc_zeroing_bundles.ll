; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; Julia 1.14 (julia#60924) zeroes the GC pointer fields of a new object in
; late-gc-lowering, from the julia.gc_alloc_ptr_offsets operand bundle on the
; allocation. The shadow allocation must carry the same bundle.

declare double @__enzyme_fwddiff(...)

define double @caller(double %x, double %dx) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @f, double %x, double %dx)
  ret double %r
}

define double @f(double %x) {
entry:
  %obj = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null) [ "julia.gc_alloc_ptr_offsets"(i64 0) ]
  %fld = getelementptr inbounds i8, ptr addrspace(10) %obj, i64 8
  store double %x, ptr addrspace(10) %fld, align 8
  call void @use(ptr addrspace(10) %obj)
  %v = load double, ptr addrspace(10) %fld, align 8
  %r = fmul double %v, %v
  ret double %r
}

define void @use(ptr addrspace(10) nocapture readonly %y) nofree {
entry:
  ret void
}

declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10)) inaccessiblememonly allocsize(1)

; CHECK: define internal double @fwddiffef(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %obj = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null) #{{[0-9]+}} [ "julia.gc_alloc_ptr_offsets"(i64 0) ]
; CHECK-NEXT:   %{{.+}} = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null) #{{[0-9]+}} [ "julia.gc_alloc_ptr_offsets"(i64 0) ]
