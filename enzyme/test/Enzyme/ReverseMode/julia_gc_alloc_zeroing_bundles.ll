; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; Julia 1.14 (julia#60924) zeroes the GC pointer fields of a new object in
; late-gc-lowering, from the julia.gc_alloc_ptr_offsets operand bundle on the
; allocation. On such a Julia, the shadow allocation is zeroed the same way,
; with a julia.gc_alloc_zeroinit bundle over the whole object instead of a
; memset, and is marked allockind("alloc,zeroed").

declare void @__enzyme_autodiff(...)

define void @caller(double %x) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_out", double %x)
  ret void
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

; CHECK: define internal { double } @diffef(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"obj'mi" = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null) #[[ZEROED:[0-9]+]] [ "julia.gc_alloc_zeroinit"(i64 0, i64 16) ]
; CHECK-NEXT:   %obj = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null) #{{[0-9]+}} [ "julia.gc_alloc_ptr_offsets"(i64 0) ]
; CHECK-NOT: @llvm.memset
; CHECK: ret { double }

; CHECK: attributes #[[ZEROED]] = { {{.*}}allockind("alloc,zeroed"){{.*}} }
