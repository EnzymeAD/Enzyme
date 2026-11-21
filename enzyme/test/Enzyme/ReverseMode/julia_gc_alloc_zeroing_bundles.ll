; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; Julia 1.13.4 and 1.14 (julia#60924) null the GC pointer fields of a new
; object in late-gc-lowering, from the julia.gc_alloc_ptr_offsets operand bundle
; on the allocation. The shadow allocation carries the same bundle, so Enzyme
; only zeroes the bytes between the GC pointer fields.

declare void @__enzyme_autodiff(...)

define void @caller(double %x) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_out", double %x)
  call void (...) @__enzyme_autodiff(ptr @g, metadata !"enzyme_out", double %x)
  call void (...) @__enzyme_autodiff(ptr @h, metadata !"enzyme_out", double %x)
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

; A Tuple{Float64,Any}: a double at offset 0, a GC pointer at offset 8. Codegen
; marks a fully initialized object allockind("alloc,uninitialized"). The shadow
; is not initialized by the primal's stores, so it must not keep that.
define double @g(double %x) {
entry:
  %obj = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null) #0 [ "julia.gc_alloc_ptr_offsets"(i64 8) ]
  store double %x, ptr addrspace(10) %obj, align 8
  call void @use(ptr addrspace(10) %obj)
  %v = load double, ptr addrspace(10) %obj, align 8
  %r = fmul double %v, %v
  ret double %r
}

; Two adjacent GC pointers at offsets 8 and 0 (in either order), a double at 16.
define double @h(double %x) {
entry:
  %obj = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 24, ptr addrspace(10) null) [ "julia.gc_alloc_ptr_offsets"(i64 8, i64 0) ]
  %fld = getelementptr inbounds i8, ptr addrspace(10) %obj, i64 16
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

attributes #0 = { allockind("alloc,uninitialized") }

; CHECK: define internal { double } @diffef(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"obj'mi" = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null) #{{[0-9]+}} [ "julia.gc_alloc_ptr_offsets"(i64 0) ]
; CHECK-NEXT:   %[[fgep:.+]] = getelementptr inbounds i8, ptr addrspace(10) %"obj'mi", i64 8
; CHECK-NEXT:   call void @llvm.memset.p10.i64(ptr addrspace(10) nonnull %[[fgep]], i8 0, i64 8, i1 false)
; CHECK-NEXT:   %obj = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null) #{{[0-9]+}} [ "julia.gc_alloc_ptr_offsets"(i64 0) ]

; CHECK: define internal { double } @diffeg(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"obj'mi" = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null) #[[shadowattrs:[0-9]+]] [ "julia.gc_alloc_ptr_offsets"(i64 8) ]
; CHECK-NEXT:   call void @llvm.memset.p10.i64(ptr addrspace(10) nonnull %"obj'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %obj = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null) #[[primalattrs:[0-9]+]] [ "julia.gc_alloc_ptr_offsets"(i64 8) ]

; CHECK: define internal { double } @diffeh(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"obj'mi" = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 24, ptr addrspace(10) null) #{{[0-9]+}} [ "julia.gc_alloc_ptr_offsets"(i64 8, i64 0) ]
; CHECK-NEXT:   %[[hgep:.+]] = getelementptr inbounds i8, ptr addrspace(10) %"obj'mi", i64 16
; CHECK-NEXT:   call void @llvm.memset.p10.i64(ptr addrspace(10) nonnull %[[hgep]], i8 0, i64 8, i1 false)
; CHECK-NEXT:   %obj = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 24, ptr addrspace(10) null) #{{[0-9]+}} [ "julia.gc_alloc_ptr_offsets"(i64 8, i64 0) ]

; CHECK-DAG: attributes #[[primalattrs]] = { {{.*}}allockind("alloc,uninitialized") }
; CHECK-DAG: attributes #[[shadowattrs]] = { {{.*}}allockind("alloc") }
