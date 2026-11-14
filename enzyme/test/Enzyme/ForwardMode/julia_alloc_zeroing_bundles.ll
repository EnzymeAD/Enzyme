; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; Julia 1.13.4 and later leave zeroing the GC pointers of a new object to
; late-gc-lowering, which finds what to zero in bundles on the allocation. The
; shadow allocation has the same layout, so takes the same bundles.

@ty = external addrspace(10) global i8

declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))
declare double @__enzyme_fwddiff(...)

define double @f(ptr %task, double %x, i64 %n) {
top:
  %a = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 16, ptr addrspace(10) @ty) [ "julia.gc_alloc_ptr_offsets"(i64 8) ]
  %b = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 32, ptr addrspace(10) @ty) [ "julia.gc_alloc_zeroinit"(i64 16, i64 16) ]
  %c = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 %n, ptr addrspace(10) @ty) [ "julia.gc_alloc_zeroinit_indirect"(i64 8, i64 %n) ]
  store double %x, ptr addrspace(10) %a, align 8
  %bp = getelementptr inbounds i8, ptr addrspace(10) %b, i64 8
  store ptr addrspace(10) %a, ptr addrspace(10) %bp, align 8
  store ptr addrspace(10) %b, ptr addrspace(10) %c, align 8
  %b2 = load ptr addrspace(10), ptr addrspace(10) %c, align 8
  %bp2 = getelementptr inbounds i8, ptr addrspace(10) %b2, i64 8
  %a2 = load ptr addrspace(10), ptr addrspace(10) %bp2, align 8
  %v = load double, ptr addrspace(10) %a2, align 8
  ret double %v
}

define double @df(ptr %task, double %x, i64 %n) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_const", ptr %task, double %x, double 1.0, metadata !"enzyme_const", i64 %n)
  ret double %r
}

; CHECK: define internal double @fwddiffef(ptr %task, double %x, double %"x'", i64 %n)
; CHECK-NEXT: top:
; CHECK-NEXT:   %0 = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 16, ptr addrspace(10) @ty) #{{[0-9]+}} [ "julia.gc_alloc_ptr_offsets"(i64 8) ]
; CHECK-NEXT:   %1 = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 32, ptr addrspace(10) @ty) #{{[0-9]+}} [ "julia.gc_alloc_zeroinit"(i64 16, i64 16) ]
; CHECK-NEXT:   %2 = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 %n, ptr addrspace(10) @ty) #{{[0-9]+}} [ "julia.gc_alloc_zeroinit_indirect"(i64 8, i64 %n) ]
; CHECK-NEXT:   store double %"x'", ptr addrspace(10) %0, align 8
