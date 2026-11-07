; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Julia-style callers that, after calling @sq (which reads x), fill memory
; allocated only afterwards. That memory did not exist while @sq ran, so
; filling it cannot overwrite what @sq read, and @sq need not cache x[0] for
; the reverse pass. In @after the data belongs to a Memory allocated after the
; call (its data pointer is loaded from the field after the length), in
; @header to an array header allocated after the call whose data field is set
; to that Memory's data. In @inloop the Memory is allocated in a loop around
; the call, so a later iteration's call reads memory an earlier iteration
; writes, and x[0] must be cached.

@memty = external addrspace(10) global i8
@arrty = external addrspace(10) global i8

declare noalias nonnull ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10), i64) #0
declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))
declare ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10), ptr) #1
declare double @__enzyme_autodiff(...)

define double @sq(ptr %x) {
  %v = load double, ptr %x, align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @sq2(ptr %x) {
  %v = load double, ptr %x, align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @sq3(ptr %x) {
  %v = load double, ptr %x, align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @after(ptr %x, i64 %n) {
top:
  %v = call double @sq(ptr %x)
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  %m11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %m11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %d13 = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %m, ptr %data)
  %nd = sitofp i64 %n to double
  store double %nd, ptr addrspace(13) %d13, align 8
  ret double %v
}

define double @header(ptr %x, i64 %n) {
top:
  %v = call double @sq2(ptr %x)
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  %m11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %m11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %a = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 24, ptr addrspace(10) @arrty)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  store ptr %data, ptr addrspace(11) %a11, align 8
  %memp = getelementptr inbounds i8, ptr addrspace(11) %a11, i64 8
  store ptr addrspace(10) %m, ptr addrspace(11) %memp, align 8
  %lenp = getelementptr inbounds i8, ptr addrspace(11) %a11, i64 16
  store i64 %n, ptr addrspace(11) %lenp, align 8
  %p = load ptr, ptr addrspace(11) %a11, align 8
  %mem = load ptr addrspace(10), ptr addrspace(11) %memp, align 8
  %d13 = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %mem, ptr %p)
  %nd = sitofp i64 %n to double
  store double %nd, ptr addrspace(13) %d13, align 8
  ret double %v
}

define double @inloop(ptr %x, i64 %n) {
top:
  br label %loop

loop:
  %i = phi i64 [ 0, %top ], [ %inc, %loop ]
  %acc = phi double [ 0.000000e+00, %top ], [ %add, %loop ]
  %v = call double @sq3(ptr %x)
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  %m11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %m11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %d13 = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %m, ptr %data)
  %nd = sitofp i64 %n to double
  store double %nd, ptr addrspace(13) %d13, align 8
  %add = fadd double %acc, %v
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret double %add
}

define void @test(ptr %x, ptr %dx, i64 %n) {
  call double (...) @__enzyme_autodiff(ptr @after, ptr %x, ptr %dx, i64 %n)
  call double (...) @__enzyme_autodiff(ptr @header, ptr %x, ptr %dx, i64 %n)
  call double (...) @__enzyme_autodiff(ptr @inloop, ptr %x, ptr %dx, i64 %n)
  ret void
}

attributes #0 = { "enzyme_ReadOnlyOrThrow" "enzyme_inactive" }
attributes #1 = { nounwind memory(none) }

; @sq need not cache x[0]: its augmented pass keeps nothing, and its reverse
; pass loads x again.

; CHECK: define internal void @augmented_sq(ptr nocapture readonly %x, ptr nocapture %"x'")
; CHECK-NEXT:   %v = load double, ptr %x, align 8
; CHECK-NEXT:   ret void

; CHECK: define internal void @diffesq(ptr nocapture readonly %x, ptr nocapture %"x'", double %differeturn)
; CHECK:   %v = load double, ptr %x, align 8

; CHECK: define internal void @augmented_sq2(ptr nocapture readonly %x, ptr nocapture %"x'")
; CHECK-NEXT:   %v = load double, ptr %x, align 8
; CHECK-NEXT:   ret void

; CHECK: define internal void @diffesq2(ptr nocapture readonly %x, ptr nocapture %"x'", double %differeturn)
; CHECK:   %v = load double, ptr %x, align 8

; In the loop, x[0] is cached: the augmented pass returns it to the reverse pass.

; CHECK: define internal double @augmented_sq3(ptr nocapture readonly %x, ptr nocapture %"x'")
; CHECK:   %v = load double, ptr %x, align 8

; CHECK: define internal void @diffesq3(ptr nocapture readonly %x, ptr nocapture %"x'", double %differeturn, double %v)
