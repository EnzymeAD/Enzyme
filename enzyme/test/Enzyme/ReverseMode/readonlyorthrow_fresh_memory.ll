; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Julia-style functions that copy an array into a new Memory, as `x .- μ` does.
; For n == 0 Julia uses the empty Memory singleton instead of allocating, which
; Enzyme.jl marks enzymejl_empty_memory. The data pointer is loaded from the
; field after the length. Writing that data writes memory that did not exist
; before the call. As @fill does not return it, @fill is fully
; read-only-or-throw; @fill_ret, which returns the Memory, is only local.

@memty = external addrspace(10) global i8
@empty = external addrspace(10) global i8, !enzymejl_empty_memory !0
@other = external addrspace(10) global i8

declare noalias nonnull ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10), i64) #0
declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))
declare ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10), ptr) #1

define void @fill(ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %isempty = icmp eq i64 %n, 0
  br i1 %isempty, label %exit, label %alloc

alloc:
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  br label %join

join:
  %mem = phi ptr addrspace(10) [ %m, %alloc ]
  %mem11 = addrspacecast ptr addrspace(10) %mem to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %mem11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %join ], [ %inc, %loop ]
  %xp = getelementptr inbounds double, ptr addrspace(11) %x, i64 %i
  %xi = load double, ptr addrspace(11) %xp, align 8
  %zp = getelementptr inbounds double, ptr %data, i64 %i
  store double %xi, ptr %zp, align 8
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define ptr addrspace(10) @fill_ret(ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  %mem11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %mem11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr %data, align 8
  ret ptr addrspace(10) %m
}

; Here the Memory is either a new one or the empty singleton.

define void @fill_or_empty(ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %isempty = icmp eq i64 %n, 0
  br i1 %isempty, label %join, label %alloc

alloc:
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  br label %join

join:
  %mem = phi ptr addrspace(10) [ @empty, %top ], [ %m, %alloc ]
  %mem11 = addrspacecast ptr addrspace(10) %mem to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %mem11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  br i1 %isempty, label %exit, label %loop

loop:
  %i = phi i64 [ 0, %join ], [ %inc, %loop ]
  %xp = getelementptr inbounds double, ptr addrspace(11) %x, i64 %i
  %xi = load double, ptr addrspace(11) %xp, align 8
  %zp = getelementptr inbounds double, ptr %data, i64 %i
  store double %xi, ptr %zp, align 8
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Likewise with a select instead of a phi.

define void @fill_or_empty_select(ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %isempty = icmp eq i64 %n, 0
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  %mem = select i1 %isempty, ptr addrspace(10) @empty, ptr addrspace(10) %m
  %mem11 = addrspacecast ptr addrspace(10) %mem to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %mem11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr %data, align 8
  ret void
}

; Any other global may be a Memory existing before the call, so a function
; that may write its data must not be marked.

define void @fill_or_other(ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %isempty = icmp eq i64 %n, 0
  br i1 %isempty, label %join, label %alloc

alloc:
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  br label %join

join:
  %mem = phi ptr addrspace(10) [ @other, %top ], [ %m, %alloc ]
  %mem11 = addrspacecast ptr addrspace(10) %mem to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %mem11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  br i1 %isempty, label %exit, label %loop

loop:
  %i = phi i64 [ 0, %join ], [ %inc, %loop ]
  %xp = getelementptr inbounds double, ptr addrspace(11) %x, i64 %i
  %xi = load double, ptr addrspace(11) %xp, align 8
  %zp = getelementptr inbounds double, ptr %data, i64 %i
  store double %xi, ptr %zp, align 8
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Only the field after the length holds the Memory's own data. A pointer
; loaded from elsewhere in the object may point to memory existing before the
; call, so a function that writes through it must not be marked.

define void @fill_wrong_field(ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  %mem11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %mem11, i64 16
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr %data, align 8
  ret void
}

; The data is written through one load of the data field and returned through
; another: any load of the field yields the written data.

define ptr @fill_ret_data(ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  %mem11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %mem11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr %data, align 8
  %data2 = load ptr, ptr addrspace(11) %datap, align 8
  ret ptr %data2
}

; The data pointer of the Memory is kept in an object of the function and
; written through what is loaded back from it; the Memory is returned.

define ptr addrspace(10) @fill_ret_through_box(ptr %task, ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  %mem11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %mem11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %box = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) null)
  %box11 = addrspacecast ptr addrspace(10) %box to ptr addrspace(11)
  store ptr %data, ptr addrspace(11) %box11, align 8
  %data2 = load ptr, ptr addrspace(11) %box11, align 8
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr %data2, align 8
  ret ptr addrspace(10) %m
}

; As Julia 1.11+ emits it, the data pointer is rooted with julia.gc_loaded,
; which takes the Memory only to keep it alive. Not returning the data keeps
; the function fully read-only-or-throw; returning the rooted pointer does not.

define void @fill_gc_loaded(ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  %mem11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %mem11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %rooted = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %m, ptr %data)
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr addrspace(13) %rooted, align 8
  ret void
}

define ptr addrspace(13) @fill_gc_loaded_ret(ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  %mem11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %mem11, i64 8
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %rooted = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %m, ptr %data)
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr addrspace(13) %rooted, align 8
  ret ptr addrspace(13) %rooted
}

attributes #0 = { "enzyme_ReadOnlyOrThrow" }
attributes #1 = { nounwind memory(none) }

!0 = !{}

; CHECK: define void @fill({{.*}}) #[[FILL:[0-9]+]] {
; CHECK: define ptr addrspace(10) @fill_ret({{.*}}) #[[LOCAL:[0-9]+]] {
; CHECK: define void @fill_or_empty({{.*}}) #[[FILL]] {
; CHECK: define void @fill_or_empty_select({{.*}}) #[[FILL]] {
; CHECK: define void @fill_or_other(
; CHECK-NOT: #[[FILL]]
; CHECK-SAME: {
; CHECK: define void @fill_wrong_field(
; CHECK-NOT: #[[FILL]]
; CHECK-SAME: {
; CHECK: define ptr @fill_ret_data({{.*}}) #[[LOCAL]] {
; CHECK: define ptr addrspace(10) @fill_ret_through_box({{.*}}) #[[LOCAL]] {
; CHECK: define void @fill_gc_loaded({{.*}}) #[[FILL]] {
; CHECK: define ptr addrspace(13) @fill_gc_loaded_ret({{.*}}) #[[LOCAL]] {
; CHECK-DAG: attributes #[[FILL]] = { {{.*}}"enzyme_ReadOnlyOrThrow"{{.*}} }
; CHECK-DAG: attributes #[[LOCAL]] = { {{.*}}"enzyme_LocalReadOnlyOrThrow"{{.*}} }
