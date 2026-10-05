; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Julia-style functions that copy an array into a new Memory, as `x .- μ` does.
; For n == 0 Julia uses the empty Memory singleton instead of allocating, which
; Enzyme.jl marks enzymejl_empty_memory. The data pointer is loaded from the
; field after the length. Writing that data writes memory that did not exist
; before the call, so @fill only writes local memory.

@memty = external addrspace(10) global i8
@empty = external addrspace(10) global i8, !enzymejl_empty_memory !0
@other = external addrspace(10) global i8

declare noalias nonnull ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10), i64) #0

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

attributes #0 = { "enzyme_ReadOnlyOrThrow" }

!0 = !{}

; CHECK: define void @fill({{.*}}) #[[FILL:[0-9]+]] {
; CHECK: define void @fill_or_empty({{.*}}) #[[FILL]] {
; CHECK: define void @fill_or_other(
; CHECK-NOT: #[[FILL]]
; CHECK-SAME: {
; CHECK: define void @fill_wrong_field(
; CHECK-NOT: #[[FILL]]
; CHECK-SAME: {
; CHECK: attributes #[[FILL]] = { {{.*}}"enzyme_LocalReadOnlyOrThrow"{{.*}} }
