; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Like readonlyorthrow_fresh_memory.ll, but the data is written through a
; pointer loaded from the data field, at offset 0, of a Julia array. The field
; can change, so it is only fresh if the array was allocated here, its address
; does not escape, and every store to the field stores fresh data.

@memty = external addrspace(10) global i8
@arrty = external addrspace(10) global i8
@escape = external global ptr addrspace(10)

declare noalias nonnull ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10), i64) #0
declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10)) #0
declare noalias nonnull ptr addrspace(10) @jl_alloc_array_1d(ptr addrspace(10), i64) #0
declare void @jl_array_grow_end(ptr addrspace(10), i64)

; Julia 1.11+: the array is allocated like any object and pointed at the data
; of a new Memory.

define void @fill_array(ptr %task, ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  %m11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %mdatap = getelementptr inbounds i8, ptr addrspace(11) %m11, i64 8
  %mdata = load ptr, ptr addrspace(11) %mdatap, align 8
  %a = call ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 24, ptr addrspace(10) @arrty)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  store ptr null, ptr addrspace(11) %a11, align 8
  store ptr %mdata, ptr addrspace(11) %a11, align 8
  %amemp = getelementptr inbounds i8, ptr addrspace(11) %a11, i64 8
  store ptr addrspace(10) %m, ptr addrspace(11) %amemp, align 8
  %data = load ptr, ptr addrspace(11) %a11, align 8
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr %data, align 8
  ret void
}

; Julia 1.10: jl_alloc_array_1d allocates the array with fresh data.

define void @fill_array_110(ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %a = call ptr addrspace(10) @jl_alloc_array_1d(ptr addrspace(10) @arrty, i64 %n)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  %data = load ptr, ptr addrspace(11) %a11, align 8
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr %data, align 8
  ret void
}

; The data field holds data that may exist before the call.

define void @fill_array_other_data(ptr %task, ptr addrspace(11) nocapture readonly %x, ptr %other) {
top:
  %a = call ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 24, ptr addrspace(10) @arrty)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  store ptr %other, ptr addrspace(11) %a11, align 8
  %data = load ptr, ptr addrspace(11) %a11, align 8
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr %data, align 8
  ret void
}

; The array escapes, so other code may change its data field.

define void @fill_array_escaped(ptr %task, ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %m = call ptr addrspace(10) @jl_alloc_genericmemory(ptr addrspace(10) @memty, i64 %n)
  %m11 = addrspacecast ptr addrspace(10) %m to ptr addrspace(11)
  %mdatap = getelementptr inbounds i8, ptr addrspace(11) %m11, i64 8
  %mdata = load ptr, ptr addrspace(11) %mdatap, align 8
  %a = call ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 24, ptr addrspace(10) @arrty)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  store ptr %mdata, ptr addrspace(11) %a11, align 8
  store ptr addrspace(10) %a, ptr @escape, align 8
  %data = load ptr, ptr addrspace(11) %a11, align 8
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr %data, align 8
  ret void
}

; The array is passed to a call that may change its data field.

define void @fill_array_110_grown(ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %a = call ptr addrspace(10) @jl_alloc_array_1d(ptr addrspace(10) @arrty, i64 %n)
  call void @jl_array_grow_end(ptr addrspace(10) %a, i64 1)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  %data = load ptr, ptr addrspace(11) %a11, align 8
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr %data, align 8
  ret void
}

; Only the field at offset 0 holds the array's data.

define void @fill_array_wrong_field(ptr addrspace(11) nocapture readonly %x, i64 %n) {
top:
  %a = call ptr addrspace(10) @jl_alloc_array_1d(ptr addrspace(10) @arrty, i64 %n)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  %datap = getelementptr inbounds i8, ptr addrspace(11) %a11, i64 16
  %data = load ptr, ptr addrspace(11) %datap, align 8
  %xi = load double, ptr addrspace(11) %x, align 8
  store double %xi, ptr %data, align 8
  ret void
}

attributes #0 = { "enzyme_ReadOnlyOrThrow" }

; CHECK: define void @fill_array({{.*}}) #[[FILL:[0-9]+]] {
; CHECK: define void @fill_array_110({{.*}}) #[[FILL]] {
; CHECK: define void @fill_array_other_data(
; CHECK-NOT: #[[FILL]]
; CHECK-SAME: {
; CHECK: define void @fill_array_escaped(
; CHECK-NOT: #[[FILL]]
; CHECK-SAME: {
; CHECK: define void @fill_array_110_grown(
; CHECK-NOT: #[[FILL]]
; CHECK-SAME: {
; CHECK: define void @fill_array_wrong_field(
; CHECK-NOT: #[[FILL]]
; CHECK-SAME: {
; CHECK: attributes #[[FILL]] = { {{.*}}"enzyme_LocalReadOnlyOrThrow"{{.*}} }
