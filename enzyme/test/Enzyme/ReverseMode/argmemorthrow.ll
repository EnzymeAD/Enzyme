; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

declare void @throw_oob() cold noreturn

define void @checked_write(ptr %dst, ptr %src, i64 %i, i64 %n) {
entry:
  %oob = icmp uge i64 %i, %n
  br i1 %oob, label %err, label %ok

err:
  call void @throw_oob()
  unreachable

ok:
  %sp = getelementptr inbounds double, ptr %src, i64 %i
  %v = load double, ptr %sp, align 8
  %dp = getelementptr inbounds double, ptr %dst, i64 %i
  store double %v, ptr %dp, align 8
  ret void
}

define double @caller_alloca(ptr %src, i64 %n) {
entry:
  %tmp = alloca [4 x double], align 8
  call void @checked_write(ptr %tmp, ptr %src, i64 0, i64 %n)
  %v = load double, ptr %tmp, align 8
  ret double %v
}

define void @store_fresh_ptr(ptr %slot) {
entry:
  %m = call noalias ptr @malloc(i64 8)
  store i64 7, ptr %m, align 8
  store ptr %m, ptr %slot, align 8
  ret void
}

define ptr @caller_alloca_ret_ptr() {
entry:
  %slot = alloca ptr, align 8
  call void @store_fresh_ptr(ptr %slot)
  %p = load ptr, ptr %slot, align 8
  ret ptr %p
}

define void @rec_slice(ptr %dst, i64 %n) {
entry:
  %done = icmp eq i64 %n, 0
  br i1 %done, label %exit, label %step

step:
  store double 0.0, ptr %dst, align 8
  %next_ptr = getelementptr inbounds double, ptr %dst, i64 1
  %next_n = sub i64 %n, 1
  call void @rec_slice(ptr %next_ptr, i64 %next_n)
  br label %exit

exit:
  ret void
}

define void @rec_swap(ptr %a, ptr %b, i64 %n) {
entry:
  %done = icmp eq i64 %n, 0
  br i1 %done, label %exit, label %step

step:
  store double 1.0, ptr %a, align 8
  %next_n = sub i64 %n, 1
  call void @rec_swap(ptr %b, ptr %a, i64 %next_n)
  br label %exit

exit:
  ret void
}

define void @rec_chase(ptr %node, i64 %n) {
entry:
  %done = icmp eq i64 %n, 0
  br i1 %done, label %exit, label %step

step:
  %val_p = getelementptr inbounds { ptr, double }, ptr %node, i64 0, i32 1
  store double 0.0, ptr %val_p, align 8
  %next = load ptr, ptr %node, align 8
  %next_n = sub i64 %n, 1
  call void @rec_chase(ptr %next, i64 %next_n)
  br label %exit

exit:
  ret void
}

define double @upgraded_from_argmem(ptr %x) "enzyme_ArgMemOnlyOrThrow" {
entry:
  %v = load double, ptr %x, align 8
  ret double %v
}

declare noalias ptr @malloc(i64)

; CHECK: define void @checked_write(ptr {{.*}}writeonly{{.*}} %dst, ptr {{.*}}readonly{{.*}} %src, i64 %i, i64 %n) #[[ARGMEM_THROW:[0-9]+]]
; CHECK: define double @caller_alloca(ptr {{.*}}readonly{{.*}} %src, i64 %n) #[[RO_THROW:[0-9]+]]
; CHECK: define void @store_fresh_ptr(ptr {{.*}}writeonly{{.*}} %slot) #[[ARGMEM_THROW]]
; CHECK: define ptr @caller_alloca_ret_ptr() #[[LOCAL_THROW:[0-9]+]]
; CHECK: define void @rec_slice(ptr {{.*}}writeonly{{.*}} %dst, i64 %n) #[[ARGMEM_NW:[0-9]+]]
; CHECK-LABEL: define void @rec_swap(
; CHECK-NOT: readonly
; CHECK-SAME: ptr {{.*}}%a, ptr {{.*}}%b, i64 %n) #[[ARGMEM_NW]]
; CHECK: define void @rec_chase(ptr {{.*}}%node, i64 %n) #[[CHASE:[0-9]+]]
; CHECK: define double @upgraded_from_argmem(ptr {{.*}}readonly{{.*}} %x) #[[RO_NW:[0-9]+]]

; CHECK-DAG: attributes #[[ARGMEM_THROW]] = { memory(read, argmem: readwrite, inaccessiblemem: readwrite) "enzyme_ArgMemOnlyOrThrow" }
; CHECK-DAG: attributes #[[RO_THROW]] = { memory(read, inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" }
; CHECK-DAG: attributes #[[LOCAL_THROW]] = { memory(read, argmem: readwrite, inaccessiblemem: readwrite) "enzyme_LocalReadOnlyOrThrow" }
; CHECK-DAG: attributes #[[ARGMEM_NW]] = { nounwind memory(read, argmem: readwrite, inaccessiblemem: readwrite) "enzyme_ArgMemOnlyOrThrow" }
; CHECK-DAG: attributes #[[RO_NW]] = { nounwind memory(read, inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" }
; CHECK-DAG: attributes #[[CHASE]] = { nounwind }
