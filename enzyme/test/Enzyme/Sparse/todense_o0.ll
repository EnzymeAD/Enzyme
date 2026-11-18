; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="always-inline,function(gvn,sroa),enzyme" -enzyme-postopt=1 -enzyme-auto-sparsity=1 -S | %lli - ; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -enzyme-auto-sparsity=1 -S | %lli - ; fi

declare ptr @__enzyme_todense(...)
declare void @__enzyme_autodiff(...)
declare void @__enzyme_fwddiff(...)

@enzyme_const = external global i32, align 4
@enzyme_dup = external global i32, align 4
@enzyme_dupnoneed = external global i32, align 4

define void @accumulate(i64 %row, i64 %col, double %val, ptr %acc) #0 {
entry:
  %count = load i64, ptr %acc, align 8
  %next = add i64 %count, 1
  store i64 %next, ptr %acc, align 8
  %sum_ptr = getelementptr inbounds double, ptr %acc, i64 1
  %old_sum = load double, ptr %sum_ptr, align 8
  %abs_val = call double @llvm.fabs.f64(double %val)
  %new_sum = fadd double %old_sum, %abs_val
  store double %new_sum, ptr %sum_ptr, align 8
  ret void
}

declare double @llvm.fabs.f64(double)

define internal double @seed_load(i64 %off, i64 %col) #1 {
entry:
  %off.addr = alloca i64, align 8
  %col.addr = alloca i64, align 8
  store i64 %off, ptr %off.addr, align 8
  store i64 %col, ptr %col.addr, align 8
  %0 = load i64, ptr %off.addr, align 8
  %idx = lshr exact i64 %0, 3
  %1 = load i64, ptr %col.addr, align 8
  %c = icmp eq i64 %idx, %1
  %r = uitofp i1 %c to double
  ret double %r
}

define internal void @nostore(double %v, i64 %off, i64 %col) #1 {
entry:
  ret void
}

define internal double @zero_load(i64 %off, i64 %col, ptr %acc) #1 {
entry:
  ret double 0.000000e+00
}

define internal void @acc_store(double %v, i64 %off, i64 %col, ptr %acc) #1 {
entry:
  %v.addr = alloca double, align 8
  %off.addr = alloca i64, align 8
  %col.addr = alloca i64, align 8
  %acc.addr = alloca ptr, align 8
  store double %v, ptr %v.addr, align 8
  store i64 %off, ptr %off.addr, align 8
  store i64 %col, ptr %col.addr, align 8
  store ptr %acc, ptr %acc.addr, align 8
  %0 = load double, ptr %v.addr, align 8
  %z = fcmp oeq double %0, 0.000000e+00
  br i1 %z, label %end, label %do

do:
  %1 = load i64, ptr %off.addr, align 8
  %idx = lshr exact i64 %1, 3
  %2 = load i64, ptr %col.addr, align 8
  %3 = load double, ptr %v.addr, align 8
  %4 = load ptr, ptr %acc.addr, align 8
  call void @accumulate(i64 %2, i64 %idx, double %3, ptr %4)
  br label %end

end:
  ret void
}

define internal double @energy(i64 %n, ptr %x) #1 {
entry:
  %n.addr = alloca i64, align 8
  %x.addr = alloca ptr, align 8
  %out = alloca double, align 8
  %i = alloca i64, align 8
  store i64 %n, ptr %n.addr, align 8
  store ptr %x, ptr %x.addr, align 8
  store double 0.000000e+00, ptr %out, align 8
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:
  %0 = load i64, ptr %i, align 8
  %1 = load i64, ptr %n.addr, align 8
  %cmp = icmp ult i64 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %2 = load ptr, ptr %x.addr, align 8
  %3 = load i64, ptr %i, align 8
  %p0 = getelementptr inbounds double, ptr %2, i64 %3
  %v0 = load double, ptr %p0, align 8
  %sq = fmul fast double %v0, %v0
  %cur = load double, ptr %out, align 8
  %add = fadd fast double %cur, %sq
  store double %add, ptr %out, align 8
  br label %for.inc

for.inc:
  %4 = load i64, ptr %i, align 8
  %inc = add i64 %4, 1
  store i64 %inc, ptr %i, align 8
  br label %for.cond

for.end:
  %res = load double, ptr %out, align 8
  ret double %res
}

define internal void @grad_energy(i64 %n, ptr %x, ptr %dx) #1 {
entry:
  %n.addr = alloca i64, align 8
  %x.addr = alloca ptr, align 8
  %dx.addr = alloca ptr, align 8
  store i64 %n, ptr %n.addr, align 8
  store ptr %x, ptr %x.addr, align 8
  store ptr %dx, ptr %dx.addr, align 8
  %c = load i32, ptr @enzyme_const, align 4
  %0 = load i64, ptr %n.addr, align 8
  %d = load i32, ptr @enzyme_dup, align 4
  %1 = load ptr, ptr %x.addr, align 8
  %2 = load ptr, ptr %dx.addr, align 8
  call void (...) @__enzyme_autodiff(ptr @energy, i32 %c, i64 %0, i32 %d, ptr %1, ptr %2)
  ret void
}

; CHECK-LABEL: define void @hess_o0(
; CHECK-NOT: @__enzyme_todense
; CHECK: call void @hess_o0.invertfor.cond.i.i(
; CHECK-NOT: call void @hess_o0.invertfor.cond.i.i(
define void @hess_o0(i64 %n, ptr %x, ptr %acc) #2 {
entry:
  %n.addr = alloca i64, align 8
  %x.addr = alloca ptr, align 8
  %acc.addr = alloca ptr, align 8
  %i = alloca i64, align 8
  %dx = alloca ptr, align 8
  %ddx = alloca ptr, align 8
  store i64 %n, ptr %n.addr, align 8
  store ptr %x, ptr %x.addr, align 8
  store ptr %acc, ptr %acc.addr, align 8
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:
  %0 = load i64, ptr %i, align 8
  %1 = load i64, ptr %n.addr, align 8
  %cmp = icmp ult i64 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %2 = load i64, ptr %i, align 8
  %t1 = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %2)
  store ptr %t1, ptr %dx, align 8
  %3 = load i64, ptr %i, align 8
  %4 = load ptr, ptr %acc.addr, align 8
  %t2 = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store, i64 %3, ptr %4)
  store ptr %t2, ptr %ddx, align 8
  %c = load i32, ptr @enzyme_const, align 4
  %5 = load i64, ptr %n.addr, align 8
  %d = load i32, ptr @enzyme_dup, align 4
  %6 = load ptr, ptr %x.addr, align 8
  %7 = load ptr, ptr %dx, align 8
  %dn = load i32, ptr @enzyme_dupnoneed, align 4
  %8 = load ptr, ptr %ddx, align 8
  call void (...) @__enzyme_fwddiff(ptr @grad_energy, i32 %c, i64 %5, i32 %d, ptr %6, ptr %7, i32 %dn, ptr inttoptr (i64 1 to ptr), ptr %8)
  br label %for.inc

for.inc:
  %9 = load i64, ptr %i, align 8
  %inc = add i64 %9, 1
  store i64 %inc, ptr %i, align 8
  br label %for.cond

for.end:
  ret void
}

define internal void @vec_sq(i64 %n, ptr %x, ptr %y) #1 {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %xp = getelementptr inbounds double, ptr %x, i64 %i
  %xv = load double, ptr %xp, align 8, !tbaa !4
  %sq = fmul fast double %xv, %xv
  %yp = getelementptr inbounds double, ptr %y, i64 %i
  store double %sq, ptr %yp, align 8, !tbaa !4
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: define i64 @jac_tbaa(
; CHECK-NOT: @__enzyme_todense
; CHECK: %[[NNZ:.+]] = load i64, ptr %acc, align 8
; CHECK-NEXT: ret i64 %[[NNZ]]
define i64 @jac_tbaa(i64 %n, ptr %x) {
entry:
  %acc = alloca [2 x i64], align 8
  store i64 0, ptr %acc, align 8, !tbaa !0
  %sum_ptr = getelementptr inbounds double, ptr %acc, i64 1
  store double 0.000000e+00, ptr %sum_ptr, align 8, !tbaa !4
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %i)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store, i64 %i, ptr %acc)
  %c = load i32, ptr @enzyme_const, align 4
  %d = load i32, ptr @enzyme_dup, align 4
  %dn = load i32, ptr @enzyme_dupnoneed, align 4
  call void (...) @__enzyme_fwddiff(ptr @vec_sq, i32 %c, i64 %n, i32 %d, ptr %x, ptr %dx, i32 %dn, ptr inttoptr (i64 1 to ptr), ptr %dy)
  %inc = add nuw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %for.end, label %for.body

for.end:
  %nnz = load i64, ptr %acc, align 8, !tbaa !0
  ret i64 %nnz
}

define i32 @main() {
entry:
  %x = alloca [4 x double], align 8
  %acc = alloca [2 x i64], align 8
  store double 1.000000e+00, ptr %x, align 8
  %x1 = getelementptr inbounds double, ptr %x, i64 1
  store double 2.000000e+00, ptr %x1, align 8
  %x2 = getelementptr inbounds double, ptr %x, i64 2
  store double 3.000000e+00, ptr %x2, align 8
  %x3 = getelementptr inbounds double, ptr %x, i64 3
  store double 4.000000e+00, ptr %x3, align 8
  store i64 0, ptr %acc, align 8
  %sum_ptr = getelementptr inbounds double, ptr %acc, i64 1
  store double 0.000000e+00, ptr %sum_ptr, align 8
  call void @hess_o0(i64 4, ptr %x, ptr %acc)
  %count = load i64, ptr %acc, align 8
  %ok_count = icmp eq i64 %count, 4
  br i1 %ok_count, label %check_sum, label %fail

check_sum:
  %sum = load double, ptr %sum_ptr, align 8
  %ok_sum = fcmp oeq double %sum, 8.000000e+00
  br i1 %ok_sum, label %check_tbaa, label %fail

check_tbaa:
  %nnz = call i64 @jac_tbaa(i64 4, ptr %x)
  %ok_nnz = icmp eq i64 %nnz, 4
  br i1 %ok_nnz, label %pass, label %fail

pass:
  ret i32 0

fail:
  ret i32 1
}

attributes #0 = { noinline optnone "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }
attributes #2 = { noinline optnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"long", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"double", !2, i64 0}
