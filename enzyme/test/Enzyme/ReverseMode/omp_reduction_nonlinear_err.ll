; RUN: if [ %llvmver -lt 16 ]; then not %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme -S 2>&1 | FileCheck %s; fi
; RUN: not %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S 2>&1 | FileCheck %s

; Active nonlinear OpenMP reductions such as `reduction(*:prod)` are not
; order-free additive reductions and must emit a structured NoDerivative
; diagnostic rather than crashing or miscomputing gradients.

; CHECK: unsupported nonlinear active OpenMP reduction

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@loc_str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@loc = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @loc_str }, align 8
@red_lock = common global [8 x i32] zeroinitializer, align 8

declare void @__kmpc_fork_call(ptr, i32, ptr, ...)
declare i32 @__kmpc_reduce_nowait(ptr, i32, i32, i64, ptr, ptr, ptr)
declare void @__kmpc_end_reduce_nowait(ptr, i32, ptr)
declare double @__enzyme_autodiff(...)

define internal void @mul_red_func(ptr %lhs_list, ptr %rhs_list) {
entry:
  %lhs_ptr = load ptr, ptr %lhs_list, align 8
  %rhs_ptr = load ptr, ptr %rhs_list, align 8
  %lhs = load double, ptr %lhs_ptr, align 8
  %rhs = load double, ptr %rhs_ptr, align 8
  %mul = fmul double %lhs, %rhs
  store double %mul, ptr %lhs_ptr, align 8
  ret void
}

define internal void @omp_outlined(ptr noalias %gtid_ptr, ptr noalias %btid_ptr, ptr %arr, ptr %prod) {
entry:
  %priv = alloca double, align 8
  %red_list = alloca [1 x ptr], align 8
  %v = load double, ptr %arr, align 8
  store double %v, ptr %priv, align 8
  store ptr %priv, ptr %red_list, align 8
  %gtid = load i32, ptr %gtid_ptr, align 4
  %code = call i32 @__kmpc_reduce_nowait(ptr @loc, i32 %gtid, i32 1, i64 8, ptr %red_list, ptr @mul_red_func, ptr @red_lock)
  switch i32 %code, label %done [
    i32 1, label %crit
  ]

crit:
  %cur = load double, ptr %prod, align 8
  %loc_val = load double, ptr %priv, align 8
  %next = fmul double %cur, %loc_val
  store double %next, ptr %prod, align 8
  call void @__kmpc_end_reduce_nowait(ptr @loc, i32 %gtid, ptr @red_lock)
  br label %done

done:
  ret void
}

define double @product_loss(ptr %arr) {
entry:
  %prod = alloca double, align 8
  store double 1.000000e+00, ptr %prod, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @loc, i32 2, ptr @omp_outlined, ptr %arr, ptr %prod)
  %res = load double, ptr %prod, align 8
  ret double %res
}

define void @caller(ptr %arr, ptr %darr) {
entry:
  %call = call double (...) @__enzyme_autodiff(ptr @product_loss, ptr %arr, ptr %darr)
  ret void
}
