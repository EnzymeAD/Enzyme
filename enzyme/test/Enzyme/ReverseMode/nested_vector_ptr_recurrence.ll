; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -mem2reg -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg)" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg)" -S | %lli -

%Vec = type { ptr, ptr, ptr }

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare double @__enzyme_autodiff(ptr, ...)
declare void @abort()

define internal void @init_leaf_vec(ptr %dst, double %val) noinline {
entry:
  %buf = call noalias ptr @malloc(i64 16)
  %end_cap = getelementptr double, ptr %buf, i64 2
  %finish_slot = getelementptr %Vec, ptr %dst, i64 0, i32 1
  %cap_slot = getelementptr %Vec, ptr %dst, i64 0, i32 2
  store ptr %buf, ptr %dst, align 8
  store ptr %buf, ptr %finish_slot, align 8
  store ptr %end_cap, ptr %cap_slot, align 8
  %guard_slot = alloca ptr, align 8
  store ptr %buf, ptr %guard_slot, align 8
  br label %loop

loop:
  %cur = phi ptr [ %buf, %entry ], [ %next, %loop ]
  store double %val, ptr %cur, align 8
  %next = getelementptr double, ptr %cur, i64 1
  store ptr %next, ptr %guard_slot, align 8
  %done = icmp eq ptr %next, %end_cap
  br i1 %done, label %exit, label %loop

exit:
  store ptr %next, ptr %finish_slot, align 8
  ret void
}

define internal void @copy_vec1(ptr %dst, ptr readonly %src) noinline {
entry:
  %src_begin = load ptr, ptr %src, align 8
  %src_finish_slot = getelementptr %Vec, ptr %src, i64 0, i32 1
  %src_end = load ptr, ptr %src_finish_slot, align 8
  %buf = call noalias ptr @malloc(i64 16)
  %end_cap = getelementptr double, ptr %buf, i64 2
  %finish_slot = getelementptr %Vec, ptr %dst, i64 0, i32 1
  %cap_slot = getelementptr %Vec, ptr %dst, i64 0, i32 2
  store ptr %buf, ptr %dst, align 8
  store ptr %end_cap, ptr %cap_slot, align 8
  %guard_slot = alloca ptr, align 8
  store ptr %buf, ptr %guard_slot, align 8
  br label %loop

loop:
  %in_ptr = phi ptr [ %src_begin, %entry ], [ %in_next, %loop ]
  %cur = load ptr, ptr %guard_slot, align 8
  %v = load double, ptr %in_ptr, align 8
  store double %v, ptr %cur, align 8
  %next = getelementptr double, ptr %cur, i64 1
  store ptr %next, ptr %guard_slot, align 8
  %in_next = getelementptr double, ptr %in_ptr, i64 1
  %done = icmp eq ptr %in_next, %src_end
  br i1 %done, label %exit, label %loop

exit:
  %final = load ptr, ptr %guard_slot, align 8
  store ptr %final, ptr %finish_slot, align 8
  ret void
}

define internal void @fill_vec2(ptr %dst, ptr readonly %proto) noinline {
entry:
  %buf = call noalias ptr @malloc(i64 48)
  %end_cap = getelementptr %Vec, ptr %buf, i64 2
  %finish_slot = getelementptr %Vec, ptr %dst, i64 0, i32 1
  %cap_slot = getelementptr %Vec, ptr %dst, i64 0, i32 2
  store ptr %buf, ptr %dst, align 8
  store ptr %end_cap, ptr %cap_slot, align 8
  %guard_slot = alloca ptr, align 8
  store ptr %buf, ptr %guard_slot, align 8
  br label %loop

loop:
  %cur = load ptr, ptr %guard_slot, align 8
  call void @copy_vec1(ptr %cur, ptr %proto)
  %next = getelementptr %Vec, ptr %cur, i64 1
  store ptr %next, ptr %guard_slot, align 8
  %done = icmp eq ptr %next, %end_cap
  br i1 %done, label %exit, label %loop

exit:
  %final = load ptr, ptr %guard_slot, align 8
  store ptr %final, ptr %finish_slot, align 8
  ret void
}

define internal void @copy_vec2(ptr %dst, ptr readonly %src) noinline {
entry:
  %src_begin = load ptr, ptr %src, align 8
  %src_finish_slot = getelementptr %Vec, ptr %src, i64 0, i32 1
  %src_end = load ptr, ptr %src_finish_slot, align 8
  %buf = call noalias ptr @malloc(i64 48)
  %end_cap = getelementptr %Vec, ptr %buf, i64 2
  %finish_slot = getelementptr %Vec, ptr %dst, i64 0, i32 1
  %cap_slot = getelementptr %Vec, ptr %dst, i64 0, i32 2
  store ptr %buf, ptr %dst, align 8
  store ptr %end_cap, ptr %cap_slot, align 8
  %guard_slot = alloca ptr, align 8
  store ptr %buf, ptr %guard_slot, align 8
  br label %loop

loop:
  %in_ptr = phi ptr [ %src_begin, %entry ], [ %in_next, %loop ]
  %cur = phi ptr [ %buf, %entry ], [ %next, %loop ]
  call void @copy_vec1(ptr %cur, ptr %in_ptr)
  %next = getelementptr %Vec, ptr %cur, i64 1
  store ptr %next, ptr %guard_slot, align 8
  %in_next = getelementptr %Vec, ptr %in_ptr, i64 1
  %done = icmp eq ptr %in_next, %src_end
  br i1 %done, label %exit, label %loop

exit:
  store ptr %next, ptr %finish_slot, align 8
  ret void
}

define internal void @fill_vec3(ptr %dst, ptr readonly %proto) noinline {
entry:
  %buf = call noalias ptr @malloc(i64 48)
  %end_cap = getelementptr %Vec, ptr %buf, i64 2
  %finish_slot = getelementptr %Vec, ptr %dst, i64 0, i32 1
  %cap_slot = getelementptr %Vec, ptr %dst, i64 0, i32 2
  store ptr %buf, ptr %dst, align 8
  store ptr %end_cap, ptr %cap_slot, align 8
  %guard_slot = alloca ptr, align 8
  store ptr %buf, ptr %guard_slot, align 8
  br label %loop

loop:
  %cur = load ptr, ptr %guard_slot, align 8
  call void @copy_vec2(ptr %cur, ptr %proto)
  %next = getelementptr %Vec, ptr %cur, i64 1
  store ptr %next, ptr %guard_slot, align 8
  %done = icmp eq ptr %next, %end_cap
  br i1 %done, label %exit, label %loop

exit:
  %final = load ptr, ptr %guard_slot, align 8
  store ptr %final, ptr %finish_slot, align 8
  ret void
}

define internal void @copy_vec3(ptr %dst, ptr readonly %src) noinline {
entry:
  %src_begin = load ptr, ptr %src, align 8
  %src_finish_slot = getelementptr %Vec, ptr %src, i64 0, i32 1
  %src_end = load ptr, ptr %src_finish_slot, align 8
  %buf = call noalias ptr @malloc(i64 48)
  %end_cap = getelementptr %Vec, ptr %buf, i64 2
  %finish_slot = getelementptr %Vec, ptr %dst, i64 0, i32 1
  %cap_slot = getelementptr %Vec, ptr %dst, i64 0, i32 2
  store ptr %buf, ptr %dst, align 8
  store ptr %end_cap, ptr %cap_slot, align 8
  %guard_slot = alloca ptr, align 8
  store ptr %buf, ptr %guard_slot, align 8
  br label %loop

loop:
  %in_ptr = phi ptr [ %src_begin, %entry ], [ %in_next, %loop ]
  %cur = phi ptr [ %buf, %entry ], [ %next, %loop ]
  call void @copy_vec2(ptr %cur, ptr %in_ptr)
  %next = getelementptr %Vec, ptr %cur, i64 1
  store ptr %next, ptr %guard_slot, align 8
  %in_next = getelementptr %Vec, ptr %in_ptr, i64 1
  %done = icmp eq ptr %in_next, %src_end
  br i1 %done, label %exit, label %loop

exit:
  store ptr %next, ptr %finish_slot, align 8
  ret void
}

define internal void @fill_vec4(ptr %dst, ptr readonly %proto) noinline {
entry:
  %buf = call noalias ptr @malloc(i64 48)
  %end_cap = getelementptr %Vec, ptr %buf, i64 2
  %finish_slot = getelementptr %Vec, ptr %dst, i64 0, i32 1
  %cap_slot = getelementptr %Vec, ptr %dst, i64 0, i32 2
  store ptr %buf, ptr %dst, align 8
  store ptr %end_cap, ptr %cap_slot, align 8
  %guard_slot = alloca ptr, align 8
  store ptr %buf, ptr %guard_slot, align 8
  br label %loop

loop:
  %cur = load ptr, ptr %guard_slot, align 8
  call void @copy_vec3(ptr %cur, ptr %proto)
  %next = getelementptr %Vec, ptr %cur, i64 1
  store ptr %next, ptr %guard_slot, align 8
  %done = icmp eq ptr %next, %end_cap
  br i1 %done, label %exit, label %loop

exit:
  %final = load ptr, ptr %guard_slot, align 8
  store ptr %final, ptr %finish_slot, align 8
  ret void
}

define double @nested_vec_sum(double %x) {
entry:
  %v1 = alloca %Vec, align 8
  %v2 = alloca %Vec, align 8
  %v3 = alloca %Vec, align 8
  %v4 = alloca %Vec, align 8
  call void @init_leaf_vec(ptr %v1, double %x)
  call void @fill_vec2(ptr %v2, ptr %v1)
  call void @fill_vec3(ptr %v3, ptr %v2)
  call void @fill_vec4(ptr %v4, ptr %v3)

  %p4 = load ptr, ptr %v4, align 8
  %p3_0 = load ptr, ptr %p4, align 8
  %p2_0 = load ptr, ptr %p3_0, align 8
  %p1_0 = load ptr, ptr %p2_0, align 8
  %val0 = load double, ptr %p1_0, align 8

  %v4_1 = getelementptr %Vec, ptr %p4, i64 1
  %p3_1_base = load ptr, ptr %v4_1, align 8
  %v3_1 = getelementptr %Vec, ptr %p3_1_base, i64 1
  %p2_1_base = load ptr, ptr %v3_1, align 8
  %v2_1 = getelementptr %Vec, ptr %p2_1_base, i64 1
  %p1_1_base = load ptr, ptr %v2_1, align 8
  %elem_1 = getelementptr double, ptr %p1_1_base, i64 1
  %val1 = load double, ptr %elem_1, align 8

  %t0 = fmul double %val0, 3.000000e+00
  %t1 = fmul double %val1, 5.000000e+00
  %res = fadd double %t0, %t1
  ret double %res
}

define double @grad_nested_vec(double %x) {
entry:
  %dx = call double (ptr, ...) @__enzyme_autodiff(ptr @nested_vec_sum, double %x)
  ret double %dx
}

define i32 @main() {
entry:
  %dx = call double @grad_nested_vec(double 2.000000e+00)
  %diff = fsub double %dx, 8.000000e+00
  %neg = fsub double -0.000000e+00, %diff
  %is_neg = fcmp olt double %diff, 0.000000e+00
  %abs = select i1 %is_neg, double %neg, double %diff
  %ok = fcmp olt double %abs, 1.000000e-06
  br i1 %ok, label %done, label %fail

fail:
  call void @abort()
  unreachable

done:
  ret i32 0
}

; CHECK: define internal { double } @diffenested_vec_sum(double %x, double %differeturn)
; CHECK: call { double } @diffeinit_leaf_vec(
