; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S 2>&1 | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S 2>&1 | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL

; Verify that deallocations of pointers merged through PHINodes (both in loops
; and across branches) are recognized as guaranteed frees without emitting
; spurious "freeing without malloc" warnings, leaking memory, or double-freeing
; when only a subset of incoming allocations are needed in the reverse pass.

; CHECK-NOT: freeing without malloc
; EVAL: 1.000000 7.000000 28.000000 0

@enzyme_const = global i32 0
@enzyme_dupnoneed = global i32 0
@live_allocations = global i64 0, align 8
@.fmt = private unnamed_addr constant [13 x i8] c"%f %f %f %ld\00", align 1

declare void @__enzyme_autodiff0(...)
declare double @__enzyme_autodiff(ptr, ...)
declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare void @abort()
declare i32 @printf(ptr, ...)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

define noalias ptr @_mlir_memref_to_llvm_alloc(i64 %size) noinline {
entry:
  %current = load i64, ptr @live_allocations, align 8
  %incremented = add i64 %current, 1
  store i64 %incremented, ptr @live_allocations, align 8
  %total = add i64 %size, 8
  %raw = call noalias ptr @malloc(i64 %total)
  store i64 3405691582, ptr %raw, align 8
  %user = getelementptr inbounds i8, ptr %raw, i64 8
  ret ptr %user
}

define void @_mlir_memref_to_llvm_free(ptr nocapture %user) noinline {
entry:
  %raw = getelementptr inbounds i8, ptr %user, i64 -8
  %magic = load i64, ptr %raw, align 8
  %valid_magic = icmp eq i64 %magic, 3405691582
  br i1 %valid_magic, label %valid, label %invalid

invalid:
  call void @abort()
  unreachable

valid:
  store i64 3735928559, ptr %raw, align 8
  store double -999999.0, ptr %user, align 8
  call void @free(ptr %raw)
  %current = load i64, ptr @live_allocations, align 8
  %decremented = sub i64 %current, 1
  store i64 %decremented, ptr @live_allocations, align 8
  ret void
}

define void @mlir_loop_phi_free(ptr %dst, ptr %active_unused) {
.start:
  %first = call ptr @_mlir_memref_to_llvm_alloc(i64 32)
  br label %.critedge

.critedge:
  %.pn1130 = phi ptr [ %first, %.start ], [ %second, %.critedge ]
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %.pn1130, i64 1, i1 false)
  call void @_mlir_memref_to_llvm_free(ptr %.pn1130)
  %second = call ptr @_mlir_memref_to_llvm_alloc(i64 32)
  br label %.critedge
}

define void @compile_infinite_loop_grad() {
entry:
  call void (...) @__enzyme_autodiff0(ptr @mlir_loop_phi_free, ptr @enzyme_const, ptr null, ptr @enzyme_dupnoneed, ptr null, ptr null)
  ret void
}

define void @opaque_fill(ptr nocapture %buffer, double %value) noinline {
entry:
  store double %value, ptr %buffer, align 8
  ret void
}

define double @mixed_diamond(double %x, i1 %cond, ptr nocapture writeonly %sink) {
entry:
  br i1 %cond, label %forward_only, label %needed_in_reverse

forward_only:
  %alloc_a = call noalias ptr @_mlir_memref_to_llvm_alloc(i64 8)
  call void @opaque_fill(ptr %alloc_a, double 5.000000e+00)
  %val_a = load double, ptr %alloc_a, align 8
  store double %val_a, ptr %sink, align 8
  %res_a = fadd double %val_a, %x
  br label %merge

needed_in_reverse:
  %alloc_b = call noalias ptr @_mlir_memref_to_llvm_alloc(i64 8)
  call void @opaque_fill(ptr %alloc_b, double 7.000000e+00)
  %val_b = load double, ptr %alloc_b, align 8
  %res_b = fmul double %val_b, %x
  br label %merge

merge:
  %merged_ptr = phi ptr [ %alloc_a, %forward_only ], [ %alloc_b, %needed_in_reverse ]
  %merged_res = phi double [ %res_a, %forward_only ], [ %res_b, %needed_in_reverse ]
  call void @_mlir_memref_to_llvm_free(ptr %merged_ptr)
  ret double %merged_res
}

define double @loop_carried_alloc(double %x, i64 %count) {
start:
  %first = call noalias ptr @_mlir_memref_to_llvm_alloc(i64 8)
  store double %x, ptr %first, align 8
  br label %loop

loop:
  %iv = phi i64 [ 0, %start ], [ %iv.next, %loop ]
  %current = phi ptr [ %first, %start ], [ %next, %loop ]
  %accum = phi double [ 0.000000e+00, %start ], [ %accum.next, %loop ]
  %val = load double, ptr %current, align 8
  %sq = fmul double %val, %val
  %accum.next = fadd double %accum, %sq
  call void @_mlir_memref_to_llvm_free(ptr %current)
  %next = call noalias ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %val.next = fadd double %val, 1.000000e+00
  store double %val.next, ptr %next, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %continue = icmp ult i64 %iv.next, %count
  br i1 %continue, label %loop, label %exit

exit:
  call void @_mlir_memref_to_llvm_free(ptr %next)
  ret double %accum.next
}

define i32 @main() {
entry:
  %sink = alloca double, align 8
  %d1 = call double (ptr, ...) @__enzyme_autodiff(ptr @mixed_diamond, double 3.000000e+00, ptr @enzyme_const, i1 true, ptr @enzyme_const, ptr %sink)
  %d2 = call double (ptr, ...) @__enzyme_autodiff(ptr @mixed_diamond, double 3.000000e+00, ptr @enzyme_const, i1 false, ptr @enzyme_const, ptr %sink)
  %d3 = call double (ptr, ...) @__enzyme_autodiff(ptr @loop_carried_alloc, double 2.000000e+00, i64 4)
  %remaining = load i64, ptr @live_allocations, align 8
  call i32 (ptr, ...) @printf(ptr @.fmt, double %d1, double %d2, double %d3, i64 %remaining)
  ret i32 0
}

; CHECK-LABEL: define internal void @diffemlir_loop_phi_free(
; CHECK: .start:
; CHECK:   %[[FIRST:.+]] = call ptr @_mlir_memref_to_llvm_alloc(i64 32)
; CHECK: .critedge:
; CHECK:   %[[PHI:.+]] = phi ptr [ %[[FIRST]], %.start ], [ %[[SECOND:.+]], %.critedge ]
; CHECK:   call void @_mlir_memref_to_llvm_free(ptr %[[PHI]])
; CHECK:   %[[SECOND]] = call ptr @_mlir_memref_to_llvm_alloc(i64 32)
; CHECK: invert.start:
; CHECK-NOT: call void @_mlir_memref_to_llvm_free(
; CHECK:   ret void
