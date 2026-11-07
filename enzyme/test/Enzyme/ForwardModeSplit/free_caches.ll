; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | %lli -
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | %lli -

%struct.Node = type { ptr, ptr }

declare noalias ptr @malloc(i64)
declare noalias ptr @calloc(i64, i64)
declare void @free(ptr)
declare void @abort()
declare double @llvm.fabs.f64(double)

declare { ptr, double } @__enzyme_augmentfwd(...)
declare double @__enzyme_fwdsplit(...)

; 1. Nested dynamic outer loop + static inner loop with preheader bypass.
define double @nested_list_sum(ptr noalias readonly %node, i64 %times) noinline {
entry:
  %is_null = icmp eq ptr %node, null
  br i1 %is_null, label %cleanup, label %outer.header

outer.header:
  %cur = phi ptr [ %next_node, %outer.latch ], [ %node, %entry ]
  %outer_sum = phi double [ %inner_add, %outer.latch ], [ 0.000000e+00, %entry ]
  %vals_slot = getelementptr inbounds %struct.Node, ptr %cur, i64 0, i32 0
  %vals = load ptr, ptr %vals_slot, align 8
  br label %inner.body

inner.body:
  %iv = phi i64 [ 0, %outer.header ], [ %iv_next, %inner.body ]
  %inner_sum = phi double [ %outer_sum, %outer.header ], [ %inner_add, %inner.body ]
  %elem_ptr = getelementptr inbounds double, ptr %vals, i64 %iv
  %elem = load double, ptr %elem_ptr, align 8
  %sq = fmul double %elem, %elem
  %inner_add = fadd double %inner_sum, %sq
  %iv_next = add nuw i64 %iv, 1
  %done_inner = icmp eq i64 %iv, %times
  br i1 %done_inner, label %outer.latch, label %inner.body

outer.latch:
  %next_slot = getelementptr inbounds %struct.Node, ptr %cur, i64 0, i32 1
  %next_node = load ptr, ptr %next_slot, align 8
  %done_outer = icmp eq ptr %next_node, null
  br i1 %done_outer, label %cleanup, label %outer.header

cleanup:
  %result = phi double [ 0.000000e+00, %entry ], [ %inner_add, %outer.latch ]
  ret double %result
}

; 2. Loop with both preheader bypass and multi-exit early break.
define double @early_break_cache(ptr noalias readonly %arr, i64 %n, i64 %stop_at) noinline {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i_next, %latch ]
  %acc = phi double [ 1.000000e+00, %entry ], [ %next_acc, %latch ]
  %gep = getelementptr inbounds double, ptr %arr, i64 %i
  %val = load double, ptr %gep, align 8
  %prod = fmul double %acc, %val
  %next_acc = fadd double %prod, %val
  %hit_stop = icmp eq i64 %i, %stop_at
  br i1 %hit_stop, label %exit, label %latch

latch:
  %i_next = add nuw i64 %i, 1
  %done = icmp eq i64 %i_next, %n
  br i1 %done, label %exit, label %loop

exit:
  %res = phi double [ 1.000000e+00, %entry ], [ %next_acc, %loop ], [ %next_acc, %latch ]
  ret double %res
}

; Helper writing two nonlinear outputs into a caller-provided buffer.
define void @fill_buf(ptr noalias nocapture %buf, double %x) noinline {
entry:
  %sq = fmul double %x, %x
  store double %sq, ptr %buf, align 8
  %p1 = getelementptr inbounds double, ptr %buf, i64 1
  %triple = fmul double %x, 3.000000e+00
  store double %triple, ptr %p1, align 8
  ret void
}

; 3. Entry alloca passed to non-inlined subcall (!enzyme_fromstack heap upgrade).
define double @subcall_fromstack(double %x) noinline {
entry:
  %buf = alloca [2 x double], align 8
  call void @fill_buf(ptr %buf, double %x)
  %v0 = load double, ptr %buf, align 8
  %p1 = getelementptr inbounds double, ptr %buf, i64 1
  %v1 = load double, ptr %p1, align 8
  %prod = fmul double %v0, %v1
  ret double %prod
}

; 4. Local shadow-only allocation without explicit free in primal.
define double @local_calloc_no_free(double %x, i64 %count) noinline {
entry:
  %mem = call noalias ptr @calloc(i64 8, i64 %count)
  store double %x, ptr %mem, align 8
  %loaded = load double, ptr %mem, align 8
  %res = fmul double %loaded, %x
  ret double %res
}

; 5. In-loop explicit malloc + free (rematerialized primal and local shadow).
define double @loop_malloc_free(double %x, i64 %n) noinline {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i_next, %loop ]
  %sum = phi double [ 0.000000e+00, %entry ], [ %next_sum, %loop ]
  %mem = call noalias ptr @malloc(i64 8)
  %idx_fp = uitofp i64 %i to double
  %scale = fadd double %idx_fp, 1.000000e+00
  %scaled = fmul double %x, %scale
  store double %scaled, ptr %mem, align 8
  %loaded = load double, ptr %mem, align 8
  %term = fmul double %loaded, %loaded
  call void @free(ptr %mem)
  %next_sum = fadd double %sum, %term
  %i_next = add nuw i64 %i, 1
  %done = icmp eq i64 %i_next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret double %next_sum
}

; 6. Whole-allocation tape cache with post-dominating free, tested with both
;    enzyme_nofree (reusing tape) and default freeing in the same module.
define double @tape_cached_whole_alloc(double %x) noinline {
entry:
  %mem = call noalias ptr @malloc(i64 16)
  call void @fill_buf(ptr %mem, double %x)
  %v0 = load double, ptr %mem, align 8
  %p1 = getelementptr inbounds double, ptr %mem, i64 1
  %v1 = load double, ptr %p1, align 8
  %prod = fmul double %v0, %v1
  call void @free(ptr %mem)
  ret double %prod
}

; 7. Subcall returning heap allocation freed by the caller.
define noalias nonnull ptr @make_heap_box(double %x) noinline {
entry:
  %mem = call noalias nonnull ptr @malloc(i64 8)
  store double %x, ptr %mem, align 8
  ret ptr %mem
}

define double @caller_frees_returned_heap_ptr(double %x) noinline {
entry:
  %box = call ptr @make_heap_box(double %x)
  %v = load double, ptr %box, align 8
  %prod = fmul double %v, %x
  call void @free(ptr %box)
  ret double %prod
}

define void @check_close(double %actual, double %expected) noinline {
entry:
  %diff = fsub double %actual, %expected
  %abs = call double @llvm.fabs.f64(double %diff)
  %bad = fcmp ogt double %abs, 1.000000e-09
  br i1 %bad, label %fail, label %ok

fail:
  call void @abort()
  unreachable

ok:
  ret void
}

define i32 @main() {
entry:
  ; --- Test 1a: nested_list_sum with null head (0 outer iterations) ---
  %aug_null = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @nested_list_sum, ptr null, ptr null, i64 2)
  %tape_null = extractvalue { ptr, double } %aug_null, 0
  %d_null = call double (...) @__enzyme_fwdsplit(ptr @nested_list_sum, ptr null, ptr null, i64 2, ptr %tape_null)
  call void @check_close(double %d_null, double 0.000000e+00)

  ; --- Test 1b: nested_list_sum with 2 nodes x 2 elements ---
  %v1 = alloca [2 x double], align 8
  %dv1 = alloca [2 x double], align 8
  %v2 = alloca [2 x double], align 8
  %dv2 = alloca [2 x double], align 8
  %n1 = alloca %struct.Node, align 8
  %dn1 = alloca %struct.Node, align 8
  %n2 = alloca %struct.Node, align 8
  %dn2 = alloca %struct.Node, align 8

  store double 2.000000e+00, ptr %v1, align 8
  %v1_1 = getelementptr inbounds double, ptr %v1, i64 1
  store double 3.000000e+00, ptr %v1_1, align 8
  store double 1.000000e+00, ptr %dv1, align 8
  %dv1_1 = getelementptr inbounds double, ptr %dv1, i64 1
  store double 2.000000e+00, ptr %dv1_1, align 8

  store double 4.000000e+00, ptr %v2, align 8
  %v2_1 = getelementptr inbounds double, ptr %v2, i64 1
  store double 5.000000e+00, ptr %v2_1, align 8
  store double 3.000000e+00, ptr %dv2, align 8
  %dv2_1 = getelementptr inbounds double, ptr %dv2, i64 1
  store double 4.000000e+00, ptr %dv2_1, align 8

  store ptr %v2, ptr %n2, align 8
  %n2_next = getelementptr inbounds %struct.Node, ptr %n2, i64 0, i32 1
  store ptr null, ptr %n2_next, align 8
  store ptr %dv2, ptr %dn2, align 8
  %dn2_next = getelementptr inbounds %struct.Node, ptr %dn2, i64 0, i32 1
  store ptr null, ptr %dn2_next, align 8

  store ptr %v1, ptr %n1, align 8
  %n1_next = getelementptr inbounds %struct.Node, ptr %n1, i64 0, i32 1
  store ptr %n2, ptr %n1_next, align 8
  store ptr %dv1, ptr %dn1, align 8
  %dn1_next = getelementptr inbounds %struct.Node, ptr %dn1, i64 0, i32 1
  store ptr %dn2, ptr %dn1_next, align 8

  ; sum = 2^2 + 3^2 + 4^2 + 5^2; dsum = 2*(2*1 + 3*2 + 4*3 + 5*4) = 80
  %aug_list = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @nested_list_sum, ptr %n1, ptr %dn1, i64 1)
  %tape_list = extractvalue { ptr, double } %aug_list, 0
  %d_list = call double (...) @__enzyme_fwdsplit(ptr @nested_list_sum, ptr %n1, ptr %dn1, i64 1, ptr %tape_list)
  call void @check_close(double %d_list, double 8.000000e+01)

  ; --- Test 2: early_break_cache (bypass, early break, and full loop) ---
  %arr = alloca [3 x double], align 8
  %darr = alloca [3 x double], align 8
  store double 2.000000e+00, ptr %arr, align 8
  %arr1 = getelementptr inbounds double, ptr %arr, i64 1
  store double 3.000000e+00, ptr %arr1, align 8
  %arr2 = getelementptr inbounds double, ptr %arr, i64 2
  store double 4.000000e+00, ptr %arr2, align 8
  store double 1.000000e+00, ptr %darr, align 8
  %darr1 = getelementptr inbounds double, ptr %darr, i64 1
  store double 1.000000e+00, ptr %darr1, align 8
  %darr2 = getelementptr inbounds double, ptr %darr, i64 2
  store double 1.000000e+00, ptr %darr2, align 8

  %aug_eb0 = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @early_break_cache, ptr %arr, ptr %darr, i64 0, i64 5)
  %tape_eb0 = extractvalue { ptr, double } %aug_eb0, 0
  %d_eb0 = call double (...) @__enzyme_fwdsplit(ptr @early_break_cache, ptr %arr, ptr %darr, i64 0, i64 5, ptr %tape_eb0)
  call void @check_close(double %d_eb0, double 0.000000e+00)

  ; stop_at = 1: i=0 -> acc = 1*2 + 2 = 4, dacc = 2*1 = 2;
  ;              i=1 -> acc = 4*3 + 3 = 15, dacc = 2*3 + 4*1 + 1 = 11
  %aug_eb1 = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @early_break_cache, ptr %arr, ptr %darr, i64 3, i64 1)
  %tape_eb1 = extractvalue { ptr, double } %aug_eb1, 0
  %d_eb1 = call double (...) @__enzyme_fwdsplit(ptr @early_break_cache, ptr %arr, ptr %darr, i64 3, i64 1, ptr %tape_eb1)
  call void @check_close(double %d_eb1, double 1.100000e+01)

  ; --- Test 3: subcall_fromstack (3 * x^3 -> d = 9 * x^2 * dx = 54 at x=2, dx=1.5) ---
  %aug_fs = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @subcall_fromstack, metadata !"enzyme_dup", double 2.000000e+00, double 1.500000e+00)
  %tape_fs = extractvalue { ptr, double } %aug_fs, 0
  %d_fs = call double (...) @__enzyme_fwdsplit(ptr @subcall_fromstack, double 2.000000e+00, double 1.500000e+00, ptr %tape_fs)
  call void @check_close(double %d_fs, double 5.400000e+01)

  ; --- Test 4: local_calloc_no_free (x^2 -> d = 2 * x * dx = 12 at x=3, dx=2) ---
  %aug_cal = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @local_calloc_no_free, metadata !"enzyme_dup", double 3.000000e+00, double 2.000000e+00, i64 2)
  %tape_cal = extractvalue { ptr, double } %aug_cal, 0
  %d_cal = call double (...) @__enzyme_fwdsplit(ptr @local_calloc_no_free, double 3.000000e+00, double 2.000000e+00, i64 2, ptr %tape_cal)
  call void @check_close(double %d_cal, double 1.200000e+01)

  ; --- Test 5: loop_malloc_free (sum_{i=0..2} ((i+1)*x)^2 = 14*x^2 -> d = 28*x*dx = 56 at x=2, dx=1) ---
  %aug_lmf = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @loop_malloc_free, metadata !"enzyme_dup", double 2.000000e+00, double 1.000000e+00, i64 3)
  %tape_lmf = extractvalue { ptr, double } %aug_lmf, 0
  %d_lmf = call double (...) @__enzyme_fwdsplit(ptr @loop_malloc_free, double 2.000000e+00, double 1.000000e+00, i64 3, ptr %tape_lmf)
  call void @check_close(double %d_lmf, double 5.600000e+01)

  ; --- Test 6: tape_cached_whole_alloc reused with enzyme_nofree then freed ---
  %aug_tc = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @tape_cached_whole_alloc, metadata !"enzyme_dup", double 2.000000e+00, double 1.000000e+00)
  %tape_tc = extractvalue { ptr, double } %aug_tc, 0
  %d_tc_nofree = call double (...) @__enzyme_fwdsplit(ptr @tape_cached_whole_alloc, metadata !"enzyme_nofree", double 2.000000e+00, double 1.000000e+00, ptr %tape_tc)
  call void @check_close(double %d_tc_nofree, double 3.600000e+01)
  %d_tc_free = call double (...) @__enzyme_fwdsplit(ptr @tape_cached_whole_alloc, double 2.000000e+00, double 2.000000e+00, ptr %tape_tc)
  call void @check_close(double %d_tc_free, double 7.200000e+01)

  ; --- Test 7: caller_frees_returned_heap_ptr (x^2 -> d = 2 * x * dx = 10 at x=5, dx=1) ---
  %aug_box = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @caller_frees_returned_heap_ptr, metadata !"enzyme_dup", double 5.000000e+00, double 1.000000e+00)
  %tape_box = extractvalue { ptr, double } %aug_box, 0
  %d_box = call double (...) @__enzyme_fwdsplit(ptr @caller_frees_returned_heap_ptr, double 5.000000e+00, double 1.000000e+00, ptr %tape_box)
  call void @check_close(double %d_box, double 1.000000e+01)

  ret i32 0
}

; CHECK-LABEL: define internal double @fwddiffenested_list_sum(
; CHECK: entry:
; CHECK:   %truetape = load { ptr, ptr, ptr, ptr }, ptr %tapeArg
; CHECK:   call void @free(ptr {{(nonnull )?}}%tapeArg)
; CHECK: cleanup.loopexit:
; CHECK:   call void @free(ptr
; CHECK:   call void @free(ptr
; CHECK:   call void @free(ptr
; CHECK:   call void @free(ptr
; CHECK: cleanup:
; CHECK-NOT: call void @free(
; CHECK:   ret double

; CHECK-LABEL: define internal double @fwddiffeearly_break_cache(
; CHECK: entry:
; CHECK:   %truetape = load { ptr, ptr }, ptr %tapeArg
; CHECK:   call void @free(ptr {{(nonnull )?}}%tapeArg)
; CHECK: exit.loopexit:
; CHECK:   call void @free(ptr
; CHECK:   call void @free(ptr
; CHECK: exit:
; CHECK-NOT: call void @free(
; CHECK:   ret double

; CHECK-LABEL: define internal double @fwddiffesubcall_fromstack(
; CHECK:   %[[BUF:.+]] = load ptr, ptr %tapeArg
; CHECK:   call void @free(ptr {{(nonnull )?}}%tapeArg)
; CHECK:   call void @free(ptr %[[BUF]])
; CHECK-NEXT:   ret double

; CHECK-LABEL: define internal double @fwddiffelocal_calloc_no_free(
; CHECK:   %[[MEM:.+]] = call noalias ptr @calloc(i64 8, i64 %count)
; CHECK:   %[[SHADOW:.+]] = call noalias nonnull ptr @calloc(i64 8, i64 %count)
; CHECK:   call void @free(ptr nonnull %[[SHADOW]])
; CHECK:   call void @free(ptr %[[MEM]])
; CHECK-NEXT:   ret double

; CHECK-LABEL: define internal double @fwddiffeloop_malloc_free(
; CHECK: loop:
; CHECK:   %[[LMEM:.+]] = call {{.*}}ptr @malloc(i64 8)
; CHECK:   %[[LSHAD:.+]] = call {{.*}}ptr @malloc(i64 8)
; CHECK:   call void @free(ptr %[[LMEM]])
; CHECK:   call void @free(ptr %[[LSHAD]])
; CHECK: exit:
; CHECK:   ret double

; First instantiation (enzyme_nofree): frees local scratch shadow, preserves tapeArg and cached %mem.
; CHECK-LABEL: define internal double @fwddiffetape_cached_whole_alloc(
; CHECK:   %[[NMEM:.+]] = load ptr, ptr %tapeArg
; CHECK-NOT: call void @free(ptr {{.*}}%tapeArg)
; CHECK:   %[[NSHAD:.+]] = call {{.*}}ptr @malloc(i64 16)
; CHECK-NOT: call void @free(ptr {{.*}}%[[NMEM]])
; CHECK:   call void @free(ptr %[[NSHAD]])
; CHECK-NOT: call void @free(
; CHECK:   ret double

; Second instantiation (freeMemory=true): frees tapeArg, cached %mem, and local scratch shadow.
; CHECK-LABEL: define internal double @fwddiffetape_cached_whole_alloc.{{[0-9]+}}(
; CHECK:   %[[FMEM:.+]] = load ptr, ptr %tapeArg
; CHECK:   call void @free(ptr {{(nonnull )?}}%tapeArg)
; CHECK:   %[[FSHAD:.+]] = call {{.*}}ptr @malloc(i64 16)
; CHECK:   call void @free(ptr %[[FMEM]])
; CHECK:   call void @free(ptr %[[FSHAD]])
; CHECK:   ret double

; CHECK-LABEL: define internal double @fwddiffecaller_frees_returned_heap_ptr(
; CHECK:   call void @free(ptr {{(nonnull )?}}%tapeArg)
; CHECK:   %[[CALLRET:.+]] = call { ptr, ptr } @fwddiffemake_heap_box(
; CHECK:   %[[RBOX:.+]] = extractvalue { ptr, ptr } %[[CALLRET]], 0
; CHECK:   %[[RSBOX:.+]] = extractvalue { ptr, ptr } %[[CALLRET]], 1
; CHECK:   call void @free(ptr %[[RBOX]])
; CHECK:   call void @free(ptr %[[RSBOX]])
; CHECK:   ret double

; CHECK-LABEL: define internal { ptr, ptr } @fwddiffemake_heap_box(
; CHECK-NOT: call void @free(
; CHECK:   ret { ptr, ptr }
