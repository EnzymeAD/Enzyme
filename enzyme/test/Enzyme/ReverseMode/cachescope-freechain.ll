; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; In the reverse pass of @g, %p is needed for the derivative of %q. Unwrapping
; %p first unwraps the branch condition %cond, whose phi %cphi moves the
; builder into a new block. Unwrapping %p then gives up, erasing the blocks it
; made, including the lookups of %x from the cache that the reverse pass had
; just created for it. When %x is later restored from the tape,
; cacheForReverse removes that cache: its stores and frees. The free was the
; last user of the cache allocation, so erasing the operands of the free also
; erased the allocation, while scopeMap still held it in an AssertingVH
; ("An asserting value handle still pointed to this value!" on a build with
; LLVM_ENABLE_ABI_BREAKING_CHECKS, a use after free otherwise).

declare ptr @__enzyme_augmentfwd(...)
declare void @__enzyme_reverse(...)

define void @g(ptr noalias %a, ptr noalias %out, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  %par = and i64 %i, 1
  %c0 = icmp eq i64 %par, 0
  %pa = getelementptr inbounds double, ptr %a, i64 %i
  %x = load double, ptr %pa, align 8
  br i1 %c0, label %T, label %J

T:
  %v = load double, ptr %pa, align 8
  %vv = fmul double %v, %v
  %pb = getelementptr inbounds double, ptr %out, i64 %n
  %po2 = getelementptr inbounds double, ptr %pb, i64 %i
  store double %vv, ptr %po2, align 8
  br label %J

J:
  %cphi = phi double [ %v, %T ], [ 1.000000e+00, %loop ]
  %cond = fcmp ogt double %cphi, 0.000000e+00
  br i1 %cond, label %merge, label %Q

Q:
  %x2 = fmul double %x, 3.000000e+00
  br label %merge

merge:
  %p = phi double [ %x, %J ], [ %x2, %Q ]
  %q = fmul double %p, %p
  store double 0.000000e+00, ptr %pa, align 8
  %po = getelementptr inbounds double, ptr %out, i64 %i
  store double %q, ptr %po, align 8
  br label %latch

latch:
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @dg(ptr %a, ptr %da, ptr %out, ptr %dout, i64 %n) {
entry:
  %tape = call ptr (...) @__enzyme_augmentfwd(ptr @g, metadata !"enzyme_dup", ptr %a, ptr %da, metadata !"enzyme_dup", ptr %out, ptr %dout, i64 %n)
  call void (...) @__enzyme_reverse(ptr @g, metadata !"enzyme_dup", ptr %a, ptr %da, metadata !"enzyme_dup", ptr %out, ptr %dout, i64 %n, ptr %tape)
  ret void
}

; CHECK: define internal ptr @augmented_g(ptr noalias {{.*}}%a, ptr {{.*}}%"a'", ptr noalias {{.*}}%out, ptr {{.*}}%"out'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %tapemem = tail call noalias nonnull {{.*}}ptr @malloc(i64 16)
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %n, 8
; The augmented forward pass caches %x for every iteration and passes the
; cache to the reverse pass on the tape.
; CHECK-NEXT:   %x_malloccache = tail call noalias nonnull ptr @malloc(i64 %mallocsize)
; CHECK-NEXT:   store ptr %x_malloccache, ptr %tapemem, align 8
; CHECK-NEXT:   %mallocsize6 = mul nuw nsw i64 %n, 8
; CHECK-NEXT:   %v_malloccache = tail call noalias nonnull ptr @malloc(i64 %mallocsize6)
; CHECK-NEXT:   [[A0:%[0-9]+]] = getelementptr inbounds { ptr, ptr }, ptr %tapemem, i32 0, i32 1
; CHECK-NEXT:   store ptr %v_malloccache, ptr [[A0]], align 8
; CHECK-NEXT:   br label %loop
; CHECK-EMPTY:
; CHECK-NEXT: loop:
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %J ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %par = and i64 %iv, 1
; CHECK-NEXT:   %c0 = icmp eq i64 %par, 0
; CHECK-NEXT:   %pa = getelementptr inbounds double, ptr %a, i64 %iv
; CHECK-NEXT:   %x = load double, ptr %pa, align 8
; CHECK-NEXT:   [[A1:%[0-9]+]] = getelementptr inbounds double, ptr %x_malloccache, i64 %iv
; CHECK-NEXT:   store double %x, ptr [[A1]], align 8
; CHECK-NEXT:   br i1 %c0, label %T, label %J
; CHECK-EMPTY:
; CHECK-NEXT: T:
; CHECK-NEXT:   %v = load double, ptr %pa, align 8
; CHECK-NEXT:   %vv = fmul double %v, %v
; CHECK-NEXT:   %pb = getelementptr inbounds double, ptr %out, i64 %n
; CHECK-NEXT:   %po2 = getelementptr inbounds double, ptr %pb, i64 %iv
; CHECK-NEXT:   store double %vv, ptr %po2, align 8
; CHECK-NEXT:   [[A2:%[0-9]+]] = getelementptr inbounds double, ptr %v_malloccache, i64 %iv
; CHECK-NEXT:   store double %v, ptr [[A2]], align 8
; CHECK-NEXT:   br label %J
; CHECK-EMPTY:
; CHECK-NEXT: J:
; CHECK-NEXT:   %cphi = phi double [ %v, %T ], [ 1.000000e+00, %loop ]
; CHECK-NEXT:   %cond = fcmp ogt double %cphi, 0.000000e+00
; CHECK-NEXT:   %x2 = fmul double %x, 3.000000e+00
; CHECK-NEXT:   %spec.select = select i1 %cond, double %x, double %x2
; CHECK-NEXT:   %q = fmul double %spec.select, %spec.select
; CHECK-NEXT:   store double 0.000000e+00, ptr %pa, align 8
; CHECK-NEXT:   %po = getelementptr inbounds double, ptr %out, i64 %iv
; CHECK-NEXT:   store double %q, ptr %po, align 8
; CHECK-NEXT:   %done = icmp eq i64 %iv.next, %n
; CHECK-NEXT:   br i1 %done, label %exit, label %loop
; CHECK-EMPTY:
; CHECK-NEXT: exit:
; CHECK-NEXT:   ret ptr %tapemem
; CHECK-NEXT: }

; CHECK: define internal void @diffeg(ptr noalias {{.*}}%a, ptr {{.*}}%"a'", ptr noalias {{.*}}%out, ptr {{.*}}%"out'", i64 %n, ptr %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %truetape = load { ptr, ptr }, ptr %tapeArg, align 8
; CHECK-NEXT:   tail call void @free(ptr nonnull %tapeArg)
; Relevant part: the reverse pass reads %x from the cache on the tape. It had
; created its own cache for %x while unwrapping %p; cacheForReverse replaced
; that cache with the tape and removed its stores and frees. The cache
; allocation stays until then (it was erased with the free before).
; CHECK-NEXT:   [[R0:%[0-9]+]] = extractvalue { ptr, ptr } %truetape, 0
; CHECK-NEXT:   [[R1:%[0-9]+]] = extractvalue { ptr, ptr } %truetape, 1
; CHECK-NEXT:   [[R2:%[0-9]+]] = add i64 %n, -1
; CHECK-NEXT:   %mallocsize31 = mul nuw nsw i64 %n, 8
; CHECK-NEXT:   %p_malloccache = tail call noalias nonnull ptr @malloc(i64 %mallocsize31)
; CHECK-NEXT:   br label %loop
; CHECK-EMPTY:
; CHECK-NEXT: loop:
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %J ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %par = and i64 %iv, 1
; CHECK-NEXT:   %c0 = icmp eq i64 %par, 0
; CHECK-NEXT:   [[R3:%[0-9]+]] = getelementptr inbounds double, ptr [[R0]], i64 %iv
; %x comes from the tape, not from a cache of the reverse pass.
; CHECK-NEXT:   %x = load double, ptr [[R3]], align 8
; CHECK-NEXT:   br i1 %c0, label %T, label %J
; CHECK-EMPTY:
; CHECK-NEXT: T:
; CHECK-NEXT:   [[R4:%[0-9]+]] = getelementptr inbounds double, ptr [[R1]], i64 %iv
; CHECK-NEXT:   %v = load double, ptr [[R4]], align 8
; CHECK-NEXT:   br label %J
; CHECK-EMPTY:
; CHECK-NEXT: J:
; CHECK-NEXT:   %cphi = phi double [ %v, %T ], [ 1.000000e+00, %loop ]
; CHECK-NEXT:   %cond = fcmp ogt double %cphi, 0.000000e+00
; CHECK-NEXT:   %x2 = fmul double %x, 3.000000e+00
; CHECK-NEXT:   %spec.select = select i1 %cond, double %x, double %x2
; CHECK-NEXT:   [[R5:%[0-9]+]] = getelementptr inbounds double, ptr %p_malloccache, i64 %iv
; CHECK-NEXT:   store double %spec.select, ptr [[R5]], align 8
; CHECK-NEXT:   %done = icmp eq i64 %iv.next, %n
; CHECK-NEXT:   br i1 %done, label %invertlatch, label %loop
; CHECK-EMPTY:
; CHECK-NEXT: invertentry:
; CHECK-NEXT:   tail call void @free(ptr nonnull %p_malloccache)
; Only the tape caches are freed; no free of the removed cache remains.
; CHECK-NEXT:   tail call void @free(ptr nonnull [[R0]])
; CHECK-NEXT:   tail call void @free(ptr nonnull [[R1]])
; CHECK-NEXT:   ret void
; CHECK-EMPTY:
; CHECK-NEXT: invertloop:
; CHECK-NEXT:   %"v'de.0" = phi double [ 0.000000e+00, %invertT ], [ %"v'de.1", %invertmerge_phimerge ]
; CHECK-NEXT:   %"vv'de.0" = phi double [ 0.000000e+00, %invertT ], [ %"vv'de.1", %invertmerge_phimerge ]
; CHECK-NEXT:   %"pa'ipg_unwrap" = getelementptr inbounds double, ptr %"a'", i64 %"iv'ac.0"
; CHECK-NEXT:   [[R6:%[0-9]+]] = load double, ptr %"pa'ipg_unwrap", align 8
; CHECK-NEXT:   [[R7:%[0-9]+]] = fadd fast double [[R6]], %"x'de.0"
; CHECK-NEXT:   store double [[R7]], ptr %"pa'ipg_unwrap", align 8
; CHECK-NEXT:   [[R8:%[0-9]+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 [[R8]], label %invertentry, label %incinvertloop
; CHECK-EMPTY:
; CHECK-NEXT: incinvertloop:
; CHECK-NEXT:   [[R9:%[0-9]+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertlatch
; CHECK-EMPTY:
; CHECK-NEXT: invertT:
; CHECK-NEXT:   %"pb'ipg_unwrap" = getelementptr inbounds double, ptr %"out'", i64 %n
; CHECK-NEXT:   %"po2'ipg_unwrap" = getelementptr inbounds double, ptr %"pb'ipg_unwrap", i64 %"iv'ac.0"
; CHECK-NEXT:   [[R10:%[0-9]+]] = load double, ptr %"po2'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %"po2'ipg_unwrap", align 8
; CHECK-NEXT:   [[R11:%[0-9]+]] = fadd fast double %"vv'de.1", [[R10]]
; CHECK-NEXT:   [[R12:%[0-9]+]] = extractvalue { ptr, ptr } %truetape, 1
; CHECK-NEXT:   [[R13:%[0-9]+]] = getelementptr inbounds double, ptr [[R12]], i64 %"iv'ac.0"
; CHECK-NEXT:   [[R14:%[0-9]+]] = load double, ptr [[R13]], align 8
; CHECK-NEXT:   [[R15:%[0-9]+]] = fmul fast double [[R11]], [[R14]]
; CHECK-NEXT:   [[R16:%[0-9]+]] = fadd fast double %"v'de.1", [[R15]]
; CHECK-NEXT:   [[R17:%[0-9]+]] = fmul fast double [[R11]], [[R14]]
; CHECK-NEXT:   [[R18:%[0-9]+]] = fadd fast double [[R16]], [[R17]]
; CHECK-NEXT:   %"pa'ipg_unwrap6" = getelementptr inbounds double, ptr %"a'", i64 %"iv'ac.0"
; CHECK-NEXT:   [[R19:%[0-9]+]] = load double, ptr %"pa'ipg_unwrap6", align 8
; CHECK-NEXT:   [[R20:%[0-9]+]] = fadd fast double [[R19]], [[R18]]
; CHECK-NEXT:   store double [[R20]], ptr %"pa'ipg_unwrap6", align 8
; CHECK-NEXT:   br label %invertloop
; CHECK-EMPTY:
; CHECK-NEXT: invertmerge_phirc:
; CHECK-NEXT:   [[R21:%[0-9]+]] = extractvalue { ptr, ptr } %truetape, 1
; CHECK-NEXT:   [[R22:%[0-9]+]] = getelementptr inbounds double, ptr [[R21]], i64 %"iv'ac.0"
; CHECK-NEXT:   [[R23:%[0-9]+]] = load double, ptr [[R22]], align 8
; CHECK-NEXT:   br label %invertmerge_phimerge
; CHECK-EMPTY:
; CHECK-NEXT: invertmerge_phimerge:
; CHECK-NEXT:   [[R24:%[0-9]+]] = phi fast double [ [[R23]], %invertmerge_phirc ], [ 1.000000e+00, %invertlatch ]
; CHECK-NEXT:   %cond_unwrap = fcmp ogt double [[R24]], 0.000000e+00
; CHECK-NEXT:   [[R25:%[0-9]+]] = getelementptr inbounds double, ptr %p_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   [[R26:%[0-9]+]] = load double, ptr [[R25]], align 8
; CHECK-NEXT:   [[R27:%[0-9]+]] = fmul fast double [[R35:%[0-9]+]], [[R26]]
; CHECK-NEXT:   [[R28:%[0-9]+]] = fmul fast double [[R35]], [[R26]]
; CHECK-NEXT:   [[R29:%[0-9]+]] = fadd fast double [[R27]], [[R28]]
; CHECK-NEXT:   [[R30:%[0-9]+]] = fadd fast double %"x2'de.1", [[R29]]
; CHECK-NEXT:   [[R31:%[0-9]+]] = select fast i1 %cond_unwrap, double %"x2'de.1", double [[R30]]
; CHECK-NEXT:   [[R32:%[0-9]+]] = select fast i1 %cond_unwrap, double [[R29]], double 0.000000e+00
; CHECK-NEXT:   [[R33:%[0-9]+]] = fmul fast double [[R31]], 3.000000e+00
; CHECK-NEXT:   [[R34:%[0-9]+]] = fadd fast double [[R32]], [[R33]]
; CHECK-NEXT:   %"x2'de.0" = select i1 %cond_unwrap, double [[R31]], double 0.000000e+00
; CHECK-NEXT:   %"x'de.0" = select i1 %cond_unwrap, double [[R32]], double [[R34]]
; CHECK-NEXT:   %par_unwrap = and i64 %"iv'ac.0", 1
; CHECK-NEXT:   %c0_unwrap = icmp eq i64 %par_unwrap, 0
; CHECK-NEXT:   br i1 %c0_unwrap, label %invertT, label %invertloop
; CHECK-EMPTY:
; CHECK-NEXT: invertlatch:
; CHECK-NEXT:   %"x2'de.1" = phi double [ %"x2'de.0", %incinvertloop ], [ 0.000000e+00, %J ]
; CHECK-NEXT:   %"v'de.1" = phi double [ %"v'de.0", %incinvertloop ], [ 0.000000e+00, %J ]
; CHECK-NEXT:   %"vv'de.1" = phi double [ %"vv'de.0", %incinvertloop ], [ 0.000000e+00, %J ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ [[R9]], %incinvertloop ], [ [[R2]], %J ]
; CHECK-NEXT:   %"po'ipg_unwrap" = getelementptr inbounds double, ptr %"out'", i64 %"iv'ac.0"
; CHECK-NEXT:   [[R35]] = load double, ptr %"po'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %"po'ipg_unwrap", align 8
; CHECK-NEXT:   %"pa'ipg_unwrap7" = getelementptr inbounds double, ptr %"a'", i64 %"iv'ac.0"
; CHECK-NEXT:   store double 0.000000e+00, ptr %"pa'ipg_unwrap7", align 8
; CHECK-NEXT:   %par_unwrap8 = and i64 %"iv'ac.0", 1
; CHECK-NEXT:   %c0_unwrap9 = icmp eq i64 %par_unwrap8, 0
; CHECK-NEXT:   br i1 %c0_unwrap9, label %invertmerge_phirc, label %invertmerge_phimerge
; CHECK-NEXT: }
