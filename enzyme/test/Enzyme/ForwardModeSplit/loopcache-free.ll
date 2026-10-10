; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; The augmented pass caches x[i] in a loop cache and returns the pointer to
; it as the tape. Split forward mode has no reverse pass, so the tangent pass
; frees the cache before it returns.

define void @sumsq(i64 %n, ptr %x, ptr %s) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %acc.next, %loop ]
  %gep = getelementptr inbounds double, ptr %x, i64 %i
  %xi = load double, ptr %gep, align 8
  %sq = fmul double %xi, %xi
  %acc.next = fadd double %acc, %sq
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  store double %acc.next, ptr %s, align 8
  ret void
}

define void @caller(i64 %n, ptr %x, ptr %dx, ptr %s, ptr %ds) {
entry:
  %tape = alloca ptr, align 8
  call void (...) @__enzyme_augmentfwd(ptr @sumsq, metadata !"enzyme_allocated", i64 8, metadata !"enzyme_tape", ptr %tape, metadata !"enzyme_const", i64 %n, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %s, ptr %ds)
  call void (...) @__enzyme_fwdsplit(ptr @sumsq, metadata !"enzyme_allocated", i64 8, metadata !"enzyme_tape", ptr %tape, metadata !"enzyme_const", i64 %n, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %s, ptr %ds)
  ret void
}

declare void @__enzyme_augmentfwd(...)
declare void @__enzyme_fwdsplit(...)

; CHECK: define internal ptr @augmented_sumsq(i64 %n, ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%s, ptr {{.*}}%"s'")
; CHECK:        %[[cache:.+]] = {{(tail )?}}call noalias nonnull ptr @malloc(i64 %{{.+}}), !enzyme_cache_alloc
; CHECK:        ret ptr %[[cache]]

; CHECK: define internal void @fwddiffesumsq(i64 %n, ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%s, ptr {{.*}}%"s'", ptr %tapeArg)
; CHECK:      loop:
; CHECK:        %[[gep:.+]] = getelementptr inbounds double, ptr %tapeArg, i64 %iv
; CHECK-NEXT:   %xi = load double, ptr %[[gep]]
; CHECK:      exit:
; CHECK-NEXT:   store double %{{.+}}, ptr %"s'"
; CHECK-NEXT:   {{(tail )?}}call void @free(ptr nonnull %tapeArg), !enzyme_cache_free
; CHECK-NEXT:   ret void
