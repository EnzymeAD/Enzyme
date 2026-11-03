; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s; fi

; Like duplicatemallocloop.ll, but shaped like std::vector: the buffer is null
; when empty and is compared against null before being freed. A null check
; neither reads, writes, nor captures the memory, so it must not stop the
; allocation from being rematerialized in the reverse pass.

define dso_local double @f(ptr nocapture readonly %a0) local_unnamed_addr #0 {
entry:
  %a2 = load double, ptr %a0, align 8
  %m2 = fmul double %a2, %a2
  ret double %m2
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)

declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

define dso_local void @malloced(ptr noalias nocapture %a0, ptr noalias nocapture readonly %a1, i64 %n) #1 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %pre, label %alloc

alloc:
  %bytes = shl i64 %n, 3
  %m = call noalias ptr @malloc(i64 %bytes) #5
  br label %pre

pre:
  %a5 = phi ptr [ null, %entry ], [ %m, %alloc ]
  br label %loop

loop:
  %a9 = phi i32 [ 0, %pre ], [ %a14, %loop ]
  call void @llvm.lifetime.start.p0(i64 8, ptr %a5)
  %a10 = getelementptr inbounds double, ptr %a1, i32 %a9
  %a11 = load double, ptr %a10, align 8
  store double %a11, ptr %a5, align 8
  %a12 = call double @f(ptr %a5)
  %a13 = getelementptr inbounds double, ptr %a0, i32 %a9
  store double %a12, ptr %a13, align 8
  %a14 = add nuw nsw i32 %a9, 1
  %a15 = icmp eq i32 %a14, 10
  call void @llvm.lifetime.end.p0(i64 8, ptr %a5)
  br i1 %a15, label %exit, label %loop

exit:
  %isnull = icmp eq ptr %a5, null
  br i1 %isnull, label %done, label %dofree

dofree:
  call void @free(ptr %a5)
  br label %done

done:
  ret void
}

declare dso_local noalias ptr @malloc(i64) local_unnamed_addr #2

declare dso_local void @free(ptr nocapture) local_unnamed_addr #3

define dso_local void @derivative(ptr %a0, ptr %a1, ptr %a2, ptr %a3, i64 %a4) local_unnamed_addr #1 {
  call void (ptr, ...) @__enzyme_autodiff(ptr @malloced, ptr %a0, ptr %a1, ptr %a2, ptr %a3, i64 %a4) #6
  ret void
}

declare dso_local void @__enzyme_autodiff(ptr, ...) local_unnamed_addr #4

attributes #0 = { noinline norecurse nounwind readonly }
attributes #1 = { nounwind }
attributes #2 = { inaccessiblememonly nounwind }
attributes #3 = { inaccessiblemem_or_argmemonly nounwind }
attributes #6 = { nounwind }

; CHECK: define internal void @diffemalloced(ptr noalias nocapture %a0, ptr{{( nocapture)?}} %"a0'", ptr noalias nocapture readonly %a1, ptr{{( nocapture)?}} %"a1'", i64 %n)
; CHECK: loop:
; CHECK:   store double %a11, ptr %a5, align 8
; CHECK:   br i1 %a15, label %remat_enter, label %loop

; CHECK: remat_enter:
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %{{.+}}, %incinvertloop ], [ 9, %loop ]
; CHECK-NEXT:   %[[iv:.+]] = trunc i64 %"iv'ac.0" to i32
; CHECK-NEXT:   %a10_unwrap = getelementptr inbounds double, ptr %a1, i32 %[[iv]]
; CHECK-NEXT:   %a11_unwrap = load double, ptr %a10_unwrap, align 8
; CHECK-NEXT:   store double %a11_unwrap, ptr %a5, align 8
; CHECK:   call void @diffef(ptr %a5, ptr %{{.+}}, double %{{.+}}, double %{{.+}})
