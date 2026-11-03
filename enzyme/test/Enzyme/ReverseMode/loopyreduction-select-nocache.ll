; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; A running max is differentiated through the index of the last iteration that
; selected a new element; the reverse pass never reads the start value, so the
; start value, which cannot be recomputed here (x[0] is overwritten after the
; loop), must not be put on the tape.

define void @maxred(ptr noalias %x, i64 %n, ptr noalias %out) {
entry:
  %x0 = load double, ptr %x, align 8, !tbaa !0
  br label %loop

loop:
  %i = phi i64 [ 1, %entry ], [ %i.next, %loop ]
  %m = phi double [ %x0, %entry ], [ %m.next, %loop ]
  %p = getelementptr inbounds double, ptr %x, i64 %i
  %xi = load double, ptr %p, align 8, !tbaa !0
  %c = fcmp olt double %m, %xi
  %m.next = select i1 %c, double %xi, double %m
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  store double %m.next, ptr %out, align 8, !tbaa !0
  store double 0.000000e+00, ptr %x, align 8, !tbaa !0
  ret void
}

define ptr @aug(ptr %x, ptr %dx, i64 %n, ptr %out, ptr %dout) {
  %t = call ptr (...) @__enzyme_augmentfwd(ptr @maxred, ptr %x, ptr %dx, i64 %n, ptr %out, ptr %dout)
  ret ptr %t
}

define void @rev(ptr %x, ptr %dx, i64 %n, ptr %out, ptr %dout, ptr %tape) {
  call void (...) @__enzyme_reverse(ptr @maxred, ptr %x, ptr %dx, i64 %n, ptr %out, ptr %dout, ptr %tape)
  ret void
}

declare ptr @__enzyme_augmentfwd(...)
declare void @__enzyme_reverse(...)

!0 = !{!1, !1, i64 0}
!1 = !{!"double", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}

; The tape holds only the per-iteration conditions; the start value %x0 is not
; stored (without the fix the tape was { double, ptr } with %x0 in it).

; CHECK: define internal ptr @augmented_maxred(ptr noalias {{.*}}%x, ptr {{.*}}%"x'", i64 %n, ptr noalias {{.*}}%out, ptr {{.*}}%"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[tape:.+]] = {{(tail )?}}call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) ptr @malloc(i64 8)
; CHECK-NEXT:   %x0 = load double, ptr %x, align 8{{.*}}
; CHECK-NEXT:   %[[nm2:.+]] = add i64 %n, -2
; CHECK-NEXT:   %[[cnt:.+]] = add nsw i64 %[[nm2]], 1
; CHECK-NEXT:   %[[ccache:.+]] = {{(tail )?}}call noalias nonnull ptr @malloc(i64 %[[cnt]]), !enzyme_cache_alloc
; CHECK-NEXT:   store ptr %[[ccache]], ptr %[[tape]], align 8
; CHECK-NEXT:   br label %loop

; CHECK: loop:
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %m = phi double [ %x0, %entry ], [ %m.next, %loop ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %p = getelementptr inbounds {{(nuw )?}}double, ptr %x, i64 %iv.next
; CHECK-NEXT:   %xi = load double, ptr %p, align 8{{.*}}
; CHECK-NEXT:   %c = fcmp olt double %m, %xi
; CHECK-NEXT:   %[[cptr:.+]] = getelementptr inbounds {{(nuw )?}}i1, ptr %[[ccache]], i64 %iv
; CHECK-NEXT:   store i1 %c, ptr %[[cptr]], align 1, !invariant.group
; CHECK-NEXT:   %m.next = select i1 %c, double %xi, double %m
; CHECK-NEXT:   %i.next = add nuw i64 %iv.next, 1
; CHECK-NEXT:   %done = icmp eq i64 %i.next, %n
; CHECK-NEXT:   br i1 %done, label %exit, label %loop

; CHECK: exit:
; CHECK-NEXT:   store double %m.next, ptr %out, align 8{{.*}}
; CHECK-NEXT:   store double 0.000000e+00, ptr %x, align 8{{.*}}
; CHECK-NEXT:   ret ptr %[[tape]]
; CHECK-NEXT: }

; The reverse pass rebuilds the last selecting iteration from the cached
; conditions; the start value's adjoint is taken when that index is 0, and its
; primal is never read.

; CHECK: define internal void @diffemaxred(ptr noalias {{.*}}%x, ptr {{.*}}%"x'", i64 %n, ptr noalias {{.*}}%out, ptr {{.*}}%"out'", ptr %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %truetape = load ptr, ptr %tapeArg, align 8, !enzyme_mustcache
; CHECK-NEXT:   {{(tail )?}}call void @free(ptr nonnull %tapeArg)
; CHECK-NEXT:   %[[rnm2:.+]] = add i64 %n, -2
; CHECK-NEXT:   br label %loop

; CHECK: loop:
; CHECK-NEXT:   %[[idx:.+]] = phi i64 [ 0, %entry ], [ %[[idxn:.+]], %loop ]
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %[[rcptr:.+]] = getelementptr inbounds {{(nuw )?}}i1, ptr %truetape, i64 %iv
; CHECK-NEXT:   %c = load i1, ptr %[[rcptr]], align 1, !invariant.group
; CHECK-NEXT:   %[[idxn]] = select i1 %c, i64 %iv.next, i64 %[[idx]]
; CHECK-NEXT:   %i.next = add nuw i64 %iv.next, 1
; CHECK-NEXT:   %done = icmp eq i64 %i.next, %n
; CHECK-NEXT:   br i1 %done, label %invertexit, label %loop

; CHECK: invertentry:
; CHECK-NEXT:   %[[dx0old:.+]] = load double, ptr %"x'", align 8{{.*}}
; CHECK-NEXT:   %[[dx0new:.+]] = fadd fast double %[[dx0old]], %[[x0de:.+]]
; CHECK-NEXT:   store double %[[dx0new]], ptr %"x'", align 8{{.*}}
; CHECK-NEXT:   {{(tail )?}}call void @free(ptr nonnull %truetape), !enzyme_cache_free
; CHECK-NEXT:   ret void

; CHECK: invertloop:
; CHECK-NEXT:   %"m.next'de.0" = phi double [ %[[dout:.+]], %invertexit ], [ %[[mnde:.+]], %incinvertloop ]
; CHECK-NEXT:   %"x0'de.0" = phi double [ 0.000000e+00, %invertexit ], [ %[[x0de]], %incinvertloop ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[rnm2]], %invertexit ], [ %[[ivdec:.+]], %incinvertloop ]
; CHECK-NEXT:   %iv.next_unwrap = add nuw nsw i64 %"iv'ac.0", 1
; CHECK-NEXT:   %[[issel:.+]] = icmp eq i64 %[[idxn]], %iv.next_unwrap
; CHECK-NEXT:   %[[dxi:.+]] = select fast i1 %[[issel]], double %"m.next'de.0", double 0.000000e+00
; CHECK-NEXT:   %"p'ipg_unwrap" = getelementptr inbounds {{(nuw )?}}double, ptr %"x'", i64 %iv.next_unwrap
; CHECK-NEXT:   %[[dpold:.+]] = load double, ptr %"p'ipg_unwrap", align 8{{.*}}
; CHECK-NEXT:   %[[dpnew:.+]] = fadd fast double %[[dpold]], %[[dxi]]
; CHECK-NEXT:   store double %[[dpnew]], ptr %"p'ipg_unwrap", align 8{{.*}}
; CHECK-NEXT:   %[[first:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[mnde]] = select fast i1 %[[first]], double 0.000000e+00, double %"m.next'de.0"
; CHECK-NEXT:   %[[start:.+]] = icmp eq i64 %[[idxn]], 0
; CHECK-NEXT:   %[[dstart:.+]] = select fast i1 %[[start]], double %"m.next'de.0", double 0.000000e+00
; CHECK-NEXT:   %[[x0acc:.+]] = fadd fast double %"x0'de.0", %[[dstart]]
; CHECK-NEXT:   %[[x0de]] = select fast i1 %[[first]], double %[[x0acc]], double %"x0'de.0"
; CHECK-NEXT:   br i1 %[[first]], label %invertentry, label %incinvertloop

; CHECK: incinvertloop:
; CHECK-NEXT:   %[[ivdec]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertloop

; CHECK: invertexit:
; CHECK-NEXT:   store double 0.000000e+00, ptr %"x'", align 8{{.*}}
; CHECK-NEXT:   %[[dout]] = load double, ptr %"out'", align 8{{.*}}
; CHECK-NEXT:   store double 0.000000e+00, ptr %"out'", align 8{{.*}}
; CHECK-NEXT:   br label %invertloop
; CHECK-NEXT: }
