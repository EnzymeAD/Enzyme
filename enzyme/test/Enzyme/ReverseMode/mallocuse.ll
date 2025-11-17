; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -opaque-pointers -S | FileCheck %s

declare ptr @__enzyme_virtualreverse(...)

declare ptr @malloc(i64)

define void @my_model.fullgrad1() {
  %z = call ptr (...) @__enzyme_virtualreverse(ptr nonnull @_take)
  ret void
}

define double @_take(ptr %a0, i1 %a1) {
  %a3 = tail call ptr @malloc(i64 10)
  %a4 = tail call ptr @malloc(i64 10)
  %a5 = ptrtoint ptr %a4 to i64
  %a6 = or i64 %a5, 1
  %a7 = inttoptr i64 %a6 to ptr
  %a8 = load double, ptr %a7, align 8
  store double %a8, ptr %a0, align 8
  br i1 %a1, label %.lr.ph, label %.lr.ph1.peel.next

.lr.ph1.peel.next:                                ; preds = %2
  %.pre = load double, ptr %a4, align 8
  ret double %.pre

.lr.ph:                                           ; preds = %.lr.ph, %2
  %a9 = load double, ptr %a3, align 4
  store double %a9, ptr %a4, align 8
  br label %.lr.ph
}

; CHECK: define internal { ptr, double } @augmented__take(ptr nocapture writeonly %a0, ptr nocapture %"a0'", i1 %a1)
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) ptr @malloc(i64 8)
; CHECK-NEXT:   %a3 = tail call ptr @malloc(i64 10)
; CHECK-NEXT:   %a4 = tail call ptr @malloc(i64 10)
; CHECK-NEXT:   store ptr %a4, ptr %malloccall, align 8
; CHECK-NEXT:   %a5 = ptrtoint ptr %a4 to i64
; CHECK-NEXT:   %a6 = or i64 %a5, 1
; CHECK-NEXT:   %a7 = inttoptr i64 %a6 to ptr
; CHECK-NEXT:   %a8 = load double, ptr %a7, align 8
; CHECK-NEXT:   store double %a8, ptr %a0, align 8
; CHECK-NEXT:   br i1 %a1, label %.lr.ph, label %.lr.ph1.peel.next

; CHECK: .lr.ph1.peel.next:                                ; preds = %0
; CHECK-NEXT:   %.pre = load double, ptr %a4, align 8, !alias.scope !10, !noalias !13
; CHECK-NEXT:   %.fca.0.insert = insertvalue { ptr, double } poison, ptr %malloccall, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { ptr, double } %.fca.0.insert, double %.pre, 1
; CHECK-NEXT:   ret { ptr, double } %.fca.1.insert

; CHECK: .lr.ph:                                           ; preds = %0, %.lr.ph
; CHECK-NEXT:   %a9 = load double, ptr %a3, align 4
; CHECK-NEXT:   store double %a9, ptr %a4, align 8
; CHECK-NEXT:   br label %.lr.ph
; CHECK-NEXT: }

; CHECK: define internal void @diffe_take(ptr %a0, ptr %"a0'", i1 %a1, double %differeturn, ptr %tapeArg)
; CHECK-NEXT:   tail call void @free(ptr nonnull %tapeArg)
; CHECK-NEXT:   br i1 %a1, label %.lr.ph, label %invert.lr.ph1.peel.next

; CHECK: .lr.ph:                                           ; preds = %0, %.lr.ph
; CHECK-NEXT:   br label %.lr.ph

; CHECK: invert.lr.ph1.peel.next:                          ; preds = %0
; CHECK-NEXT:   store double 0.000000e+00, ptr %"a0'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
