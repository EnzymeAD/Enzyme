; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -opaque-pointers -S | FileCheck %s; fi

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
