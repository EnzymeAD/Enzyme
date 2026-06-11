; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -mem2reg -sroa -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,sroa,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

declare noalias i8* @malloc(i64)
declare void @free(i8*)

define i8* @augmented_inner(double %x) {
entry:
  %ptr = call i8* @malloc(i64 8), !enzyme_tape_allocation !0
  ret i8* %ptr
}

define void @diffeinner(i8* %tape) {
entry:
  call void @free(i8* %tape), !enzyme_tape_free !0
  ret void
}

define double @outer(double %x) {
entry:
  %tape = call i8* @augmented_inner(double %x)
  call void @diffeinner(i8* %tape)
  ret double %x
}

define void @active(double %x) {
entry:
  call void (...) @__enzyme_autodiff(double (double)* @outer, double %x)
  ret void
}

declare void @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffeouter(double %x, double %differeturn)
; CHECK:   call void @diffediffeinner
; CHECK:   call { double } @diffeaugmented_inner

; CHECK: define internal void @diffediffeinner(ptr %tape, ptr %"tape'")
; CHECK-NOT: free
; CHECK: ret void

; CHECK: define internal { double } @diffeaugmented_inner(double %x, { ptr, ptr } %tapeArg)
; CHECK: call void @free
; CHECK: ret { double }

!0 = !{}
