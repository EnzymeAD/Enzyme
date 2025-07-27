; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -opaque-pointers -S | FileCheck %s

declare ptr @__enzyme_virtualreverse(...)

declare ptr @malloc(i64)

define void @my_model.fullgrad1() {
  %z = call ptr (...) @__enzyme_virtualreverse(ptr nonnull @_take)
  ret void
}

define double @_take(ptr %a0, i1 %a1, i1 %a2) {
  br i1 %a1, label %merge, label %fval

fval:
  unreachable

merge:
  store double 3.1, ptr %a0
  br i1 %a2, label %tval, label %zval

tval:
  ret double 2.6

zval:
  store double 3.2, ptr %a0
  br i1 %a1, label %fval2, label %retb

fval2:
  unreachable
  
retb:
  ret double 2.7
}

; CHECK: define internal void @diffe_take(ptr %a0, ptr %"a0'", i1 %a1, i1 %a2, double %differeturn, ptr %tapeArg)
; CHECK-NEXT:   tail call void @free(ptr nonnull %tapeArg)
; CHECK-NEXT:   br i1 true, label %merge, label %fval

; CHECK: fval:                                             ; preds = %0
; CHECK-NEXT:   unreachable

; CHECK: merge:                                            ; preds = %0
; CHECK-NEXT:   br i1 %a2, label %tval, label %zval

; CHECK: tval:                                             ; preds = %merge
; CHECK-NEXT:   br label %inverttval

; CHECK: zval:                                             ; preds = %merge
; CHECK-NEXT:   br i1 false, label %fval2, label %retb

; CHECK: fval2:                                            ; preds = %zval
; CHECK-NEXT:   unreachable

; CHECK: retb:                                             ; preds = %zval
; CHECK-NEXT:   br label %invertretb

