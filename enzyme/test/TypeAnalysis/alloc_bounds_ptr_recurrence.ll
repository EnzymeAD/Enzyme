; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -disable-output | FileCheck %s

%Vec = type { ptr, ptr, ptr }

declare noalias ptr @calloc(i64, i64)

define double @callee(ptr %out_vec, double %val, i1 %pick_first) {
entry:
  %buf = call noalias ptr @calloc(i64 2, i64 24)
  %end_cap = getelementptr %Vec, ptr %buf, i64 2
  %finish_slot = getelementptr %Vec, ptr %out_vec, i64 0, i32 1
  %cap_slot = getelementptr %Vec, ptr %out_vec, i64 0, i32 2
  store ptr %buf, ptr %out_vec, align 8
  store ptr %end_cap, ptr %cap_slot, align 8
  %guard_slot = alloca ptr, align 8
  store ptr %buf, ptr %guard_slot, align 8
  br label %loop

loop:
  %cur = load ptr, ptr %guard_slot, align 8
  store double %val, ptr %cur, align 8
  %next = getelementptr %Vec, ptr %cur, i64 1
  store ptr %next, ptr %guard_slot, align 8
  %done = icmp eq ptr %next, %end_cap
  br i1 %done, label %exit, label %loop

exit:
  store ptr %next, ptr %finish_slot, align 8
  %first_ptr = load ptr, ptr %out_vec, align 8
  %second_ptr = getelementptr %Vec, ptr %first_ptr, i64 1
  %second_val = load double, ptr %second_ptr, align 8
  %end_ptr = load ptr, ptr %cap_slot, align 8
  %chosen = select i1 %pick_first, ptr %first_ptr, ptr %end_ptr
  %chosen_val = load double, ptr %chosen, align 8
  %sum = fadd double %second_val, %chosen_val
  ret double %sum
}

; CHECK-LABEL: callee - {[-1]:Float@double}
; CHECK:   %buf = call noalias ptr @calloc(i64 2, i64 24): {[-1]:Pointer, [-1,0]:Float@double, [-1,24]:Float@double}
; CHECK:   %guard_slot = alloca ptr, align 8: {[-1]:Pointer, [-1,-1]:Pointer, [-1,-1,0]:Float@double, [-1,-1,24]:Float@double}
