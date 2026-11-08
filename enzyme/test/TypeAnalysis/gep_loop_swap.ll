; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=chained_gep_loop -o /dev/null | FileCheck %s --check-prefix=CHAINED; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=chained_gep_loop -S -o /dev/null | FileCheck %s --check-prefix=CHAINED
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=unrolled_epilogue_swap -o /dev/null | FileCheck %s --check-prefix=EPILOGUE; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=unrolled_epilogue_swap -S -o /dev/null | FileCheck %s --check-prefix=EPILOGUE

define void @chained_gep_loop(i64 %n, ptr %d) {
entry:
  %cmp = icmp eq i64 %n, 0
  br i1 %cmp, label %exit, label %loop

loop:
  %i = phi i64 [ %next, %loop ], [ 0, %entry ]
  %idx = shl i64 %i, 2
  %base = getelementptr inbounds double, ptr %d, i64 %idx
  %a = getelementptr inbounds double, ptr %base, i64 2
  %b = getelementptr inbounds double, ptr %base, i64 4
  %va = load i64, ptr %a, align 8
  %vb = load i64, ptr %b, align 8
  store i64 %vb, ptr %a, align 8
  store i64 %va, ptr %b, align 8
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  %d2 = getelementptr inbounds double, ptr %d, i64 2
  %v2 = load double, ptr %d2, align 8
  %v0 = load double, ptr %d, align 8
  %sum = fadd double %v2, %v0
  store double %sum, ptr %d, align 8
  ret void
}

define void @unrolled_epilogue_swap(i64 %n, ptr %d) {
entry:
  %d4 = getelementptr double, ptr %d, i64 4
  %is_zero = icmp eq i64 %n, 0
  br i1 %is_zero, label %exit, label %check

check:
  %odd = and i64 %n, 1
  %is_one = icmp eq i64 %n, 1
  br i1 %is_one, label %epilogue_Check, label %preheader

preheader:
  %even_n = and i64 %n, -2
  br label %unrolled_loop

unrolled_loop:
  %iv = phi i64 [ 0, %preheader ], [ %iv_next, %unrolled_loop ]
  %off0 = shl i64 %iv, 2
  %idx_a0 = or i64 %off0, 2
  %ptr_a0 = getelementptr inbounds double, ptr %d, i64 %idx_a0
  %val_a0 = load i64, ptr %ptr_a0, align 8
  %ptr_b0 = getelementptr double, ptr %d4, i64 %off0
  %val_b0 = load i64, ptr %ptr_b0, align 8
  store i64 %val_b0, ptr %ptr_a0, align 8
  store i64 %val_a0, ptr %ptr_b0, align 8
  %iv_next = add i64 %iv, 2
  %loop_done = icmp eq i64 %iv_next, %even_n
  br i1 %loop_done, label %loop_exit, label %unrolled_loop

loop_exit:
  %unrolled_end = shl i64 %iv_next, 2
  br label %epilogue_Check

epilogue_Check:
  %epilogue_off = phi i64 [ 0, %check ], [ %unrolled_end, %loop_exit ]
  %skip_epilogue = icmp eq i64 %odd, 0
  br i1 %skip_epilogue, label %exit, label %epilogue

epilogue:
  %epi_idx_a = or i64 %epilogue_off, 2
  %epi_ptr_a = getelementptr inbounds double, ptr %d, i64 %epi_idx_a
  %epi_val_a = load i64, ptr %epi_ptr_a, align 8
  %epi_ptr_b = getelementptr double, ptr %d4, i64 %epilogue_off
  %epi_val_b = load i64, ptr %epi_ptr_b, align 8
  store i64 %epi_val_b, ptr %epi_ptr_a, align 8
  store i64 %epi_val_a, ptr %epi_ptr_b, align 8
  br label %exit

exit:
  %d2 = getelementptr inbounds double, ptr %d, i64 2
  %v2 = load double, ptr %d2, align 8
  %v0 = load double, ptr %d, align 8
  %sum = fadd double %v2, %v0
  store double %sum, ptr %d, align 8
  ret void
}

; CHAINED: chained_gep_loop
; CHAINED:   %va = load i64, ptr %a, align 8: {[-1]:Float@double}
; CHAINED-NEXT:   %vb = load i64, ptr %b, align 8: {[-1]:Float@double}

; EPILOGUE: unrolled_epilogue_swap
; EPILOGUE:   %epi_idx_a = or i64 %epilogue_off, 2: {[-1]:Integer}
; EPILOGUE-NEXT:   %epi_ptr_a = getelementptr inbounds double, ptr %d, i64 %epi_idx_a: {[-1]:Pointer, [-1,0]:Float@double, [-1,16]:Float@double}
; EPILOGUE-NEXT:   %epi_val_a = load i64, ptr %epi_ptr_a, align 8: {[-1]:Float@double}
; EPILOGUE-NEXT:   %epi_ptr_b = getelementptr double, ptr %d4, i64 %epilogue_off: {[-1]:Pointer, [-1,0]:Float@double, [-1,48]:Float@double}
; EPILOGUE-NEXT:   %epi_val_b = load i64, ptr %epi_ptr_b, align 8: {[-1]:Float@double}
