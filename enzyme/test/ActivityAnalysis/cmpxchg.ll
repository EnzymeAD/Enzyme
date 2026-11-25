; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=int_field_cas -o /dev/null | FileCheck %s --check-prefix=INTCAS; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=int_field_cas -S -o /dev/null | FileCheck %s --check-prefix=INTCAS
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=global_inactive_cas -o /dev/null | FileCheck %s --check-prefix=INACTIVECAS; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=global_inactive_cas -S -o /dev/null | FileCheck %s --check-prefix=INACTIVECAS
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=float_bitcast_cas -o /dev/null | FileCheck %s --check-prefix=FLOATCAS; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=float_bitcast_cas -S -o /dev/null | FileCheck %s --check-prefix=FLOATCAS

@counter = global i64 0, align 8, !enzyme_inactive !0

define void @int_field_cas(ptr %state) {
entry:
  %val = load double, ptr %state, align 8
  %sq = fmul double %val, %val
  store double %sq, ptr %state, align 8
  %count_ptr = getelementptr inbounds { double, i64 }, ptr %state, i64 0, i32 1
  %init = load atomic i64, ptr %count_ptr monotonic, align 8
  br label %cas_loop

cas_loop:
  %old = phi i64 [ %init, %entry ], [ %cur, %cas_loop ]
  %inc = add i64 %old, 1
  %next = urem i64 %inc, 1024
  %cx = cmpxchg ptr %count_ptr, i64 %old, i64 %next seq_cst monotonic, align 8
  %cur = extractvalue { i64, i1 } %cx, 0
  %ok = extractvalue { i64, i1 } %cx, 1
  br i1 %ok, label %exit, label %cas_loop

exit:
  ret void
}

define double @global_inactive_cas(double %x) {
entry:
  %init = load atomic i64, ptr @counter monotonic, align 8
  br label %cas_loop

cas_loop:
  %old = phi i64 [ %init, %entry ], [ %cur, %cas_loop ]
  %next = add i64 %old, 1
  %cx = cmpxchg ptr @counter, i64 %old, i64 %next seq_cst monotonic, align 8
  %cur = extractvalue { i64, i1 } %cx, 0
  %ok = extractvalue { i64, i1 } %cx, 1
  br i1 %ok, label %exit, label %cas_loop

exit:
  %sq = fmul double %x, %x
  ret double %sq
}

define void @float_bitcast_cas(ptr %state, double %delta) {
entry:
  %init_bits = load atomic i64, ptr %state monotonic, align 8
  br label %cas_loop

cas_loop:
  %old_bits = phi i64 [ %init_bits, %entry ], [ %cur_bits, %cas_loop ]
  %old_val = bitcast i64 %old_bits to double
  %next_val = fadd double %old_val, %delta
  %next_bits = bitcast double %next_val to i64
  %cx = cmpxchg ptr %state, i64 %old_bits, i64 %next_bits acq_rel acquire, align 8
  %cur_bits = extractvalue { i64, i1 } %cx, 0
  %ok = extractvalue { i64, i1 } %cx, 1
  br i1 %ok, label %exit, label %cas_loop

exit:
  ret void
}

!0 = !{}

; INTCAS: ptr %state: icv:0
; INTCAS: entry
; INTCAS-NEXT:   %val = load double, ptr %state, align 8: icv:0 ici:0
; INTCAS-NEXT:   %sq = fmul double %val, %val: icv:0 ici:0
; INTCAS-NEXT:   store double %sq, ptr %state, align 8: icv:1 ici:0
; INTCAS-NEXT:   %count_ptr = getelementptr inbounds { double, i64 }, ptr %state, i64 0, i32 1: icv:0 ici:1
; INTCAS-NEXT:   %init = load atomic i64, ptr %count_ptr monotonic, align 8: icv:1 ici:0
; INTCAS-NEXT:   br label %cas_loop: icv:1 ici:1
; INTCAS-NEXT: cas_loop
; INTCAS-NEXT:   %old = phi i64 [ %init, %entry ], [ %cur, %cas_loop ]: icv:1 ici:1
; INTCAS-NEXT:   %inc = add i64 %old, 1: icv:1 ici:1
; INTCAS-NEXT:   %next = urem i64 %inc, 1024: icv:1 ici:1
; INTCAS-NEXT:   %cx = cmpxchg ptr %count_ptr, i64 %old, i64 %next seq_cst monotonic, align 8: icv:1 ici:1
; INTCAS-NEXT:   %cur = extractvalue { i64, i1 } %cx, 0: icv:1 ici:1
; INTCAS-NEXT:   %ok = extractvalue { i64, i1 } %cx, 1: icv:1 ici:1

; INACTIVECAS: double %x: icv:0
; INACTIVECAS: entry
; INACTIVECAS-NEXT:   %init = load atomic i64, ptr @counter monotonic, align 8: icv:1 ici:1
; INACTIVECAS-NEXT:   br label %cas_loop: icv:1 ici:1
; INACTIVECAS-NEXT: cas_loop
; INACTIVECAS-NEXT:   %old = phi i64 [ %init, %entry ], [ %cur, %cas_loop ]: icv:1 ici:1
; INACTIVECAS-NEXT:   %next = add i64 %old, 1: icv:1 ici:1
; INACTIVECAS-NEXT:   %cx = cmpxchg ptr @counter, i64 %old, i64 %next seq_cst monotonic, align 8: icv:1 ici:1
; INACTIVECAS-NEXT:   %cur = extractvalue { i64, i1 } %cx, 0: icv:1 ici:1
; INACTIVECAS-NEXT:   %ok = extractvalue { i64, i1 } %cx, 1: icv:1 ici:1

; FLOATCAS: ptr %state: icv:0
; FLOATCAS-NEXT: double %delta: icv:0
; FLOATCAS-NEXT: entry
; FLOATCAS-NEXT:   %init_bits = load atomic i64, ptr %state monotonic, align 8: icv:0 ici:0
; FLOATCAS-NEXT:   br label %cas_loop: icv:1 ici:1
; FLOATCAS-NEXT: cas_loop
; FLOATCAS-NEXT:   %old_bits = phi i64 [ %init_bits, %entry ], [ %cur_bits, %cas_loop ]: icv:0 ici:0
; FLOATCAS-NEXT:   %old_val = bitcast i64 %old_bits to double: icv:0 ici:0
; FLOATCAS-NEXT:   %next_val = fadd double %old_val, %delta: icv:0 ici:0
; FLOATCAS-NEXT:   %next_bits = bitcast double %next_val to i64: icv:0 ici:0
; FLOATCAS-NEXT:   %cx = cmpxchg ptr %state, i64 %old_bits, i64 %next_bits acq_rel acquire, align 8: icv:0 ici:0
; FLOATCAS-NEXT:   %cur_bits = extractvalue { i64, i1 } %cx, 0: icv:0 ici:0
; FLOATCAS-NEXT:   %ok = extractvalue { i64, i1 } %cx, 1: icv:1 ici:1
