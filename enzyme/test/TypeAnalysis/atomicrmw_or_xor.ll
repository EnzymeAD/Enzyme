; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define double @caller(ptr %flags, ptr %sign_slot, <2 x double> %vec_in, <1 x float> %vec1_in) {
entry:
  %old_flag = atomicrmw or ptr %flags, i32 1 seq_cst, align 4
  %flag_ok = icmp ne i32 %old_flag, -1
  %old_bits = atomicrmw xor ptr %sign_slot, i64 -9223372036854775808 seq_cst, align 8
  %old_val = bitcast i64 %old_bits to double
  %vec_bits = bitcast <2 x double> %vec_in to <2 x i64>
  %vec_or = or <2 x i64> %vec_bits, <i64 0, i64 undef>
  %vec_out = bitcast <2 x i64> %vec_or to <2 x double>
  %elem0 = extractelement <2 x double> %vec_out, i64 0
  %vec1_bits = bitcast <1 x float> %vec1_in to <1 x i32>
  %vec1_or = or <1 x i32> <i32 0>, %vec1_bits
  %vec1_out = bitcast <1 x i32> %vec1_or to <1 x float>
  %elem1 = extractelement <1 x float> %vec1_out, i64 0
  %elem1_ext = fpext float %elem1 to double
  %sum0 = fadd double %old_val, %elem0
  %sum1 = fadd double %sum0, %elem1_ext
  %res = select i1 %flag_ok, double %sum1, double 0.000000e+00
  ret double %res
}

; CHECK: ptr %flags: {[-1]:Pointer}
; CHECK-NEXT: ptr %sign_slot: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT: <2 x double> %vec_in: {[-1]:Float@double}
; CHECK-NEXT: <1 x float> %vec1_in: {[-1]:Float@float}
; CHECK-NEXT: entry
; CHECK-NEXT:   %old_flag = atomicrmw or ptr %flags, i32 1 seq_cst, align 4: {}
; CHECK-NEXT:   %flag_ok = icmp ne i32 %old_flag, -1: {[-1]:Integer}
; CHECK-NEXT:   %old_bits = atomicrmw xor ptr %sign_slot, i64 -9223372036854775808 seq_cst, align 8: {[-1]:Float@double}
; CHECK-NEXT:   %old_val = bitcast i64 %old_bits to double: {[-1]:Float@double}
; CHECK-NEXT:   %vec_bits = bitcast <2 x double> %vec_in to <2 x i64>: {[-1]:Float@double}
; CHECK-NEXT:   %vec_or = or <2 x i64> %vec_bits, <i64 0, i64 undef>: {[0]:Float@double}
; CHECK-NEXT:   %vec_out = bitcast <2 x i64> %vec_or to <2 x double>: {[0]:Float@double}
; CHECK:   %vec1_or = or <1 x i32> zeroinitializer, %vec1_bits: {[-1]:Float@float}
