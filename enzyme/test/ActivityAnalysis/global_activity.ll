; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=f -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f -S | FileCheck %s

%Pair = type { i32, i32 }
%Node = type { ptr }

@int_pair = global %Pair zeroinitializer, align 4
@sentinel = internal global %Node { ptr @sentinel }, align 8
@dead_global = internal global double 0.000000e+00, align 8
@const_float_global = internal global double 2.000000e+00, align 8

define double @f(double %x, i32 %n) {
entry:
  store i32 %n, ptr @int_pair, align 4
  %f1 = getelementptr inbounds %Pair, ptr @int_pair, i64 0, i32 1
  %n2 = mul i32 %n, 3
  store i32 %n2, ptr %f1, align 4
  %v0 = load i32, ptr @int_pair, align 4
  %v1 = load i32, ptr %f1, align 4
  %self = load ptr, ptr @sentinel, align 8
  %cmp = icmp eq ptr %self, @sentinel
  store double %x, ptr @dead_global, align 8
  store double 3.000000e+00, ptr @const_float_global, align 8
  %cf = load double, ptr @const_float_global, align 8
  %sel = select i1 %cmp, double %x, double %cf
  ret double %sel
}

; CHECK: double %x: icv:0
; CHECK-NEXT: i32 %n: icv:1
; CHECK-NEXT: entry
; CHECK-NEXT:   store i32 %n, ptr @int_pair, align 4: icv:1 ici:1
; CHECK-NEXT:   %f1 = getelementptr inbounds %Pair, ptr @int_pair, i64 0, i32 1: icv:1 ici:1
; CHECK-NEXT:   %n2 = mul i32 %n, 3: icv:1 ici:1
; CHECK-NEXT:   store i32 %n2, ptr %f1, align 4: icv:1 ici:1
; CHECK-NEXT:   %v0 = load i32, ptr @int_pair, align 4: icv:1 ici:1
; CHECK-NEXT:   %v1 = load i32, ptr %f1, align 4: icv:1 ici:1
; CHECK-NEXT:   %self = load ptr, ptr @sentinel, align 8: icv:1 ici:1
; CHECK-NEXT:   %cmp = icmp eq ptr %self, @sentinel: icv:1 ici:1
; CHECK-NEXT:   store double %x, ptr @dead_global, align 8: icv:1 ici:1
; CHECK-NEXT:   store double 3.000000e+00, ptr @const_float_global, align 8: icv:1 ici:1
; CHECK-NEXT:   %cf = load double, ptr @const_float_global, align 8: icv:1 ici:1
; CHECK-NEXT:   %sel = select i1 %cmp, double %x, double %cf: icv:0 ici:0
; CHECK-NEXT:   ret double %sel: icv:1 ici:1
