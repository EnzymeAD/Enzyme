; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=test_self_const -S -o /dev/null | FileCheck %s --check-prefix=SELFCONST; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=test_mut_const -S -o /dev/null | FileCheck %s --check-prefix=MUTCONST; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=test_alias_const -S -o /dev/null | FileCheck %s --check-prefix=ALIASCONST; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=test_cyc_active -S -o /dev/null | FileCheck %s --check-prefix=CYCACTIVE; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=test_cyc_agg_first -S -o /dev/null | FileCheck %s --check-prefix=CYCAGG; fi

@ext = external global double, align 8

@self_const = internal constant ptr @self_const, align 8

@mut_a = internal constant ptr @mut_b, align 8
@mut_b = internal constant ptr @mut_a, align 8

@alias_target = internal constant ptr @alias_self, align 8
@alias_self = internal alias ptr, ptr @alias_target

@cyc_act_a = internal constant { ptr, ptr } { ptr @cyc_act_b, ptr @ext }, align 8
@cyc_act_b = internal constant ptr @cyc_act_a, align 8

define double @test_self_const(double %x) {
entry:
  %p = load ptr, ptr @self_const, align 8
  %cmp = icmp eq ptr %p, @self_const
  %r = select i1 %cmp, double %x, double 0.0
  ret double %r
}

define double @test_mut_const(double %x) {
entry:
  %p = load ptr, ptr @mut_a, align 8
  %cmp = icmp eq ptr %p, @mut_b
  %r = select i1 %cmp, double %x, double 0.0
  ret double %r
}

define double @test_alias_const(double %x) {
entry:
  %p = load ptr, ptr @alias_target, align 8
  %cmp = icmp eq ptr %p, @alias_self
  %r = select i1 %cmp, double %x, double 0.0
  ret double %r
}

define double @test_cyc_active(double %x) {
entry:
  %p = load ptr, ptr @cyc_act_b, align 8
  %gep = getelementptr { ptr, ptr }, ptr %p, i64 0, i32 1
  %ep = load ptr, ptr %gep, align 8
  %v = load double, ptr %ep, align 8
  %r = fmul double %x, %v
  ret double %r
}

define double @test_cyc_agg_first(double %x) {
entry:
  %slot = alloca { ptr, ptr }, align 8
  store { ptr, ptr } { ptr @cyc_act_b, ptr @ext }, ptr %slot, align 8
  %p1 = load ptr, ptr %slot, align 8
  %p2 = load ptr, ptr %p1, align 8
  %gep = getelementptr { ptr, ptr }, ptr %p2, i64 0, i32 1
  %ep = load ptr, ptr %gep, align 8
  %v = load double, ptr %ep, align 8
  %r = fmul double %x, %v
  ret double %r
}

; SELFCONST: double %x: icv:0
; SELFCONST: entry
; SELFCONST-NEXT:   %p = load ptr, ptr @self_const, align 8: icv:1 ici:1
; SELFCONST-NEXT:   %cmp = icmp eq ptr %p, @self_const: icv:1 ici:1
; SELFCONST-NEXT:   %r = select i1 %cmp, double %x, double 0.000000e+00: icv:0 ici:0

; MUTCONST: double %x: icv:0
; MUTCONST: entry
; MUTCONST-NEXT:   %p = load ptr, ptr @mut_a, align 8: icv:1 ici:1
; MUTCONST-NEXT:   %cmp = icmp eq ptr %p, @mut_b: icv:1 ici:1
; MUTCONST-NEXT:   %r = select i1 %cmp, double %x, double 0.000000e+00: icv:0 ici:0

; ALIASCONST: double %x: icv:0
; ALIASCONST: entry
; ALIASCONST-NEXT:   %p = load ptr, ptr @alias_target, align 8: icv:1 ici:1
; ALIASCONST-NEXT:   %cmp = icmp eq ptr %p, @alias_self: icv:1 ici:1
; ALIASCONST-NEXT:   %r = select i1 %cmp, double %x, double 0.000000e+00: icv:0 ici:0

; CYCACTIVE: double %x: icv:0
; CYCACTIVE: entry
; CYCACTIVE-NEXT:   %p = load ptr, ptr @cyc_act_b, align 8: icv:0 ici:1
; CYCACTIVE-NEXT:   %gep = getelementptr { ptr, ptr }, ptr %p, i64 0, i32 1: icv:0 ici:1
; CYCACTIVE-NEXT:   %ep = load ptr, ptr %gep, align 8: icv:0 ici:1
; CYCACTIVE-NEXT:   %v = load double, ptr %ep, align 8: icv:0 ici:0
; CYCACTIVE-NEXT:   %r = fmul double %x, %v: icv:0 ici:0

; CYCAGG: double %x: icv:0
; CYCAGG: entry
; CYCAGG-NEXT:   %slot = alloca { ptr, ptr }, align 8: icv:0 ici:1
; CYCAGG-NEXT:   store { ptr, ptr } { ptr @cyc_act_b, ptr @ext }, ptr %slot, align 8: icv:1 ici:0
; CYCAGG-NEXT:   %p1 = load ptr, ptr %slot, align 8: icv:0 ici:1
; CYCAGG-NEXT:   %p2 = load ptr, ptr %p1, align 8: icv:0 ici:1
; CYCAGG-NEXT:   %gep = getelementptr { ptr, ptr }, ptr %p2, i64 0, i32 1: icv:0 ici:1
; CYCAGG-NEXT:   %ep = load ptr, ptr %gep, align 8: icv:0 ici:1
; CYCAGG-NEXT:   %v = load double, ptr %ep, align 8: icv:0 ici:0
; CYCAGG-NEXT:   %r = fmul double %x, %v: icv:0 ici:0
