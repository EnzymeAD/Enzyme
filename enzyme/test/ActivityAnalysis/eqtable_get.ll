; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f -S | FileCheck %s
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=g -o /dev/null | FileCheck %s --check-prefix=ACTIVE; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=g -S | FileCheck %s --check-prefix=ACTIVE

; jl_eqtable_get is a load from the table: a buffer fetched from an inactive
; table (with an inactive default) is inactive, even if active data is later
; stored into it.
define double @f(double %x) {
entry:
  %table = call i8* @get_table(), !enzyme_inactive !0
  %key = call i8* @get_key(), !enzyme_inactive !0
  %dflt = call i8* @get_dflt(), !enzyme_inactive !0
  %buf = call i8* @ijl_eqtable_get(i8* %table, i8* %key, i8* %dflt)
  %bufd = bitcast i8* %buf to double*
  store double %x, double* %bufd, align 8
  %r = load double, double* %bufd, align 8
  %sq = fmul double %r, %r
  ret double %sq
}

; With an active table, the buffer remains active.
define double @g(double %x, i8* %table) {
entry:
  %key = call i8* @get_key(), !enzyme_inactive !0
  %dflt = call i8* @get_dflt(), !enzyme_inactive !0
  %buf = call i8* @ijl_eqtable_get(i8* %table, i8* %key, i8* %dflt)
  %bufd = bitcast i8* %buf to double*
  store double %x, double* %bufd, align 8
  %r = load double, double* %bufd, align 8
  %sq = fmul double %r, %r
  ret double %sq
}

declare i8* @get_table()
declare i8* @get_key()
declare i8* @get_dflt()
declare i8* @ijl_eqtable_get(i8*, i8*, i8*)

!0 = !{}

; CHECK: double %x: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %table = call {{i8\*|ptr}} @get_table(), !enzyme_inactive !0: icv:1 ici:1
; CHECK-NEXT:   %key = call {{i8\*|ptr}} @get_key(), !enzyme_inactive !0: icv:1 ici:1
; CHECK-NEXT:   %dflt = call {{i8\*|ptr}} @get_dflt(), !enzyme_inactive !0: icv:1 ici:1
; CHECK-NEXT:   %buf = call {{i8\*|ptr}} @ijl_eqtable_get({{i8\*|ptr}} %table, {{i8\*|ptr}} %key, {{i8\*|ptr}} %dflt): icv:1 ici:1

; ACTIVE: double %x: icv:0
; ACTIVE-NEXT: {{i8\*|ptr}} %table: icv:0
; ACTIVE-NEXT: entry
; ACTIVE-NEXT:   %key = call {{i8\*|ptr}} @get_key(), !enzyme_inactive !0: icv:1 ici:1
; ACTIVE-NEXT:   %dflt = call {{i8\*|ptr}} @get_dflt(), !enzyme_inactive !0: icv:1 ici:1
; ACTIVE-NEXT:   %buf = call {{i8\*|ptr}} @ijl_eqtable_get({{i8\*|ptr}} %table, {{i8\*|ptr}} %key, {{i8\*|ptr}} %dflt): icv:0 ici:0
