; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f_attr -S -o /dev/null | FileCheck %s --check-prefix=ATTR; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f_attr_val -S -o /dev/null | FileCheck %s --check-prefix=ATTRVAL; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f_call_ret_attr -S -o /dev/null | FileCheck %s --check-prefix=RETATTR; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f_md -S -o /dev/null | FileCheck %s --check-prefix=MD; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f_md_val -S -o /dev/null | FileCheck %s --check-prefix=MDVAL; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f_md_inactive -S -o /dev/null | FileCheck %s --check-prefix=MDINACTIVE; fi

declare i64 @opaque_int(i64) "enzyme_active"
declare i64 @opaque_int_val(i64) "enzyme_active_val"
declare i64 @opaque_plain(i64)

define i64 @f_attr(i64 %x) {
entry:
  %r = call i64 @opaque_int(i64 %x)
  ret i64 %r
}

define i64 @f_attr_val(i64 %x) {
entry:
  %r = call i64 @opaque_int_val(i64 %x)
  ret i64 %r
}

define i64 @f_call_ret_attr(i64 %x) {
entry:
  %r = call "enzyme_active" i64 @opaque_plain(i64 %x)
  ret i64 %r
}

define i64 @f_md(i64 %x) {
entry:
  %r = add i64 %x, 1, !enzyme_active !0
  ret i64 %r
}

define i64 @f_md_val(i64 %x) {
entry:
  %r = add i64 %x, 1, !enzyme_active_val !0
  ret i64 %r
}

define double @f_md_inactive(double %x) {
entry:
  %r = fmul double %x, %x, !enzyme_inactive !0
  ret double %r
}

!0 = !{}

; ATTR: i64 %x: icv:1
; ATTR: entry
; ATTR-NEXT:   %r = call i64 @opaque_int(i64 %x): icv:0 ici:0

; ATTRVAL: i64 %x: icv:1
; ATTRVAL: entry
; ATTRVAL-NEXT:   %r = call i64 @opaque_int_val(i64 %x): icv:0 ici:

; RETATTR: i64 %x: icv:1
; RETATTR: entry
; RETATTR-NEXT:   %r = call "enzyme_active" i64 @opaque_plain(i64 %x): icv:0 ici:

; MD: i64 %x: icv:1
; MD: entry
; MD-NEXT:   %r = add i64 %x, 1, !enzyme_active !0: icv:0 ici:0

; MDVAL: i64 %x: icv:1
; MDVAL: entry
; MDVAL-NEXT:   %r = add i64 %x, 1, !enzyme_active_val !0: icv:0 ici:1

; MDINACTIVE: double %x: icv:0
; MDINACTIVE: entry
; MDINACTIVE-NEXT:   %r = fmul double %x, %x, !enzyme_inactive !0: icv:1 ici:1
