; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-export-strong-zero -S | FileCheck %s; fi

; Separate compilation: a callee declared inactive (here without a body) is
; not differentiated through an external derivative table, and a function
; declared inactive is never exported. With -enzyme-export-strong-zero both
; the plain and the strong-zero variants of an exported derivative exist.

declare void @log_msg(ptr) nofree "enzyme_inactive" "enzyme_no_escaping_allocation"

define void @sq(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %x) "enzyme_export_derivative"="reverse" {
  call void @log_msg(ptr %x)
  %v = load double, ptr %x
  %m = fmul double %v, %v
  store double %m, ptr %x
  ret void
}

define void @quiet(ptr %x) "enzyme_inactive" "enzyme_export_derivative"="reverse" {
  call void @log_msg(ptr %x)
  ret void
}

; CHECK-DAG: @__enzyme_sep_rev_w1_sq = constant { ptr, ptr }
; CHECK-DAG: @__enzyme_sep_rev_w1_sz_sq = constant { ptr, ptr }
; CHECK-NOT: __enzyme_sep_{{.*}}log_msg
; CHECK-NOT: __enzyme_sep_{{.*}}quiet
