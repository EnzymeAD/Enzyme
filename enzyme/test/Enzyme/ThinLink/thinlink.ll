; RUN: if [ %llvmver -ge 16 ]; then rm -rf %t && mkdir -p %t && %opt < %s %OPnewLoadEnzyme -passes="enzyme-summary" -enzyme-summary-out=%t/main.json -disable-output && %opt < %S/Inputs/thinlink_g.ll.in %OPnewLoadEnzyme -passes="enzyme-summary" -enzyme-summary-out=%t/g.json -disable-output && python3 %S/../../../scripts/enzyme_thinlink.py --inactive both --out %t/plan %t/main.json %t/g.json | FileCheck %s --check-prefix=REPORT; fi
; RUN: if [ %llvmver -ge 16 ]; then FileCheck %s --check-prefix=EXPORTS < %t/plan/g.exports; fi
; RUN: if [ %llvmver -ge 16 ]; then FileCheck %s --check-prefix=PARAMS < %t/plan/inactive_params.txt; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %S/Inputs/thinlink_g.ll.in %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-export-list=%t/plan/g.exports -enzyme-inactive-params=%t/plan/inactive_params.txt -S | FileCheck %s --check-prefix=DEF; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-inactive-params=%t/plan/inactive_params.txt -S | FileCheck %s --check-prefix=USE; fi

; Thin-link planning for separate compilation: summaries of two modules are
; combined into the export list of @g's module (the one variant @f's call
; graph needs) and the parameters that carry no floating-point data; both
; modules then agree on the name of @g's derivative.

@enzyme_strong_zero = external global i32

declare void @g(ptr, ptr)
declare void @__enzyme_autodiff(...)

define void @f(ptr %x, ptr %n) {
  call void @g(ptr %x, ptr %n)
  ret void
}

define void @caller(ptr %x, ptr %dx, ptr %n) {
  call void (...) @__enzyme_autodiff(ptr @f, ptr @enzyme_strong_zero, ptr %x, ptr %dx, ptr %n, ptr %n)
  ret void
}

; REPORT: exports: 1 functions, 1 variants, from 1 modules
; REPORT: variants used: ['reverse+sz']
; REPORT: inactive parameters of exported functions: 1 of 2
; REPORT: inferred 2

; EXPORTS: g reverse+sz

; PARAMS: g 1

; DEF: @__enzyme_sep_rev_w1_sz_c2_g = constant { ptr, ptr }

; USE: @__enzyme_sep_rev_w1_sz_c2_g = external constant { ptr, ptr }
