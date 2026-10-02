; RUN: if [ %llvmver -ge 16 ]; then printf "sq reverse,reverse+o1\n" > %t.vars; %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-export-list=%t.vars -S | FileCheck %s --check-prefix=DEF; fi
; RUN: if [ %llvmver -ge 16 ]; then printf "ext reverse,reverse+o1\n" > %t.imports; %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-import-variants=%t.imports -S | FileCheck %s --check-prefix=USE; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -S | FileCheck %s --check-prefix=DEFAULT; fi

; Separate compilation: a derivative may be exported in variants that assume
; some arguments are not overwritten after the call (_o<hex mask>), so they
; need not be cached. A caller picks, of the variants listed by
; -enzyme-import-variants, the one its own analysis of the call allows: here
; @user's local %t is not written after the call to @ext, %x is (by the
; store), so the variant keeping argument 0 (%t) is used. Without the list
; every argument is assumed overwritten.

define void @sq(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %x) {
  %v = load double, ptr %x
  %m = fmul double %v, %v
  store double %m, ptr %x
  ret void
}

declare void @ext(ptr, ptr)

define void @user(ptr %x) {
  %t = alloca double
  %v = load double, ptr %x
  store double %v, ptr %t
  call void @ext(ptr %t, ptr %x)
  store double 0.0, ptr %x
  %w = load double, ptr %t
  store double %w, ptr %x
  ret void
}

declare void @__enzyme_autodiff(...)

define void @caller(ptr %x, ptr %dx) {
  call void (...) @__enzyme_autodiff(ptr @user, ptr %x, ptr %dx)
  ret void
}

; DEF-DAG: @__enzyme_sep_rev_w1_sq = constant { ptr, ptr }
; DEF-DAG: @__enzyme_sep_rev_w1_o1_sq = constant { ptr, ptr }

; USE: @__enzyme_sep_rev_w1_o1_ext = external constant { ptr, ptr }
; USE-NOT: @__enzyme_sep_rev_w1_ext =

; DEFAULT: @__enzyme_sep_rev_w1_ext = external constant { ptr, ptr }
