; RUN: if [ %llvmver -ge 16 ]; then echo "sq reverse+sz" > %t.exports; echo "cube" >> %t.exports; %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-export-derivatives=reverse -enzyme-export-list=%t.exports -S | FileCheck %s; fi

; Separate compilation: the export list names, per function, the variants to
; export (here only the strong-zero one of @sq); a function listed without
; variants gets those of -enzyme-export-derivatives, an unlisted one none.

define void @sq(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %x) {
  %v = load double, ptr %x
  %m = fmul double %v, %v
  store double %m, ptr %x
  ret void
}

define void @cube(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %x) {
  %v = load double, ptr %x
  %m = fmul double %v, %v
  %c = fmul double %m, %v
  store double %c, ptr %x
  ret void
}

define void @unlisted(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %x) {
  %v = load double, ptr %x
  %m = fadd double %v, %v
  store double %m, ptr %x
  ret void
}

; CHECK-DAG: @__enzyme_sep_rev_w1_sz_sq = constant { ptr, ptr }
; CHECK-DAG: @__enzyme_sep_rev_w1_cube = constant { ptr, ptr }
; CHECK-NOT: @__enzyme_sep_rev_w1_sq =
; CHECK-NOT: @__enzyme_sep_rev_w1_sz_cube =
; CHECK-NOT: unlisted =
