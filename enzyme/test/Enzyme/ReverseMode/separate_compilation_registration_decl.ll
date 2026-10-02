; RUN: if [ %llvmver -ge 16 ]; then printf "f 0,1\n" > %t.params; %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,enzyme" -enzyme-separate-compilation -enzyme-inactive-params=%t.params -S | FileCheck %s; fi

; Registrations written in C may declare the registered routine as a variable
; ("extern char f;"). In a module without the routine's definition, as under
; separate compilation, the declaration is made a function declaration so
; that the registration applies; and the plan's parameter indices for a
; declaration with a guessed signature are skipped where out of range.

@f = external global i8
@__enzyme_inactivefn_f = global ptr @f
@__enzyme_nofree_f = global ptr @f

define void @g() {
  ret void
}

; CHECK: declare void @f() #[[ATTR:[0-9]+]]
; CHECK: attributes #[[ATTR]] = {{.*}}"enzyme_inactive"
