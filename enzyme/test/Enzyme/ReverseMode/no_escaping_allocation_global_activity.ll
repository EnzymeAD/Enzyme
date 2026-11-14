; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -preserve-nvvm -enzyme -enzyme-preopt=false -enzyme-global-activity -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm,enzyme" -enzyme-preopt=false -enzyme-global-activity -S | FileCheck %s

; With -enzyme-global-activity, a call to an inactive function is only
; replayed as is in the reverse pass if no allocation it makes can escape;
; otherwise Enzyme asks for an augmented forward pass, which a function
; without a body does not have ("No augmented forward pass found for check").
; Registering @check with __enzyme_no_escaping_allocation lets the inactive
; call stay a plain call.

declare void @check(double*)

@__enzyme_inactivefn = global i8* bitcast (void (double*)* @check to i8*)
@__enzyme_nofree = global i8* bitcast (void (double*)* @check to i8*)
@__enzyme_no_escaping_allocation = global i8* bitcast (void (double*)* @check to i8*)

define double @square(double* %x) {
entry:
  call void @check(double* %x)
  %v = load double, double* %x
  %m = fmul double %v, %v
  ret double %m
}

declare double @__enzyme_autodiff(...)

define double @dsquare(double* %x, double* %dx) {
entry:
  %r = call double (...) @__enzyme_autodiff(double (double*)* @square, metadata !"enzyme_dup", double* %x, double* %dx)
  ret double %r
}

; CHECK: define internal void @diffesquare(double* %x, double* %"x'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NOT: augmented
; CHECK: call void @check(double* %x)
; CHECK-NEXT: %v = load double, double* %x
; CHECK: invertentry:
; CHECK: [[D:%.+]] = fmul fast double %{{.+}}, %v
; CHECK: store double %{{.+}}, double* %"x'"
; CHECK-NEXT: ret void
