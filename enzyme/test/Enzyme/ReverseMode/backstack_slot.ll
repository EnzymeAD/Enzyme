; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-julia-addr-load -S | FileCheck %s; fi

; A stack slot handed out as a Julia object (`enzyme_backstack`) is moved back
; to the default address space after differentiation. Its pointer is stored
; into a local slot and loaded back (as a cache for the reverse pass does), so
; the slot is moved to the default address space as well.

declare double @__enzyme_autodiff(...)

define double @f(double %x) {
entry:
  %a = alloca double, align 8
  %p = addrspacecast ptr %a to ptr addrspace(10), !enzyme_backstack !0
  %d = addrspacecast ptr addrspace(10) %p to ptr addrspace(11)
  store double %x, ptr addrspace(11) %d, align 8
  %slot = alloca ptr addrspace(10), align 8
  store volatile ptr addrspace(10) %p, ptr %slot, align 8
  %q = load volatile ptr addrspace(10), ptr %slot, align 8
  %qd = addrspacecast ptr addrspace(10) %q to ptr addrspace(11)
  %v = load double, ptr addrspace(11) %qd, align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @df(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x)
  ret double %r
}

!0 = !{}

; CHECK: define internal { double } @diffef(double %x, double %differeturn)
; CHECK:        %a = alloca double, align 8
; CHECK-NOT:    addrspacecast ptr %a to ptr addrspace(10)
; CHECK:        %slot = alloca ptr, align 8
; CHECK:        store volatile ptr %a, ptr %slot, align 8
; CHECK:        %q = load volatile ptr, ptr %slot, align 8
; CHECK:        load double, ptr %q
