; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; In vector forward mode the custom forward derivative is called once per
; lane with that lane's shadows; the constant pointer gets a null shadow.
; Without constant arguments (@h) the rule was used directly with [2 x ptr]
; shadows, which is invalid IR.

@__enzyme_register_derivative_g = global [2 x ptr] [ptr @g, ptr @g_fwd]

declare double @g(ptr %x, ptr %n)

declare { double, double } @g_fwd(ptr %x, ptr %dx, ptr %n, ptr %dn)

define double @f(ptr %x, ptr %n) {
entry:
  %r = call double @g(ptr %x, ptr %n)
  ret double %r
}

define [2 x double] @caller(ptr %x, ptr %dx0, ptr %dx1, ptr %n) {
entry:
  %r = call [2 x double] (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_width", i64 2, ptr %x, ptr %dx0, ptr %dx1, metadata !"enzyme_const", ptr %n)
  ret [2 x double] %r
}

@__enzyme_register_derivative_h = global [2 x ptr] [ptr @h, ptr @h_fwd]

declare double @h(ptr %x)

declare { double, double } @h_fwd(ptr %x, ptr %dx)

define double @k(ptr %x) {
entry:
  %r = call double @h(ptr %x)
  ret double %r
}

define [2 x double] @caller2(ptr %x, ptr %dx0, ptr %dx1) {
entry:
  %r = call [2 x double] (...) @__enzyme_fwddiff(ptr @k, metadata !"enzyme_width", i64 2, ptr %x, ptr %dx0, ptr %dx1)
  ret [2 x double] %r
}

declare [2 x double] @__enzyme_fwddiff(...)

; CHECK: define internal [2 x double] @fwddiffe2f(ptr %x, [2 x ptr] %"x'", ptr %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast [2 x double] @fixderivative_g(ptr %x, [2 x ptr] %"x'", ptr %n)
; CHECK-NEXT:   ret [2 x double] %0
; CHECK-NEXT: }

; CHECK: define internal [2 x double] @fixderivative_g(ptr %x, [2 x ptr] %"x'", ptr %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x ptr] %"x'", 0
; CHECK-NEXT:   %1 = call { double, double } @g_fwd(ptr %x, ptr %0, ptr %n, ptr null)
; CHECK-NEXT:   %2 = extractvalue [2 x ptr] %"x'", 1
; CHECK-NEXT:   %3 = call { double, double } @g_fwd(ptr %x, ptr %2, ptr %n, ptr null)
; CHECK-NEXT:   %4 = extractvalue { double, double } %1, 1
; CHECK-NEXT:   %5 = insertvalue [2 x double] undef, double %4, 0
; CHECK-NEXT:   %6 = extractvalue { double, double } %3, 1
; CHECK-NEXT:   %7 = insertvalue [2 x double] %5, double %6, 1
; CHECK-NEXT:   ret [2 x double] %7
; CHECK-NEXT: }

; CHECK: define internal [2 x double] @fwddiffe2k(ptr %x, [2 x ptr] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast [2 x double] @fixderivative_h(ptr %x, [2 x ptr] %"x'")
; CHECK-NEXT:   ret [2 x double] %0
; CHECK-NEXT: }

; CHECK: define internal [2 x double] @fixderivative_h(ptr %x, [2 x ptr] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x ptr] %"x'", 0
; CHECK-NEXT:   %1 = call { double, double } @h_fwd(ptr %x, ptr %0)
; CHECK-NEXT:   %2 = extractvalue [2 x ptr] %"x'", 1
; CHECK-NEXT:   %3 = call { double, double } @h_fwd(ptr %x, ptr %2)
; CHECK-NEXT:   %4 = extractvalue { double, double } %1, 1
; CHECK-NEXT:   %5 = insertvalue [2 x double] undef, double %4, 0
; CHECK-NEXT:   %6 = extractvalue { double, double } %3, 1
; CHECK-NEXT:   %7 = insertvalue [2 x double] %5, double %6, 1
; CHECK-NEXT:   ret [2 x double] %7
; CHECK-NEXT: }
