; RUN: if [ %llvmver -ge 16 ]; then printf "grid\n" > %t.inv; %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-invariant-globals=%t.inv -S | FileCheck %s; fi

; -enzyme-invariant-globals: globals a whole-program plan found nothing
; writes during differentiation are read again in the reverse pass instead
; of being cached; loads from them are marked enzyme_nocache.

@grid = global [4 x double] zeroinitializer
@state = global [4 x double] zeroinitializer

define void @f(ptr %x) {
  %g = load double, ptr @grid
  %s = load double, ptr @state
  %v = load double, ptr %x
  %m = fmul double %v, %g
  %n = fmul double %m, %s
  store double %n, ptr %x
  ret void
}

declare void @__enzyme_autodiff(...)

define void @caller(ptr %x, ptr %dx) {
  call void (...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx)
  ret void
}

; CHECK: define void @f(ptr {{.*}}%x)
; CHECK-NEXT:   %g = load double, ptr @grid, align 8, !enzyme_nocache
; CHECK-NEXT:   %s = load double, ptr @state, align 8{{$}}
