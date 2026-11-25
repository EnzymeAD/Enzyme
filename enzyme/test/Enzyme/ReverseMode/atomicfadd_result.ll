; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EXEC; fi

; The old value returned by atomicrmw fadd/fsub is active here. Its adjoint
; flows into the shadow of the memory location (the old value is what the
; location held before the update), after the shadow has been read for the
; adjoint of the added value.

declare double @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)
@fmt = private constant [7 x i8] c"%f %f\0A\00"

define double @kernel(ptr %buf, double %x) {
entry:
  %sq = fmul double %x, %x
  %old = atomicrmw fadd ptr %buf, double %sq monotonic, align 8
  %res = fmul double %old, %x
  ret double %res
}

define double @dkernel(ptr %buf, ptr %dbuf, double %x) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @kernel, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_out", double %x)
  ret double %dx
}

define i32 @main() {
  %buf = alloca double, align 8
  %dbuf = alloca double, align 8
  store double 5.0, ptr %buf
  store double 0.0, ptr %dbuf
  %dx = call double @dkernel(ptr %buf, ptr %dbuf, double 3.0)
  %db = load double, ptr %dbuf
  call i32 (ptr, ...) @printf(ptr @fmt, double %dx, double %db)
  ret i32 0
}

; d/dx (old * x) = old = 5, and d(buf) = x = 3 (old was the previous value of buf).
; EXEC: 5.000000 3.000000

; CHECK-LABEL: define internal { double } @diffekernel(
; CHECK: %old = atomicrmw fadd ptr %buf, double %sq monotonic, align 8
; CHECK: fmul fast double %{{.+}}, %old
; CHECK: %[[DOLD:.+]] = load double, ptr %"old'de"
; CHECK-NEXT: store double 0.000000e+00, ptr %"old'de"
; CHECK-NEXT: %[[DSQ:.+]] = load atomic double, ptr %"buf'" monotonic, align 8
; CHECK-NEXT: %[[OLDSQ:.+]] = load double, ptr %"sq'de"
; CHECK-NEXT: %[[NEWSQ:.+]] = fadd fast double %[[OLDSQ]], %[[DSQ]]
; CHECK-NEXT: store double %[[NEWSQ]], ptr %"sq'de"
; CHECK-NEXT: atomicrmw fadd ptr %"buf'", double %[[DOLD]] monotonic, align 8
