; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify),verify" -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; EVAL: 3.000000 3.000000 3.000000

; llvm.threadlocal.address (emitted by Clang 16+ for thread_local globals) is a
; pointer identity on the global: its shadow is the address of the shadow
; global and it carries no derivative of its own.

; CHECK: @counter_shadow = thread_local global double 0.000000e+00
@counter = thread_local global double 3.000000e+00, align 8
@.fmt = private unnamed_addr constant [13 x i8] c"%f %f %f\0A\00\00\00\00", align 1

declare ptr @llvm.threadlocal.address.p0(ptr)
declare double @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)
declare { ptr, double } @__enzyme_augmentfwd(...)
declare double @__enzyme_reverse(...)
declare i32 @printf(ptr, ...)

define double @scale(double %x) {
entry:
  %tls = call ptr @llvm.threadlocal.address.p0(ptr @counter)
  %c = load double, ptr %tls, align 8
  %r = fmul double %x, %c
  store double %r, ptr %tls, align 8
  ret double %r
}

; CHECK-LABEL: define i32 @main(
define i32 @main() {
entry:
  ; CHECK: call { double } @diffescale(double 2.000000e+00, double 1.000000e+00)
  %rev = call double (...) @__enzyme_autodiff(ptr @scale, double 2.000000e+00)
  %tls0 = call ptr @llvm.threadlocal.address.p0(ptr @counter)
  store double 3.000000e+00, ptr %tls0, align 8
  ; CHECK: call fast double @fwddiffescale(double 2.000000e+00, double 1.000000e+00)
  %fwd = call double (...) @__enzyme_fwddiff(ptr @scale, double 2.000000e+00, double 1.000000e+00)
  store double 3.000000e+00, ptr %tls0, align 8
  ; CHECK: call { ptr, double } @augmented_scale(double 2.000000e+00)
  %aug = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @scale, double 2.000000e+00)
  %tape = extractvalue { ptr, double } %aug, 0
  ; CHECK: call { double } @diffescale.{{[0-9]+}}(double 2.000000e+00, double 1.000000e+00, ptr %{{.+}})
  %split = call double (...) @__enzyme_reverse(ptr @scale, double 2.000000e+00, double 1.000000e+00, ptr %tape)
  call i32 (ptr, ...) @printf(ptr @.fmt, double %rev, double %fwd, double %split)
  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffescale(double %x, double %differeturn)
; CHECK: %[[shadow:.+]] = alloca double
; CHECK: %tls = call ptr @llvm.threadlocal.address.p0(ptr @counter)
; CHECK: load double, ptr %[[shadow]]
; CHECK: store double {{.+}}, ptr %[[shadow]]

; CHECK-LABEL: define internal double @fwddiffescale(double %x, double %"x'")
; CHECK: %[[fshadow:.+]] = alloca double
; CHECK: %tls = call ptr @llvm.threadlocal.address.p0(ptr @counter)
; CHECK: %"c'ipl" = load double, ptr %[[fshadow]]

; CHECK-LABEL: define internal { double } @diffescale.{{[0-9]+}}(double %x, double %differeturn, ptr %tapeArg)
; CHECK: call ptr @llvm.threadlocal.address.p0(ptr @counter_shadow)
