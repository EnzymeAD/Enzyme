; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: 40.000000

@.fmt = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1
@_ZZ12inline_scaleEdE5state = linkonce_odr global double 0.000000e+00, align 8
@_ZGVZ12inline_scaleEdE5state = linkonce_odr global i64 0, align 8
@_ZN6HolderIdE5valueE = weak_odr global double 0.000000e+00, align 8

; CHECK: @_ZZ12inline_scaleEdE5state_shadow = linkonce_odr global double 0.000000e+00, align 8
; CHECK-NOT: @_ZGVZ12inline_scaleEdE5state_shadow
; CHECK: @_ZN6HolderIdE5valueE_shadow = weak_odr global double 0.000000e+00, align 8

declare i32 @printf(ptr, ...)
declare double @__enzyme_autodiff(...)

define linkonce_odr void @write_odr_globals(double %x) noinline {
entry:
  %guard_byte = load atomic i8, ptr @_ZGVZ12inline_scaleEdE5state acquire, align 8
  %uninit = icmp eq i8 %guard_byte, 0
  br i1 %uninit, label %init, label %done

init:
  store atomic i8 1, ptr @_ZGVZ12inline_scaleEdE5state release, align 8
  br label %done

done:
  %sq = fmul double %x, %x
  store double %sq, ptr @_ZZ12inline_scaleEdE5state, align 8
  %three_x = fmul double 3.000000e+00, %x
  store double %three_x, ptr @_ZN6HolderIdE5valueE, align 8
  ret void
}

define linkonce_odr double @read_odr_globals(double %x) noinline {
entry:
  %s = load double, ptr @_ZZ12inline_scaleEdE5state, align 8
  %two_s = fmul double 2.000000e+00, %s
  %h = load double, ptr @_ZN6HolderIdE5valueE, align 8
  %h_x = fmul double %h, %x
  %sum = fadd double %two_s, %h_x
  ret double %sum
}

define double @compute(double %x) {
entry:
  call void @write_odr_globals(double %x)
  %res = call double @read_odr_globals(double %x)
  ret double %res
}

define i32 @main() {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @compute, double 4.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.fmt, double %dx)
  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffecompute(double %x, double %differeturn)
; CHECK: %[[DR:.+]] = call { double } @differead_odr_globals(double %x, double %{{.+}})
; CHECK: %[[DW:.+]] = call { double } @diffewrite_odr_globals(double %x, i1 %{{.+}})
