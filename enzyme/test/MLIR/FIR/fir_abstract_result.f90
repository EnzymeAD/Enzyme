! Results that flang's abstract-result conversion turns into an argument
! (here an allocatable CHARACTER and an array) get no "enzyme_type" result
! type: the annotation pass runs before that conversion, and the LLVM
! function returns void.
!
! REQUIRES: flang_enzyme_mlir
! RUN: mkdir -p %t.mod
! RUN: %flang_enzyme -module-dir %t.mod -O0 -emit-llvm %s -o - | FileCheck %s

module fir_abstract_result
contains
  ! CHECK-LABEL: define void @_QMfir_abstract_resultPname(
  function name() result(s)
    character(len=:), allocatable :: s
    s = "abc"
  end function

  ! CHECK-LABEL: define void @_QMfir_abstract_resultPtwice(
  function twice(x) result(y)
    real(8), intent(in) :: x(3)
    real(8) :: y(3)
    y = 2 * x
  end function

  ! A scalar result stays a result and keeps its type.
  ! CHECK-LABEL: define "enzyme_type"="{[-1]:Float@double}" double @_QMfir_abstract_resultPsquare(
  function square(x) result(y)
    real(8), intent(in) :: x
    real(8) :: y
    y = x * x
  end function
end module
