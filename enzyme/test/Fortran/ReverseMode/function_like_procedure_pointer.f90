! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes="preserve-nvvm" -S | FileCheck %s --check-prefix=IR --implicit-check-not=__enzyme_function_like__
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes="preserve-nvvm,enzyme,preserve-nvvm-end" -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

module function_like_procedure_pointer_test
  implicit none

  procedure(log1p_like_function), pointer, private :: &
    fn__enzyme_function_like__log1p => log1p_like_function

contains

  function log1p_like_function(x) result(y)
    double precision, value :: x
    double precision :: y

    y = 2.0d0 * x
  end function log1p_like_function

  function test(x) result(y)
    double precision, intent(in) :: x
    double precision :: y

    y = log1p_like_function(x)
  end function test

end module function_like_procedure_pointer_test

program main
  use enzyme, only: enzyme_autodiff
  use function_like_procedure_pointer_test, only: test
  implicit none

  double precision :: x, dx

  x = 2.0d0
  dx = 0.0d0
  call enzyme_autodiff(test, x, dx)

  if (abs(dx - 1.0d0 / 3.0d0) > 1.0d-10) error stop
  print *, "procedure pointer function_like passed"
end program main

! IR: "enzyme_math"="log1p"
! CHECK: procedure pointer function_like passed
