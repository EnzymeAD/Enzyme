! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -enzyme-global-activity -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme -mllvm -enzyme-global-activity %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme -mllvm -enzyme-global-activity %s -o %t2 && %t2 | FileCheck %s %}

! The time, command line and environment queries of LLVM flang's runtime are
! inactive, also with -enzyme-global-activity.

program main
  use enzyme, only: enzyme_autodiff
  implicit none
  real(8) :: x, dx

  x = 3
  dx = 0
  call enzyme_autodiff(f, x, dx)
  write(*,"(f6.2)") dx

contains

  real(8) function f(x)
    real(8), intent(in) :: x
    real(8) :: t, s
    integer(8) :: count, rate, cmax
    integer :: n, len
    character(len=8) :: date
    character(len=64) :: arg

    call cpu_time(t)
    call date_and_time(date=date)
    call system_clock(count, rate, cmax)
    n = command_argument_count()
    call get_command(arg)
    call get_command_argument(0, arg)
    call get_environment_variable("PATH", length=len)

    ! s is 1 for any value of the queries.
    s = 1
    if (t < 0 .or. rate < 0 .or. n < 0 .or. len < 0) s = 2
    f = s * x * x
  end function f

end program main

! CHECK: 6.00
