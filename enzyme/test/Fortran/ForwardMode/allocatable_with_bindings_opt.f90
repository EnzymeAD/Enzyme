! REQUIRES: fortran
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_fwddiff
  implicit none
  integer :: n
  real, allocatable :: x(:), dx(:)
  real :: y, dy

  n = 3
  allocate(x(n))
  allocate(dx(n))

  x = [2,3,4]
  dx = [1,0,0]
  y = 0
  dy = 0

  call enzyme_fwddiff(selectFirst, enzyme_const, n, &
                      enzyme_dup, x, dx, enzyme_dup, y, dy)

  print *, int(y)
  print *, int(dx(1))
  print *, int(dx(2))
  print *, int(dx(3))
  print *, int(dy)

contains

    ! TODO: Switch to assumed shape implementation once
    !       https://github.com/EnzymeAD/Enzyme/issues/2820
    !       has been addressed
    subroutine selectFirst(n, x, y)
        integer, intent(in) :: n
        real, intent(in) :: x(n)
        real, intent(inout) :: y
        y = x(1)
    end subroutine selectFirst

end program main

! CHECK: 2
! CHECK-NEXT: 1
! CHECK-NEXT: 0
! CHECK-NEXT: 0
! CHECK-NEXT: 1
