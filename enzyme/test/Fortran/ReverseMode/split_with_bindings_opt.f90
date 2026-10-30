! REQUIRES: fortran
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

program main
  use, intrinsic :: iso_c_binding, only: c_ptr
  use enzyme, only: enzyme_dup, enzyme_tape, enzyme_augmentfwd, enzyme_reverse
  implicit none
  real :: x, dx, y, dy
  real :: x2, dx2, y2, dy2
  type(c_ptr) :: tape, tape2

  ! The forward pass stores a pointer to its tape in `tape`; the reverse pass
  ! reads the cached x from it and frees it.
  x = 3
  dx = 0
  y = 0
  dy = 1
  call enzyme_augmentfwd(square, enzyme_tape, tape, enzyme_dup, x, dx, &
                         enzyme_dup, y, dy)
  print *, y
  x = 100
  call enzyme_reverse(square, enzyme_tape, tape, enzyme_dup, x, dx, &
                      enzyme_dup, y, dy)
  print *, dx

  ! Two forward passes in flight, reversed in the opposite order
  x = 2
  dx = 0
  dy = 1
  x2 = 5
  dx2 = 0
  dy2 = 1
  call enzyme_augmentfwd(square, enzyme_tape, tape, x, dx, y, dy)
  call enzyme_augmentfwd(square, enzyme_tape, tape2, x2, dx2, y2, dy2)
  call enzyme_reverse(square, enzyme_tape, tape2, x2, dx2, y2, dy2)
  call enzyme_reverse(square, enzyme_tape, tape, x, dx, y, dy)
  print *, dx
  print *, dx2

  ! A function with an active return
  x = 4
  dx = 0
  call enzyme_augmentfwd(cube, enzyme_tape, tape, x, dx)
  call enzyme_reverse(cube, enzyme_tape, tape, x, dx)
  print *, dx

contains

  subroutine square(x, y)
    real, intent(in) :: x
    real, intent(out) :: y
    y = x**2
  end subroutine square

  real function cube(x)
    real, intent(in) :: x
    cube = x**3
  end function cube

end program main

! CHECK: 9
! CHECK-NEXT: 6
! CHECK-NEXT: 4
! CHECK-NEXT: 10
! CHECK-NEXT: 48
