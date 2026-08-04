! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! NOTE: This test is only configured to run with the flang compiler.
!       For it to work with the ifx compiler we will need to figure out how to
!       handle the different signature for __enzyme_batch used by ifx.

module squareBatch
  interface
    subroutine square__enzyme_batch(sr, width_desc, width, &
                                    vec_desc, x1, x2, x3, x4, y1, y2, y3, y4)
      interface
        subroutine sr_decal(xx, yy)
          real, intent(in)  :: xx
          real, intent(out) :: yy
        end subroutine sr_decal
      end interface
      procedure(sr_decal)        :: sr
      integer, value, intent(in) :: width_desc
      integer, value, intent(in) :: width
      integer, value, intent(in) :: vec_desc
      real, intent(in)           :: x1, x2, x3, x4
      real, intent(out)          :: y1, y2, y3, y4
    end subroutine square__enzyme_batch
  end interface
contains
  ! NOTE: __enzyme_batch works more straightforwardly with subroutines than with
  !      Fortran functions
  subroutine square(x, y)
    real, intent(in)  :: x
    real, intent(out) :: y
    y = x ** 2
  end subroutine
end module

program main
  use enzyme, only: enzyme_vector, enzyme_width
  use squareBatch, only: square, square__enzyme_batch
  implicit none
  real :: x1, x2, x3, x4
  real :: y1, y2, y3, y4

  x1 = 23.1
  x2 = 10.0
  x3 = 100.0
  x4 = 3.14

  call square__enzyme_batch(square, enzyme_width, 4, &
                            enzyme_vector, x1, x2, x3, x4, &
                            y1, y2, y3, y4)

  write(*,"(f0.4)") y1
  write(*,"(f0.4)") y2
  write(*,"(f0.4)") y3
  write(*,"(f0.4)") y4
end program

! CHECK: 533.6100
! CHECK-NEXT: 100.0000
! CHECK-NEXT: 10000.0000
! CHECK-NEXT: 9.8596
