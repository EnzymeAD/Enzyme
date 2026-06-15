! RUN: if [[ %llvmver -ge 13 && %fc == ifx ]]; then %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O0 -S -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 && %fc == ifx ]]; then %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O1 -S -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 && %fc == ifx ]]; then %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O2 -S -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 && %fc == ifx ]]; then %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O3 -S -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s; fi

! NOTE: This test is only configured to run with the ifx compiler
!       For it to work with the flang compiler we will need to address
!       https://github.com/EnzymeAD/Enzyme/issues/2820

module math
  interface
    subroutine norm__enzyme_autodiff(sr, x_desc, x, dx, y_desc, y, dy)
      interface
        subroutine sr_decal(a, b)
          real, dimension(:), intent(in) :: a
          real, dimension(:), intent(out) :: b
        end subroutine
      end interface
      procedure(sr_decal) :: sr
      integer, intent(in) :: x_desc
      real, dimension(:), intent(in) :: x
      real, dimension(:), intent(in) :: dx
      integer, intent(in) :: y_desc
      real, dimension(:), intent(inout) :: y
      real, dimension(:), intent(inout) :: dy
    end subroutine
  end interface
contains
  subroutine norm(x, y)
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(out) :: y
    y(:) = x / sum(x)
  end subroutine norm
end module math

program app
  use math, only: norm, norm__enzyme_autodiff
  use enzyme, only: enzyme_dup
  implicit none
  integer, parameter :: n = 1000000
  integer, parameter :: initial_value = 20
  real :: x(n), dx(n)
  real :: y(n), dy(n)

  x(:) = initial_value
  dy(:) = 1.0

  call norm(x, y)
  print *, y(n)

  dx(:) = 0.0
  call norm__enzyme_autodiff(norm, enzyme_dup, x, dx, enzyme_dup, y, dy)
  print *, dy(n)
end program app

! CHECK: 1.E-06
! CHECK-NEXT: 0.
