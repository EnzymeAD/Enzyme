! REQUIRES: fortran
! REQUIRES: ifx
! RUN: %fc -flto -O0 -c %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O1 -c %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s

! NOTE: This test is only configured to run with the ifx compiler
!       For it to work with the flang compiler we will need to address
!       https://github.com/EnzymeAD/Enzyme/issues/2820

module selectFirstForward
    implicit none
    interface
        subroutine selectFirst__enzyme_fwddiff(fnc, x, dx, y, dy)
           interface
               subroutine fnc_decal(a, z)
                   real, allocatable, intent(in) :: a(:)
                   real, intent(inout) :: z
               end subroutine
           end interface
           procedure(fnc_decal) :: fnc
           real, allocatable, intent(in) :: x(:)
           real, allocatable, intent(inout) :: dx(:)
           real, intent(inout) :: y
           real, intent(inout) :: dy
        end subroutine
    end interface

    contains

    subroutine selectFirst(x, y)
        real, allocatable, intent(in) :: x(:)
        real, intent(inout) :: y
        y = x(1)
    end subroutine
end module

program app
    use selectFirstForward, only: selectFirst, selectFirst__enzyme_fwddiff
    implicit none
    real, allocatable :: x(:), dx(:)
    real :: y, dy

    allocate(x(3))
    allocate(dx(3))

    x = [2,3,4]
    dx = [1,0,0]
    y = 0
    dy = 0

    call selectFirst__enzyme_fwddiff(selectFirst, x, dx, y, dy)

    print *, int(y)
    print *, int(dx(1))
    print *, int(dx(2))
    print *, int(dx(3))
    print *, int(dy)
end program


! CHECK: 2
! CHECK-NEXT: 1
! CHECK-NEXT: 0
! CHECK-NEXT: 0
! CHECK-NEXT: 1
