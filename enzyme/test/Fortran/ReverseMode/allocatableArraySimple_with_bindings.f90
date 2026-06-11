! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O0 -S -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O1 -S -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O2 -S -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O3 -S -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s; fi

! NOTE: This test is only configured to run with the flang compiler
!       For it to work with the ifx compiler we will need to figure out how to
!       handle the indirection involved in the enzyme_autodiff binding

module AD
    implicit none

    contains

    ! TODO: Switch to assumed shape implementation once
    !       https://github.com/EnzymeAD/Enzyme/issues/2820
    !       has been addressed
    subroutine selectFirst(n, x, y)
        integer, intent(in) :: n
        real, intent(in) :: x(n)
        real, intent(inout) :: y
        y = x(1)
    end subroutine
end module

program app
    use AD, only: selectFirst
    use enzyme, only: enzyme_const, enzyme_dup, f__enzyme_autodiff
    implicit none
    integer :: n
    real, allocatable :: x(:), dx(:)
    real :: y, dy

    n = 3
    allocate(x(n))
    allocate(dx(n))

    x = [2,3,4]
    dx = [0,0,0]
    y = 0
    dy = 1

    call f__enzyme_autodiff(selectFirst, enzyme_const, n, &
                            enzyme_dup, x, dx, enzyme_dup, y, dy)

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
! CHECK-NEXT: 0
