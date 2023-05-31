! RUN: if [ %llvmver -ge 13 ]; then ifx -flto -O0 -c  %s -o /dev/stdout | %opt %loadEnzyme -enzyme -o %t && ifx -flto -O0 %t -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then ifx -flto -O1 -c  %s -o /dev/stdout | %opt %loadEnzyme -enzyme -o %t && ifx -flto -O1 %t -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then ifx -flto -O2 -c  %s -o /dev/stdout | %opt %loadEnzyme -enzyme -o %t && ifx -flto -O2 %t -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then ifx -flto -O3 -c  %s -o /dev/stdout | %opt %loadEnzyme -enzyme -o %t && ifx -flto -O3 %t -o %t1 && %t1 | FileCheck %s; fi

program app
    implicit none

    interface
        subroutine fort__enzyme_autodiff(fnc, x, dx, y, dy)
            interface
                subroutine fnc_decal(a, b)
                    real, intent(in) :: a
                    real, allocatable, intent(out) :: b
                end subroutine
            end interface
            procedure(fnc_decal) :: fnc
            real, intent(in) :: x
            real, intent(inout) :: dx
            real, allocatable, intent(out) :: y
            real, allocatable, intent(inout) :: dy
        end subroutine
    end interface

    real, allocatable :: result

    call square(3.0, result)
    print *, result
    print *, grad_square(3.0)

    contains

    subroutine square(x, y)
        real, intent(in) :: x
        real, allocatable, intent(out) :: y
        allocate(y)
        y = x * x
    end subroutine

    function grad_square(x) result(dx)
        real, intent(in) :: x
        real :: dx
        real, allocatable :: y, dy
        allocate(dy)
        dy = 1
        dx = 0
        call fort__enzyme_autodiff(square, x, dx, y, dy);
    end function grad_square
end program app

! CHECK: 9
! CHECK-NEXT: 6
