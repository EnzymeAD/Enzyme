! RUN: ifx -flto -O0 -c  %s -o %t && %opt %t %loadEnzyme -enzyme -o %t1 && ifx -flto -O0 %t1 -o %t2 && %t2 | FileCheck %s
! RUN: ifx -flto -O1 -c  %s -o %t && %opt %t %loadEnzyme -enzyme -o %t1 && ifx -flto -O1 %t1 -o %t2 && %t2 | FileCheck %s
! RUN: ifx -flto -O2 -c  %s -o %t && %opt %t %loadEnzyme -enzyme -o %t1 && ifx -flto -O2 %t1 -o %t2 && %t2 | FileCheck %s
! RUN: ifx -flto -O3 -c  %s -o %t && %opt %t %loadEnzyme -enzyme -o %t1 && ifx -flto -O3 %t1 -o %t2 && %t2 | FileCheck %s

module math
    interface
        subroutine square__enzyme_autodiff(fn, x, dx)
        interface
            real function fn_decal(a)
                real, intent(in) :: a
            end function
        end interface
        procedure(fn_decal) :: fn
        real, intent(in) :: x
        real, intent(inout) :: dx
        end subroutine
    end interface
contains
    real function square( x )
        real, intent(in) :: x
        square = x**2
    end function
end module math

program app
    use math
    implicit none
    real :: x, dx

    x = 3
    print *, square(x)

    dx = 0
    call square__enzyme_autodiff(square, x, dx);

    print *, dx
end program app

! CHECK: 9
! CHECK-NEXT: 6