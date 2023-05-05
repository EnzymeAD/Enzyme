! RUN: if [ %llvmver -ge 13 ]; then ifx -flto -O0 -c  %s -o /dev/stdout | %opt %loadEnzyme -enzyme -o %t && ifx -flto -O0 %t -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then ifx -flto -O1 -c  %s -o /dev/stdout | %opt %loadEnzyme -enzyme -o %t && ifx -flto -O1 %t -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then ifx -flto -O2 -c  %s -o /dev/stdout | %opt %loadEnzyme -enzyme -o %t && ifx -flto -O2 %t -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then ifx -flto -O3 -c  %s -o /dev/stdout | %opt %loadEnzyme -enzyme -o %t && ifx -flto -O3 %t -o %t1 && %t1 | FileCheck %s; fi

program app
    implicit none
    interface
        subroutine square__enzyme_autodiff(fnc, x, dx)
            interface
                real function fnc_decal(a)
                real, intent(in) :: a
                end function
            end interface
            procedure(fnc_decal) :: fnc
            real, intent(in) :: x
            real, intent(inout) :: dx
        end subroutine
    end interface

    print *, square(3.0)
    print *, grad_square(3.0)

    contains

    real function square(x)
        real, intent(in) :: x
        real, allocatable :: x_ptr

        allocate(x_ptr) !<--- for_allocate
        x_ptr = x

        square = x_ptr * x_ptr

        deallocate(x_ptr) !<--- for_dealloc_allocatable
    end function

    real function grad_square(x)
        real, intent(in) :: x

        grad_square = 0
        call square__enzyme_autodiff(square, x, grad_square);
    end function grad_square

end program app

! CHECK: 9
! CHECK-NEXT: 6
