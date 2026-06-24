! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O0 -S -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O1 -S -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O2 -S -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O3 -S -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s

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
    call square__enzyme_autodiff(square, x, dx)

    print *, dx
end program app

! CHECK: 9
! CHECK-NEXT: 6
