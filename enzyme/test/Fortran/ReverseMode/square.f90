! RUN: if [ %llvmver -ge 13 ]; then ifx -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -enzyme -o %t && ifx -flto -O0 %t -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then ifx -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -enzyme -o %t && ifx -flto -O1 %t -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then ifx -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -enzyme -o %t && ifx -flto -O2 %t -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then ifx -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -enzyme -o %t && ifx -flto -O3 %t -o %t1 && %t1 | FileCheck %s; fi

module math
contains
    real function square( x )
        real, intent(in) :: x
        square = x**2
    end function
end module math

program app
    use enzyme, only: enzyme_autodiff
    use math, only: square
    implicit none
    real :: x, dx

    x = 3
    print *, square(x)

    dx = 0
    call enzyme_autodiff(square, x, dx);

    print *, dx
end program app

! CHECK: 9
! CHECK-NEXT: 6