! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O0 -S -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 ]]; then %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O1 -S -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 ]]; then %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O2 -S -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 ]]; then %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O3 -S -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s; fi

! NOTE: This test is only configured to run with the flang compiler at -O0
!       For it to work with the ifx compiler we will need to figure out how to
!       handle the indirection involved in the enzyme_autodiff binding

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
    call enzyme_autodiff(square, x, dx)

    print *, dx
end program app

! CHECK: 9
! CHECK-NEXT: 6
