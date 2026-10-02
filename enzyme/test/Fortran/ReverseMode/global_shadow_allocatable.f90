! REQUIRES: fortran
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s

! An allocatable module variable gets a zeroed shadow with each ALLOCATE,
! and enzyme_zero_shadows zeroes the shadows in a context between gradients,
! so that they do not accumulate.

module state
  implicit none
  real(8), allocatable :: a(:)
  real(8) :: scale = 2.0d0
end module state

module funcs
  use state
  implicit none
contains
  real(8) function f(x)
    real(8), intent(in) :: x
    f = g(a, x)
  end function f

  real(8) function g(v, x)
    real(8), intent(in) :: v(3), x
    integer :: i
    g = 0
    do i = 1, 3
      g = g + scale * v(i) * x * x
    end do
  end function g
end module funcs

program main
  use, intrinsic :: iso_c_binding, only: c_ptr, c_f_pointer
  use enzyme
  use state
  use funcs
  implicit none
  type(c_ptr) :: ctx
  real(8), pointer :: da(:), dscale
  real(8) :: x, dx
  integer :: iter

  allocate(a(3))
  a = [1.0d0, 2.0d0, 3.0d0]
  ctx = enzyme_new_context(1)
  call c_f_pointer(enzyme_shadow(ctx, a, 0), da, [3])
  call c_f_pointer(enzyme_shadow(ctx, scale, 0), dscale)

  ! CHECK: 1 48. 8. 8. 8. 24.
  ! CHECK: 2 72. 18. 18. 18. 54.
  do iter = 1, 2
    call enzyme_zero_shadows(ctx)
    x = iter + 1
    dx = 0
    call enzyme_autodiff(f, enzyme_context, ctx, enzyme_dup, x, dx)
    print '(I1,5F5.0)', iter, dx, da, dscale
  end do
  deallocate(a)
end program main
