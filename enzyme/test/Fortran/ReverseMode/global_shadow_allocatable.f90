! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s

! An allocatable module variable gets a zeroed shadow with each ALLOCATE,
! and enzyme_zero_shadows zeroes the shadows in a context between gradients,
! so that they do not accumulate.

module state
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  private
  real(real64), allocatable, public :: a(:)
  real(real64), public :: scale = 2.0_real64
end module state

module funcs
  use, intrinsic :: iso_fortran_env, only: real64
  use state, only: a, scale
  implicit none
  private
  public :: f
contains
  real(real64) function f(x)
    real(real64), intent(in) :: x
    f = g(a, x)
  end function f

  real(real64) function g(v, x)
    real(real64), intent(in) :: v(3), x
    integer :: i
    g = 0
    do i = 1, 3
      g = g + scale * v(i) * x * x
    end do
  end function g
end module funcs

program main
  use, intrinsic :: iso_c_binding, only: c_ptr, c_f_pointer
  use, intrinsic :: iso_fortran_env, only: real64
  use enzyme, only: enzyme_autodiff, enzyme_context, enzyme_dup, &
                    enzyme_new_context, enzyme_shadow, enzyme_zero_shadows
  use state, only: a, scale
  use funcs, only: f
  implicit none
  type(c_ptr) :: ctx
  real(real64), pointer :: da(:), dscale
  real(real64) :: x, dx
  integer :: iter

  allocate(a(3))
  a = [1.0_real64, 2.0_real64, 3.0_real64]
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
    print "(I1,5F5.0)", iter, dx, da, dscale
  end do
  deallocate(a)
end program main
