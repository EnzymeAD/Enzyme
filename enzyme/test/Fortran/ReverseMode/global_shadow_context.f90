! REQUIRES: fortran
! UNSUPPORTED: ifx
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s

! The gradient with respect to a module variable and a COMMON block, read
! from their shadows in a context.

module params
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  private
  real(real64), public :: scale = 2.0_real64
end module params

module funcs
  use, intrinsic :: iso_fortran_env, only: real64
  use params, only: scale
  implicit none
  private
  public :: f
contains
  real(real64) function f(x)
    real(real64), intent(in) :: x(3)
    real(real64) :: w(3)
    ! allow(OB011)
    common /weights/ w
    integer :: i
    f = 0
    do i = 1, 3
      f = f + scale * w(i) * x(i)
    end do
  end function f
end module funcs

program main
  use, intrinsic :: iso_c_binding, only: c_ptr, c_f_pointer
  use, intrinsic :: iso_fortran_env, only: real64
  use enzyme, only: enzyme_autodiff, enzyme_context, enzyme_dup, &
                    enzyme_new_context, enzyme_shadow
  use params, only: scale
  use funcs, only: f
  implicit none
  real(real64) :: w(3)
  ! allow(OB011)
  common /weights/ w
  type(c_ptr) :: ctx
  real(real64), pointer :: dscale, dw(:), dw2
  real(real64) :: x(3), dx(3)

  w = [1.0_real64, 2.0_real64, 3.0_real64]
  x = [4.0_real64, 5.0_real64, 6.0_real64]
  dx = 0

  ctx = enzyme_new_context(1)
  call c_f_pointer(enzyme_shadow(ctx, scale, 0), dscale)
  call c_f_pointer(enzyme_shadow(ctx, w, 0), dw, [3])
  call c_f_pointer(enzyme_shadow(ctx, w(2), 0), dw2)
  dscale = 0
  dw = 0

  call enzyme_autodiff(f, enzyme_context, ctx, enzyme_dup, x, dx)

  ! CHECK: 2. 4. 6.
  print "(3F4.0)", dx
  ! CHECK: 32.
  print "(F4.0)", dscale
  ! CHECK: 8. 10. 12.
  print "(3F4.0)", dw
  ! CHECK: 10.
  print "(F4.0)", dw2
end program main
