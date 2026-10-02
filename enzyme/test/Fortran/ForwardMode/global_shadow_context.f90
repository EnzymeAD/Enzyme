! REQUIRES: fortran
! RUN: %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s
! RUN: %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s

! Tangents with respect to a module variable and a COMMON block, seeded in
! their shadows in a context of width 2, one direction per lane.

module params
  implicit none
  real(8) :: scale = 2.0d0
end module params

module funcs
  use params
  implicit none
contains
  subroutine s(x, y)
    real(8), intent(in) :: x
    real(8), intent(out) :: y
    real(8) :: w(3)
    common /weights/ w
    y = scale * w(2) * x
  end subroutine s
end module funcs

program main
  use, intrinsic :: iso_c_binding, only: c_ptr, c_f_pointer
  use enzyme
  use params
  use funcs
  implicit none
  real(8) :: w(3)
  common /weights/ w
  type(c_ptr) :: ctx
  real(8), pointer :: dscale0, dscale1, dw0(:), dw1(:)
  real(8) :: x, y, dy0, dy1

  w = [1.0d0, 3.0d0, 5.0d0]
  x = 7.0d0

  ctx = enzyme_new_context(2)
  call c_f_pointer(enzyme_shadow(ctx, scale, 0), dscale0)
  call c_f_pointer(enzyme_shadow(ctx, scale, 1), dscale1)
  call c_f_pointer(enzyme_shadow(ctx, w, 0), dw0, [3])
  call c_f_pointer(enzyme_shadow(ctx, w, 1), dw1, [3])
  dscale0 = 1
  dw0 = 0
  dscale1 = 0
  dw1 = [0.0d0, 1.0d0, 0.0d0]

  call enzyme_fwddiff(s, enzyme_context, ctx, enzyme_const, x, &
                      enzyme_dup, y, dy0, dy1)

  ! CHECK: 42.
  print '(F4.0)', y
  ! CHECK: 21. 14.
  print '(2F4.0)', dy0, dy1
end program main
