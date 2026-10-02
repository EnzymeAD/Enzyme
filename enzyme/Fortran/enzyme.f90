! ===- enzyme.f90 - Fortran bindings for Enzyme ---------------------------=== !
!
!                              Enzyme Project
!
!  Part of the Enzyme Project, under the Apache License v2.0 with LLVM
!  Exceptions. See https://llvm.org/LICENSE.txt for license information.
!  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!  If using this code in an academic setting, please cite the following:
!  @misc{enzymeGithub,
!   author = {William S. Moses and Valentin Churavy},
!   title = {Enzyme: High Performance Automatic Differentiation of LLVM},
!   year = {2020},
!   howpublished = {\url{https://github.com/wsmoses/Enzyme}},
!   note = {commit xxxxxxx}
!  }
!
! ===----------------------------------------------------------------------=== !
!
!  This file provides Fortran bindings for Enzyme.
!
! ===----------------------------------------------------------------------=== !
module enzyme
  use, intrinsic :: iso_c_binding, only: c_int, c_int8_t, c_ptr
  use enzyme_function_hooks, only: enzyme_autodiff => f__enzyme_autodiff, &
                                   enzyme_fwddiff  => f__enzyme_fwddiff, &
                                   enzyme_function_like => &
                                     f__enzyme_function_like
  implicit none
  private

  ! Bindings for activity descriptors
  integer(c_int), public, bind(C, name="enzyme_const")     :: enzyme_const
  integer(c_int), public, bind(C, name="enzyme_dup")       :: enzyme_dup
  integer(c_int), public, bind(C, name="enzyme_dupnoneed") :: enzyme_dupnoneed
  integer(c_int), public, bind(C, name="enzyme_out")       :: enzyme_out
  integer(c_int), public, bind(C, name="enzyme_scalar")    :: enzyme_scalar
  integer(c_int), public, bind(C, name="enzyme_width")     :: enzyme_width
  integer(c_int), public, bind(C, name="enzyme_vector")    :: enzyme_vector
  integer(c_int), public, bind(C, name="enzyme_context")   :: enzyme_context

  ! Bindings for shadow contexts. A context holds a shadow of every global
  ! (module variable, COMMON block, SAVE variable) at a fixed width:
  !
  !   type(c_ptr) :: ctx
  !   real, pointer :: dg
  !   ctx = enzyme_new_context(1)
  !   call c_f_pointer(enzyme_shadow(ctx, g, 0), dg)
  !   dg = 0
  !   call enzyme_autodiff(f, enzyme_context, ctx, x, dx)
  !
  ! after which dg holds the derivative with respect to g. Each call of
  ! enzyme_new_context in the source is one context, which must be made in
  ! the procedure that uses it. enzyme_shadow takes a global, an element of
  ! one or a whole array, and the lane, from 0, of the shadow to return. An
  ! allocatable module variable has a shadow allocated, zeroed, and freed with
  ! it by ALLOCATE and DEALLOCATE. enzyme_zero_shadows(ctx) zeroes every
  ! shadow in the context, e.g. before each gradient of a loop.
  interface
    function enzyme_new_context(width) result(ctx) &
        bind(C, name="__enzyme_context")
      import :: c_int, c_ptr
      implicit none
      integer(c_int), value :: width
      type(c_ptr) :: ctx
    end function enzyme_new_context
    function enzyme_shadow(ctx, var, lane) result(shadow) &
        bind(C, name="__enzyme_shadow")
      import :: c_int, c_int8_t, c_ptr
      implicit none
      type(c_ptr), value :: ctx
      !dir$ ignore_tkr(tkr) var
      ! Any variable, passed by its address.
      ! allow(C071)
      integer(c_int8_t), intent(in) :: var(*)
      integer(c_int), value :: lane
      type(c_ptr) :: shadow
    end function enzyme_shadow
    subroutine enzyme_zero_shadows(ctx) bind(C, name="__enzyme_zero_shadows")
      import :: c_ptr
      implicit none
      type(c_ptr), value :: ctx
    end subroutine enzyme_zero_shadows
  end interface

  ! Bindings for function hooks
  public :: enzyme_autodiff
  public :: enzyme_fwddiff
  public :: enzyme_function_like
  public :: enzyme_new_context, enzyme_shadow, enzyme_zero_shadows
end module enzyme
