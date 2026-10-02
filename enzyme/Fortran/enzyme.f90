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
  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_ptr, c_null_ptr
  use enzyme_function_hooks, only: enzyme_autodiff => f__enzyme_autodiff, &
                                   enzyme_fwddiff  => f__enzyme_fwddiff, &
                                   enzyme_function_like => &
                                     f__enzyme_function_like, &
                                   enzyme_checkpoint_for => &
                                     f__enzyme_checkpoint_for, &
                                   enzyme_fixed_point => &
                                     f__enzyme_fixed_point
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

  ! Bindings for checkpointed loops (see enzyme/checkpoint.h). A loop
  !
  !   do i = start, start + n - 1
  !     call step(i, args...)
  !   end do
  !
  ! whose step takes `integer(8), value :: i` is written
  !
  !   call enzyme_checkpoint_for(step, start, n, enzyme_scheme, scheme, config, &
  !                              [enzyme_checkpoint_region, array, bytes,] &
  !                              args...)
  !
  ! with integer(8) start, n and bytes, scheme a type(c_ptr) from
  ! enzyme_ckpt_revolve() and its siblings, and config an enzyme_ckpt_config.
  integer(c_int), public, bind(C, name="enzyme_scheme") :: enzyme_scheme
  integer(c_int), public, bind(C, name="enzyme_checkpoint_region") :: &
    enzyme_checkpoint_region

  ! Bindings for fixed-point loops (see enzyme/fixed_point.h). A loop
  !
  !   i = 0
  !   do while (step(i, args...))
  !     i = i + 1
  !   end do
  !
  ! whose step is a logical (or integer) function taking `integer(8), value ::
  ! i` and iterating the state z, is written
  !
  !   call enzyme_fixed_point(step, enzyme_fp_state, z, bytes, &
  !                           [enzyme_fp_reduction, r,] [enzyme_fp_max_iters, n,]
  !                           [enzyme_fp_control, control,] &
  !                           [enzyme_checkpoint_region, array, bytes,] args...)
  !
  ! with integer(8) bytes and n, real(8) r, and control an integer function
  ! control(cumul, reduction) with real(8) arguments (Tapenade's
  ! adFixedPoint_notReduced). The iterations are not differentiated one by
  ! one: the adjoint (in reverse mode) or the tangent (in forward mode) of the
  ! last one is iterated to convergence.
  integer(c_int), public, bind(C, name="enzyme_fp_state") :: enzyme_fp_state
  integer(c_int), public, bind(C, name="enzyme_fp_reduction") :: &
    enzyme_fp_reduction
  integer(c_int), public, bind(C, name="enzyme_fp_max_iters") :: &
    enzyme_fp_max_iters
  integer(c_int), public, bind(C, name="enzyme_fp_control") :: enzyme_fp_control

  type, public, bind(C) :: enzyme_ckpt_stats
    integer(c_int64_t) :: forward_steps = 0
    integer(c_int64_t) :: taped_steps = 0
    integer(c_int64_t) :: stores = 0
    integer(c_int64_t) :: restores = 0
    integer(c_int64_t) :: max_slots = 0
    integer(c_int64_t) :: max_bytes = 0
  end type enzyme_ckpt_stats

  type, public, bind(C) :: enzyme_ckpt_config
    ! Revolve: number of snapshot slots. Periodic: number of segments.
    integer(c_int64_t) :: snapshots = 1
    ! 1 prints a summary, 2 also every action.
    integer(c_int) :: verbose = 0
    ! A C string (c_loc of a character array ending in c_null_char), or
    ! c_null_ptr to keep every snapshot in memory.
    type(c_ptr) :: spill_dir = c_null_ptr
    ! With spill_dir: slots whose start is past this many bytes go to files.
    integer(c_int64_t) :: mem_budget = 0
    ! c_loc of an enzyme_ckpt_stats to fill in, or c_null_ptr.
    type(c_ptr) :: stats = c_null_ptr
  end type enzyme_ckpt_config

  interface
    function enzyme_ckpt_revolve() result(scheme) &
        bind(C, name="enzyme_ckpt_revolve_scheme")
      import :: c_ptr
      type(c_ptr) :: scheme
    end function enzyme_ckpt_revolve
    function enzyme_ckpt_periodic() result(scheme) &
        bind(C, name="enzyme_ckpt_periodic_scheme")
      import :: c_ptr
      type(c_ptr) :: scheme
    end function enzyme_ckpt_periodic
    function enzyme_ckpt_store_all() result(scheme) &
        bind(C, name="enzyme_ckpt_store_all_scheme")
      import :: c_ptr
      type(c_ptr) :: scheme
    end function enzyme_ckpt_store_all
  end interface

  ! Bindings for function hooks
  public :: enzyme_autodiff
  public :: enzyme_fwddiff
  public :: enzyme_function_like
  public :: enzyme_checkpoint_for
  public :: enzyme_fixed_point
  public :: enzyme_ckpt_revolve, enzyme_ckpt_periodic, enzyme_ckpt_store_all
end module enzyme
