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
  use iso_c_binding, only: c_int
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

  ! Bindings for function hooks
  public :: __enzyme_autodiff
  public :: __enzyme_fwddiff
  external :: __enzyme_autodiff
  external :: __enzyme_fwddiff
end module enzyme
