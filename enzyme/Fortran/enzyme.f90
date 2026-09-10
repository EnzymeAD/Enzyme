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

  ! Symbolic function names for enzyme_function_like
  integer(c_int), public, bind(C, name="enzyme_math_sin")   :: enzyme_sin
  integer(c_int), public, bind(C, name="enzyme_math_cos")   :: enzyme_cos
  integer(c_int), public, bind(C, name="enzyme_math_tan")   :: enzyme_tan
  integer(c_int), public, bind(C, name="enzyme_math_asin")  :: enzyme_asin
  integer(c_int), public, bind(C, name="enzyme_math_acos")  :: enzyme_acos
  integer(c_int), public, bind(C, name="enzyme_math_atan")  :: enzyme_atan
  integer(c_int), public, bind(C, name="enzyme_math_atan2") :: enzyme_atan2
  integer(c_int), public, bind(C, name="enzyme_math_exp")   :: enzyme_exp
  integer(c_int), public, bind(C, name="enzyme_math_exp2")  :: enzyme_exp2
  integer(c_int), public, bind(C, name="enzyme_math_exp10") :: enzyme_exp10
  integer(c_int), public, bind(C, name="enzyme_math_expm1") :: enzyme_expm1
  integer(c_int), public, bind(C, name="enzyme_math_log")   :: enzyme_log
  integer(c_int), public, bind(C, name="enzyme_math_log2")  :: enzyme_log2
  integer(c_int), public, bind(C, name="enzyme_math_log10") :: enzyme_log10
  integer(c_int), public, bind(C, name="enzyme_math_log1p") :: enzyme_log1p
  integer(c_int), public, bind(C, name="enzyme_math_acosh") :: enzyme_acosh
  integer(c_int), public, bind(C, name="enzyme_math_asinh") :: enzyme_asinh
  integer(c_int), public, bind(C, name="enzyme_math_atanh") :: enzyme_atanh
  integer(c_int), public, bind(C, name="enzyme_math_sqrt")  :: enzyme_sqrt
  integer(c_int), public, bind(C, name="enzyme_math_cbrt")  :: enzyme_cbrt
  integer(c_int), public, bind(C, name="enzyme_math_hypot") :: enzyme_hypot
  integer(c_int), public, bind(C, name="enzyme_math_pow")   :: enzyme_pow
  integer(c_int), public, bind(C, name="enzyme_math_erf")   :: enzyme_erf
  integer(c_int), public, bind(C, name="enzyme_math_erfc")  :: enzyme_erfc

  ! Bindings for function hooks
  public :: enzyme_autodiff
  public :: enzyme_fwddiff
  public :: enzyme_function_like
end module enzyme
