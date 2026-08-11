! ===- enzyme_function_hooks.f90 - Fortran bindings function hooks --------=== !
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
!  This file provides Fortran bindings for Enzyme's function hooks.
!
!  The fact that the double-underscore function hook names appears in the
!  implicit interfaces defined in this module is sufficient to get Enzyme to be
!  applied in the appropriate way. We provide cleaner bindings without
!  double-underscores in the main Fortran Enzyme module in enzyme.f90.
!
! ===----------------------------------------------------------------------=== !
module enzyme_function_hooks
  implicit none
  private

  ! Bindings for function hooks
  ! NOTE: Leading underscores are not permitted by some Fortran compilers so we
  !       prepend with 'f' in the Fortran versions of the function hooks.
  public :: f__enzyme_autodiff
  public :: f__enzyme_fwddiff
  external :: f__enzyme_autodiff
  external :: f__enzyme_fwddiff
end module enzyme_function_hooks
