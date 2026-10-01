! !DIR$ ENZYME on external procedures that the unit with the directives only
! declares, as MITgcm's F77 routines and its generated list of passive
! routines: the module below defines nothing and is used by nobody, yet its
! unit registers
! - a custom rule for ext_double (2x, differentiated like log1p: 1/(1+x) =
!   0.3333 at x = 2), declared by an interface body, and
! - ext_scale (x) as inactive, declared EXTERNAL: d/dx (x * ext_scale(x)) is
!   then ext_scale(3) = 3.
! The registrations are weak, so that they survive optimization before an
! LTO link; Enzyme reads and removes them.
!
! REQUIRES: flang_directives, flangenzyme
! RUN: %fc -fc1 %fc1Directives -emit-fir %loadFortran %s -o - | FileCheck %s --check-prefix=FIR
! RUN: %fc %flangDirectives -O0 %loadFlangEnzyme %loadFortran %s -o %t0 && %t0 | FileCheck %s
! RUN: %fc %flangDirectives -O2 %loadFlangEnzyme %loadFortran %s -o %t2 && %t2 | FileCheck %s
! With LTO, Enzyme runs only in the link, after the optimization of each unit.
! RUN: %fc %flangDirectives -O2 -flto=full %loadFortran -c %s -o %t.o
! RUN: %fc -O2 %lldEnzyme %t.o -o %t3 && %t3 | FileCheck %s

! FIR-DAG: fir.global weak @ext_double_.__enzyme_register_gradient
! FIR-DAG: fir.global weak @ext_scale_.__enzyme_inactivefn
! FIR-DAG: fir.global weak @ext_scale_.__enzyme_nofree

module registrations
  implicit none
  private
  interface
    subroutine ext_double(x, y)
      real, intent(in) :: x
      real, intent(out) :: y
    end subroutine
    subroutine ext_double_aug(x, dx, y, dy)
      real, intent(in) :: x, dx
      real, intent(out) :: y
      real, intent(inout) :: dy
    end subroutine
    subroutine ext_double_rev(x, dx, y, dy)
      real, intent(in) :: x, y
      real, intent(inout) :: dx, dy
    end subroutine
  end interface
  external :: ext_scale
  !dir$ enzyme custom_rule(ext_double, augmented=ext_double_aug, reverse=ext_double_rev)
  !dir$ enzyme inactive(ext_scale)
end module

subroutine ext_double(x, y)
  real, intent(in) :: x
  real, intent(out) :: y
  y = 2.0 * x
end subroutine
subroutine ext_double_aug(x, dx, y, dy)
  real, intent(in) :: x, dx
  real, intent(out) :: y
  real, intent(inout) :: dy
  call ext_double(x, y)
end subroutine
subroutine ext_double_rev(x, dx, y, dy)
  real, intent(in) :: x, y
  real, intent(inout) :: dx, dy
  dx = dx + dy / (1.0 + x)
  dy = 0.0
end subroutine
real function ext_scale(x)
  real, intent(in) :: x
  ext_scale = x
end function

real function wrapper(x)
  real, intent(in) :: x
  call ext_double(x, wrapper)
end function
real function scaled(x)
  real, intent(in) :: x
  real, external :: ext_scale
  scaled = x * ext_scale(x)
end function

program main
  use enzyme, only: enzyme_autodiff
  implicit none
  real :: x, dx
  real, external :: wrapper, scaled
  x = 2.0
  dx = 0.0
  call enzyme_autodiff(wrapper, x, dx)
  print '(F6.4)', dx
  x = 3.0
  dx = 0.0
  call enzyme_autodiff(scaled, x, dx)
  print '(F6.4)', dx
end program

! CHECK: 0.3333
! CHECK: 3.0000
