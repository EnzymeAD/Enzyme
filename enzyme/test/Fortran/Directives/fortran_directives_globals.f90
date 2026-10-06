! !DIR$ ENZYME shadow on globals that a C table cannot name, LLVM route:
! - a COMMON block, paired with its shadow block as a whole (MITgcm's state);
! - a SAVEd local, which flang emits as an internal global (ICON's saved
!   pointers), paired from inside its procedure.
! The forward derivative along the seeded shadow is x = 3 in both cases;
! without the pairing it is silently 0. timer_start is inactive.
!
! REQUIRES: flang_directives, flangenzyme
! RUN: %fc %flangDirectives -O0 %loadFlangEnzyme -mllvm -enzyme-global-activity %loadFortran %s -o %t0 && %t0 | FileCheck %s
! RUN: %fc %flangDirectives -O2 %loadFlangEnzyme -mllvm -enzyme-global-activity %loadFortran %s -o %t2 && %t2 | FileCheck %s

! allow(procedure-not-in-module)
subroutine step(x)
  implicit none
  real, intent(in) :: x
  real :: a, b
  ! allow(common-block)
  common /state/ a, b
  real :: a_d, b_d
  ! allow(common-block)
  common /state_d/ a_d, b_d
  !dir$ enzyme shadow(/state/, shadow=/state_d/)
  call timer_start()
  b = a * x
end subroutine step

! allow(procedure-not-in-module)
subroutine timer_start()
  implicit none
  !dir$ enzyme inactive
  real :: t
  ! allow(common-block)
  common /timers/ t
  t = t + 1.0
end subroutine timer_start

! allow(procedure-not-in-module)
real function cost(x)
  implicit none
  real, intent(in) :: x
  real :: a, b
  ! allow(common-block)
  common /state/ a, b
  call step(x)
  cost = b
end function cost

module m
  implicit none
  public
contains
  real function f(x, seed)
    real, intent(in) :: x
    logical, intent(in) :: seed
    real, save :: s = 2.0
    real, save :: s_d = 0.0
    !dir$ enzyme shadow(s, shadow=s_d)
    if (seed) s_d = 1.0
    f = x * s
  end function f
end module m

program main
  use enzyme, only: enzyme_const
  use m, only: f
  implicit none
  real :: a, b, a_d, b_d, x, dx, t
  ! allow(common-block)
  common /state/ a, b
  ! allow(common-block)
  common /state_d/ a_d, b_d
  real, external :: cost, f__enzyme_fwddiff
  a = 2.0
  x = 3.0
  a_d = 1.0
  b_d = 0.0
  dx = 0.0
  print "(F5.1)", f__enzyme_fwddiff(cost, x, dx)
  t = f(x, .true.)
  print "(F5.1)", f__enzyme_fwddiff(f, x, dx, enzyme_const, .false.)
end program main

! CHECK: 3.0
! CHECK-NEXT: 3.0
