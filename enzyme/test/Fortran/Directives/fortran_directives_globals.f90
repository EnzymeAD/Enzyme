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

subroutine step(x)
  real, intent(in) :: x
  real :: a, b
  common /state/ a, b
  real :: a_d, b_d
  common /state_d/ a_d, b_d
  !dir$ enzyme shadow(/state/, shadow=/state_d/)
  call timer_start()
  b = a * x
end subroutine

subroutine timer_start()
  !dir$ enzyme inactive
  real :: t
  common /timers/ t
  t = t + 1.0
end subroutine

real function cost(x)
  real, intent(in) :: x
  real :: a, b
  common /state/ a, b
  call step(x)
  cost = b
end function

module m
  implicit none
contains
  real function f(x, seed)
    real, intent(in) :: x
    logical, intent(in) :: seed
    real, save :: s = 2.0
    real, save :: s_d = 0.0
    !dir$ enzyme shadow(s, shadow=s_d)
    if (seed) s_d = 1.0
    f = x * s
  end function
end module

program main
  use enzyme, only: enzyme_const
  use m
  implicit none
  real :: a, b, a_d, b_d, x, dx, t
  common /state/ a, b
  common /state_d/ a_d, b_d
  real, external :: cost, f__enzyme_fwddiff
  a = 2.0
  x = 3.0
  a_d = 1.0
  b_d = 0.0
  dx = 0.0
  print '(F5.1)', f__enzyme_fwddiff(cost, x, dx)
  t = f(x, .true.)
  print '(F5.1)', f__enzyme_fwddiff(f, x, dx, enzyme_const, .false.)
end program

! CHECK: 3.0
! CHECK-NEXT: 3.0
