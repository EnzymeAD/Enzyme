! memcpyopt fuses the stores that zero adjacent COMMON members into one
! memset, whose type LLVM Enzyme cannot deduce from LLVM IR ("Cannot deduce
! type of memset", as in MITgcm). The COMMON layout the plugin attaches to the
! global (!enzyme_type) types it. d/dx (a x^2) at a = 2, x = 3 is 12.
!
! REQUIRES: flang_directives, flangenzyme
! RUN: %fc %flangDirectives -O2 %loadFlangEnzyme -mllvm -enzyme-global-activity %s -o %t 2>&1 | FileCheck %s --check-prefix=COMPILE --allow-empty
! RUN: %t | FileCheck %s

! COMPILE-NOT: Cannot deduce type

subroutine reset_and_step(x)
  real(8), intent(in) :: x
  real(8) :: a, b, c, d, e, f, g, h, p, q, r, s
  common /state/ a, b, c, d, e, f, g, h, p, q, r, s
  real(8) :: a_d, b_d, c_d, d_d, e_d, f_d, g_d, h_d, p_d, q_d, r_d, s_d
  common /state_d/ a_d, b_d, c_d, d_d, e_d, f_d, g_d, h_d, p_d, q_d, r_d, s_d
  !dir$ enzyme shadow(/state/, shadow=/state_d/)
  b = 0.0d0
  c = 0.0d0
  d = 0.0d0
  e = 0.0d0
  f = 0.0d0
  g = 0.0d0
  h = 0.0d0
  p = 0.0d0
  q = 0.0d0
  r = 0.0d0
  s = 0.0d0
  b = a * x
  c = b * x
end subroutine

real(8) function cost(x)
  real(8), intent(in) :: x
  real(8) :: a, b, c, d, e, f, g, h, p, q, r, s
  common /state/ a, b, c, d, e, f, g, h, p, q, r, s
  call reset_and_step(x)
  cost = c
end function

program main
  implicit none
  real(8) :: a, b, c, d, e, f, g, h, p, q, r, s, a_d, b_d, c_d, d_d, e_d, f_d, g_d, h_d, p_d, q_d, r_d, s_d, x, dx
  common /state/ a, b, c, d, e, f, g, h, p, q, r, s
  common /state_d/ a_d, b_d, c_d, d_d, e_d, f_d, g_d, h_d, p_d, q_d, r_d, s_d
  real(8), external :: cost
  a = 2.0d0
  x = 3.0d0
  dx = 0.0d0
  call f__enzyme_autodiff(cost, x, dx)
  ! d/dx (a x^2) = 2 a x = 12
  print '(F6.2)', dx
end program

! CHECK: 12.00
