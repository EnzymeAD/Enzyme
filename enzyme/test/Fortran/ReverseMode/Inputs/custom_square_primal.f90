! Primal for custom_rule_from_c.f90, whose derivative is overridden by a rule
! registered from a C translation unit.
!
! It lives in its own translation unit on purpose: Enzyme matches a custom rule
! against the *call* to fsquare, so if flang were free to inline the body into
! the caller the rule would never fire and Enzyme would silently differentiate
! the inlined body instead.  Keeping it external also mirrors the realistic
! case, where the routine being overridden comes from another file or library.
!
! bind(C) gives the routine a stable, unmangled symbol for the C side to name.
! Without `value` the arguments are passed by reference as usual, so the C
! signature is `void fsquare(const double *x, double *y)`.

subroutine fsquare(x, y) bind(C, name="fsquare")
  use iso_c_binding, only: c_double
  implicit none
  real(c_double), intent(in) :: x
  real(c_double), intent(out) :: y
  y = x * x
end subroutine fsquare
