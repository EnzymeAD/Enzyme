! Compares the reverse-mode gradient from Enzyme with central finite
! differences of the primal along a fixed direction (a dot product test):
!   <J^T w, v>  versus  (<w, f(x + h v)> - <w, f(x - h v)>) / (2 h)
module gradient_check
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  private
  public :: check_gradient, fill_deterministic

  abstract interface
    subroutine primal(n, x, m, y)
      import :: real64
      implicit none
      integer, intent(in) :: n, m
      real(real64), intent(in) :: x(n)
      real(real64), intent(out) :: y(m)
    end subroutine primal
  end interface

contains

  ! Fills v with values in [0.5, 1.5) that do not depend on the compiler's
  ! random number generator.
  subroutine fill_deterministic(v, seed)
    real(real64), intent(out) :: v(:)
    integer, intent(in) :: seed
    integer :: i
    do i = 1, size(v)
      v(i) = 0.5_real64 + modulo(0.6180339887498949_real64 * (i + seed), &
                                 1.0_real64)
    end do
  end subroutine fill_deterministic

  ! Stops with a nonzero exit code when the relative error exceeds tol.
  subroutine check_gradient(name, f, n, x, m, w, dx, v, h, tol)
    character(len=*), intent(in) :: name
    procedure(primal) :: f
    integer, intent(in) :: n, m
    real(real64), intent(in) :: x(n), w(m), dx(n), v(n), h, tol
    real(real64) :: yp(m), ym(m), fd, ad, err

    call f(n, x + h * v, m, yp)
    call f(n, x - h * v, m, ym)
    fd = (dot_product(w, yp) - dot_product(w, ym)) / (2 * h)
    ad = dot_product(dx, v)
    err = abs(ad - fd) / max(abs(fd), tiny(fd))

    write (*, '(a, ": adjoint ", es24.16, "  finite difference ", es24.16, &
              &"  relative error ", es9.2)') name, ad, fd, err
    if (.not. (err <= tol)) then
      write (*, '(a, ": FAIL (tolerance ", es9.2, ")")') name, tol
      error stop 1
    end if
    write (*, '(a, ": PASS")') name
  end subroutine check_gradient

end module gradient_check
