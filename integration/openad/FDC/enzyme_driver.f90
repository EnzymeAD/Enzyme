! Gradient of the residual of OpenAD's flow in a driven cavity example
! (Examples/FDC, the MINPACK-2 dfdcfj problem) on a 10 x 10 grid with
! Reynolds number 10, at the problem's standard starting point.
module fdc_wrapper
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  private
  public :: fdc, nx, ny

  integer, parameter :: nx = 10, ny = 10
  real(real64), parameter :: reynolds = 10

contains

  subroutine fdc(n, x, m, y)
    integer, intent(in) :: n, m
    real(real64), intent(in) :: x(n)
    real(real64), intent(out) :: y(m)
    real(real64) :: r
    external :: head

    r = reynolds
    call head(nx, ny, x, y, r)
  end subroutine fdc

end module fdc_wrapper

program fdc_enzyme
  use, intrinsic :: iso_fortran_env, only: real64
  use enzyme, only: enzyme_autodiff, enzyme_const, enzyme_dup
  use fdc_wrapper, only: fdc, nx, ny
  use gradient_check, only: check_gradient, fill_deterministic
  implicit none
  integer, parameter :: n = nx * ny, m = nx * ny
  real(real64) :: x(n), dx(n), y(m), dy(m), w(m), v(n)
  external :: init

  call init(nx, ny, x)
  dx = 0
  call fill_deterministic(w, 1)
  call fill_deterministic(v, 2)
  dy = w

  call enzyme_autodiff(fdc, enzyme_const, n, enzyme_dup, x, dx, &
                       enzyme_const, m, enzyme_dup, y, dy)

  call check_gradient("FDC", fdc, n, x, m, w, dx, v, &
                      1.0e-6_real64, 1.0e-6_real64)
end program fdc_enzyme
