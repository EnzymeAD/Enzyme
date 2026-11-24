! Gradient of the final temperatures and salinities of OpenAD's box model
! (Examples/BoxModel) with respect to the perturbations xx of the initial
! state. The model keeps its whole state in module all_globals_mod.
module boxmodel_wrapper
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  private
  public :: boxmodel

contains

  subroutine boxmodel(n, x, m, y)
    use all_globals_mod, only: xx, tnew, snew, ndim
    integer, intent(in) :: n, m
    real(real64), intent(in) :: x(n)
    real(real64), intent(out) :: y(m)
    external :: box_model_body

    xx = x
    call box_model_body
    y(1:ndim) = tnew
    y(ndim + 1:2 * ndim) = snew
  end subroutine boxmodel

end module boxmodel_wrapper

program boxmodel_enzyme
  use, intrinsic :: iso_fortran_env, only: real64
  use enzyme, only: enzyme_autodiff, enzyme_const, enzyme_dup
  use boxmodel_wrapper, only: boxmodel
  use gradient_check, only: check_gradient, fill_deterministic
  implicit none
  integer, parameter :: n = 6, m = 6
  real(real64) :: x(n), dx(n), y(m), dy(m), w(m), v(n)
  external :: box_ini_params

  call box_ini_params

  x = 0
  dx = 0
  call fill_deterministic(w, 1)
  call fill_deterministic(v, 2)
  dy = w

  call enzyme_autodiff(boxmodel, enzyme_const, n, enzyme_dup, x, dx, &
                       enzyme_const, m, enzyme_dup, y, dy)

  call check_gradient("BoxModel", boxmodel, n, x, m, w, dx, v, &
                      1.0e-6_real64, 1.0e-5_real64)
end program boxmodel_enzyme
