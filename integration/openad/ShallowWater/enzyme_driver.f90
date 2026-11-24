! Gradient of the zonal volume transport of OpenAD's shallow water example
! (Examples/ShallowWater: channel flow over a sill, Losch and Wunsch 2003)
! with respect to its control vector, the bathymetry. The gradient is compared
! with the one OpenAD computed (shallow_water_ad.out, passed in as
! grad_reference.txt) and with finite differences.
!
! Compiled with -fdefault-real-8, so the default real is real64 here as in the
! example's sources.
module shallow_water_wrapper
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  private
  public :: shallow_water

contains

  subroutine shallow_water(n, x, m, y)
    integer, intent(in) :: n, m
    real(real64), intent(in) :: x(n)
    real(real64), intent(out) :: y(m)
    real(real64) :: cost
    external :: forward_model

    call forward_model(n, x, cost)
    y(1) = cost
  end subroutine shallow_water

end module shallow_water_wrapper

program shallow_water_enzyme
  use, intrinsic :: iso_fortran_env, only: real64
  use enzyme, only: enzyme_autodiff, enzyme_const, enzyme_dup
  use adsize, only: ncmax
  use shallow_water_wrapper, only: shallow_water
  use gradient_check, only: check_gradient, fill_deterministic
  implicit none
  integer, parameter :: m = 1
  real(real64), parameter :: reference_tolerance = 1.0e-8_real64
  real(real64) :: xc(ncmax), dxc(ncmax), y(m), dy(m), w(m)
  real(real64), allocatable :: v(:), reference(:)
  real(real64) :: t0, err, max_err
  integer :: nc, i, index, unit
  external :: inifields, readparms, make_weights, read_eta_data
  external :: length_of_control_vector, map_to_control_vector

  call inifields
  call readparms
  call make_weights
  t0 = 0
  call read_eta_data(t0)
  call length_of_control_vector(nc)
  call map_to_control_vector(nc, xc)

  dxc = 0
  w = 1
  dy = w
  call enzyme_autodiff(shallow_water, enzyme_const, nc, enzyme_dup, xc, dxc, &
                       enzyme_const, m, enzyme_dup, y, dy)

  allocate (reference(nc))
  open (newunit=unit, file="grad_reference.txt", status="old", action="read")
  do i = 1, nc
    read (unit, *) index, reference(i)
  end do
  close (unit)

  max_err = 0
  do i = 1, nc
    err = abs(dxc(i) - reference(i)) / maxval(abs(reference))
    max_err = max(max_err, err)
  end do
  write (*, '("ShallowWater: largest difference to the OpenAD gradient, ", &
            &"relative to its largest entry ", es9.2)') max_err
  if (.not. (max_err <= reference_tolerance)) then
    write (*, '("ShallowWater: FAIL (tolerance ", es9.2, ")")') &
      reference_tolerance
    error stop 1
  end if

  allocate (v(nc))
  call fill_deterministic(v, 2)
  call check_gradient("ShallowWater", shallow_water, nc, xc(1:nc), m, w, &
                      dxc(1:nc), v, 1.0e-6_real64, 1.0e-5_real64)
end program shallow_water_enzyme
