! REQUIRED: fortran
! UNSUPPORTED: ifx
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s %linkFortran -o %t0 && %t0 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s %linkFortran -o %t2 && %t2 | FileCheck %s %}

! A time loop whose state is a module array, as in codes that keep their
! state in COMMON blocks, reversed with binomial checkpointing. The gradient
! must match centred finite differences of the plain loop.

module model
  implicit none
  integer, parameter :: m = 6
  real(8) :: state(m) = 0
contains
  subroutine step(i, x)
    integer(8), value :: i
    real(8), intent(inout) :: x(m)
    real(8) :: tmp(m)
    integer :: k
    do k = 1, m
      tmp(k) = x(k) + 0.1d0 * (x(mod(k, m) + 1) - x(k)) &
               + 0.05d0 * sin(state(k) + 0.01d0 * i)
    end do
    ! Explicit loops: array assignment would call into the flang runtime.
    do k = 1, m
      state(k) = 0.8d0 * state(k) + 0.2d0 * tmp(k) * x(k)
      x(k) = tmp(k)
    end do
  end subroutine step

  real(8) function loss(x)
    real(8), intent(in) :: x(m)
    integer :: k
    loss = 0
    do k = 1, m
      loss = loss + x(k)**3 + state(k)
    end do
  end function loss

  subroutine init(x)
    real(8), intent(out) :: x(m)
    integer :: k
    do k = 1, m
      x(k) = 0.5d0 + 0.1d0 * k
      state(k) = 0.1d0 * k
    end do
  end subroutine init
end module model

subroutine plain(x, y, n)
  use model
  implicit none
  real(8), intent(inout) :: x(m)
  real(8), intent(out) :: y
  integer(8), intent(in) :: n
  integer(8) :: i
  do i = 0, n - 1
    call step(i, x)
  end do
  y = loss(x)
end subroutine plain

subroutine checkpointed(x, y, n, scheme, config)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use enzyme
  use model
  implicit none
  real(8), intent(inout) :: x(m)
  real(8), intent(out) :: y
  integer(8), intent(in) :: n
  type(c_ptr), intent(in) :: scheme
  type(enzyme_ckpt_config), intent(in) :: config
  call enzyme_checkpoint_for(step, 0_8, n, enzyme_scheme, scheme, config, &
                             enzyme_checkpoint_region, x, int(8 * m, 8), x)
  y = loss(x)
end subroutine checkpointed

program main
  use, intrinsic :: iso_c_binding, only: c_ptr, c_loc
  use enzyme
  use model
  implicit none
  integer(8), parameter :: n = 25
  real(8) :: x(m), dx(m), y, dy, yp, ym, fd(m), h
  type(c_ptr) :: scheme
  type(enzyme_ckpt_config), target :: config
  type(enzyme_ckpt_stats), target :: stats
  external :: checkpointed
  integer :: k
  logical :: ok

  ! Finite differences of the plain loop.
  h = 1d-6
  do k = 1, m
    call init(x)
    x(k) = x(k) + h
    call plain(x, yp, n)
    call init(x)
    x(k) = x(k) - h
    call plain(x, ym, n)
    fd(k) = (yp - ym) / (2 * h)
  end do

  scheme = enzyme_ckpt_revolve()
  config%snapshots = 3
  config%stats = c_loc(stats)
  call init(x)
  do k = 1, m
    dx(k) = 0
  end do
  dy = 1
  call enzyme_autodiff(checkpointed, enzyme_dup, x, dx, enzyme_dup, y, dy, &
                       enzyme_const, n, enzyme_const, scheme, &
                       enzyme_const, config)

  ok = all(abs(dx - fd) <= 1d-6 * abs(fd))
  if (stats%taped_steps /= n) ok = .false.
  if (stats%max_slots > config%snapshots) ok = .false.
  if (.not. ok) then
    print *, "gradient", dx
    print *, "expected", fd
    print *, "taped steps", stats%taped_steps, "slots", stats%max_slots
  end if
  ! CHECK: ok
  if (ok) print "(a)", "ok"
end program main
