! REQUIRES: flangenzyme_lld
! The config file is all a flang user adds. Without -flto, flang differentiates.
! With -flto, the call names a function compiled in another object file, so the
! pre-link run defers and lld differentiates the linked program.
! RUN: rm -rf %t.mods && mkdir -p %t.mods
! RUN: %fc -cpp -O2 %flangEnzymeConfig -module-dir %t.mods %s -o %t0 && %t0 | FileCheck %s
! RUN: %fc -cpp -DCALLEE_ONLY -flto -O2 -c %flangEnzymeConfig -module-dir %t.mods %s -o %t.callee.o
! RUN: %fc -cpp -DCALLER_ONLY -flto -O2 -c %flangEnzymeConfig -I %t.mods %s -o %t.caller.o
! RUN: %fc -flto -O2 %flangEnzymeConfig %t.caller.o %t.callee.o -o %t1 && %t1 | FileCheck %s
! With -flto=thin, lld differentiates each module on its own, which then has to
! define the function: ThinLTO imports functions that are called, not ones that
! are only passed to __enzyme_autodiff, so here both are in one object file.
! RUN: %fc -cpp -flto=thin -O2 -c %flangEnzymeConfig -module-dir %t.mods %s -o %t.thin.o
! RUN: %fc -flto=thin -O2 %flangEnzymeConfig %t.thin.o -o %t2 && %t2 | FileCheck %s

#ifndef CALLER_ONLY
module config_file_lto_cube
  implicit none
  private
  public :: cube

contains

  real function cube(x)
    real, intent(in) :: x

    cube = x**3
  end function cube

end module config_file_lto_cube
#endif

#ifndef CALLEE_ONLY
program main
  use config_file_lto_cube, only: cube
  implicit none

  interface
    subroutine cube__enzyme_autodiff(fn, x, dx)
      implicit none

      interface
        real function fn_decl(a)
          implicit none
          real, intent(in) :: a
        end function fn_decl
      end interface

      procedure(fn_decl) :: fn
      real, intent(in) :: x
      real, intent(inout) :: dx
    end subroutine cube__enzyme_autodiff
  end interface

  real :: x, dx

  x = 2
  dx = 0
  call cube__enzyme_autodiff(cube, x, dx)
  print *, dx
end program main
#endif

! CHECK: 12
