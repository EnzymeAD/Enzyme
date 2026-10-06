! REQUIRES: flangenzyme_lld
! The config file is all a flang user adds. Without -flto, flang differentiates.
! With -flto, the call names a function compiled in another object file, so the
! pre-link run defers and lld differentiates the linked program.
! RUN: %fc -cpp -O2 %flangEnzymeConfig %s -o %t0 && %t0 | FileCheck %s
! RUN: %fc -cpp -DCALLEE_ONLY -flto -O2 -c %flangEnzymeConfig %s -o %t.callee.o
! RUN: %fc -cpp -DCALLER_ONLY -flto -O2 -c %flangEnzymeConfig %s -o %t.caller.o
! RUN: %fc -flto -O2 %flangEnzymeConfig %t.caller.o %t.callee.o -o %t1 && %t1 | FileCheck %s
! With -flto=thin, lld differentiates each module on its own after the thin
! link. Enzyme's pre-link run asks the thin link to import the function passed
! to __enzyme_autodiff, which ThinLTO would not import for a mere reference.
! RUN: %fc -cpp -DCALLEE_ONLY -flto=thin -O2 -c %flangEnzymeConfig %s -o %t.callee.thin.o
! RUN: %fc -cpp -DCALLER_ONLY -flto=thin -O2 -c %flangEnzymeConfig %s -o %t.caller.thin.o
! RUN: %fc -flto=thin -O2 %flangEnzymeConfig %t.caller.thin.o %t.callee.thin.o -o %t2 && %t2 | FileCheck %s

#ifndef CALLER_ONLY
real function cube(x)
  implicit none
  real, intent(in) :: x
  cube = x**3
end function cube
#endif

#ifndef CALLEE_ONLY
program main
  implicit none
  interface
    real function cube(x)
      implicit none
      real, intent(in) :: x
    end function cube
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
