! !DIR$ ENZYME shadow in a unit that neither defines nor otherwise uses the
! variables: both are module variables of other modules (as ICON pairs its
! state with the shadows of mo_ad_shadows). The forward derivative of x * g
! along the seeded shadow g_d = 1 is x = 3; without the pairing it is 0.
!
! REQUIRES: flang_directives, flangenzyme
! The modules, the registrations and the program are separate units, linked
! with full LTO and Enzyme in the link.
! RUN: rm -rf %t && mkdir -p %t
! RUN: %fc %flangDirectives -cpp -DMODS -O2 -flto=full -module-dir %t -c %s -o %t/mods.o
! RUN: %fc -fc1 %flangFc1Directives -cpp -DREG -emit-fir -I%t -module-dir %t %s -o - | FileCheck %s --check-prefix=FIR
! RUN: %fc %flangDirectives -cpp -DREG -O2 -flto=full -I%t -module-dir %t -c %s -o %t/reg.o
! RUN: %fc %flangDirectives -cpp -O2 -flto=full -I%t -c %s -o %t/main.o
! RUN: %fc -O2 %lldEnzyme -Wl,-mllvm=-enzyme-global-activity %t/main.o %t/mods.o %t/reg.o -o %t/a && %t/a | FileCheck %s

! The unit of the registrations only declares the variables.
! FIR-DAG: fir.global @_QMstateEg {fir.directives = [{args = {shadow = @_QMshadowsEg_d}, keyword = "shadow", prefix = "enzyme"}]} : f32{{$}}
! FIR-DAG: fir.global @_QMshadowsEg_d : f32{{$}}
! FIR-DAG: fir.global weak @_QMstateEg.__enzyme_shadow_global

#ifdef MODS
module state
  implicit none
  real :: g = 2.0
contains
  real function times_g(x)
    real, intent(in) :: x
    times_g = x * g
  end function
end module

module shadows
  implicit none
  real :: g_d = 0.0
end module
#elif defined(REG)
module registrations
  use state, only: g
  use shadows, only: g_d
  implicit none
  !dir$ enzyme shadow(g, shadow=g_d)
end module
#else
program main
  use state, only: times_g
  use shadows, only: g_d
  implicit none
  real :: x, dx
  real, external :: f__enzyme_fwddiff
  x = 3.0
  dx = 0.0
  g_d = 1.0
  print '(F6.4)', f__enzyme_fwddiff(times_g, x, dx)
end program
#endif

! CHECK: 3.0000
