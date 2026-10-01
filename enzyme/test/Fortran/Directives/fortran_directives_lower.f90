! !DIR$ ENZYME directives, defined by the FlangEnzymeMLIR plugin through
! flang's plugin directives: flang resolves their names to symbols (here
! procedures defined later under CONTAINS), lowers them to `fir.directives`
! on their subject, and writes them into the module file.
!
! REQUIRES: flang_directives
! RUN: rm -rf %t && mkdir -p %t
! RUN: %fc -fc1 %flangFc1Directives -emit-fir \
! RUN:   -module-dir %t %s -o - | FileCheck %s
! RUN: FileCheck %s --check-prefix=MOD < %t/rules.mod

module rules
  implicit none
  real :: g, g_d
  !dir$ enzyme custom_rule(f, augmented=f_aug, reverse=f_rev)
  !dir$ enzyme shadow(g, shadow=g_d)
contains
  subroutine f(x, y)
    real, intent(in) :: x
    real, intent(out) :: y
    y = 2.0 * x
  end subroutine
  subroutine f_aug(x, dx, y, dy)
    real, intent(in) :: x, dx
    real, intent(out) :: y
    real, intent(inout) :: dy
    call f(x, y)
  end subroutine
  subroutine f_rev(x, dx, y, dy)
    real, intent(in) :: x, y
    real, intent(inout) :: dx, dy
    dx = dx + dy
    dy = 0.0
  end subroutine
  real function timer()
    !dir$ enzyme inactive
    timer = 0.0
  end function
  subroutine uses_common()
    real :: a, a_d
    common /blk/ a
    common /blk_d/ a_d
    !dir$ enzyme shadow(/blk/, shadow=/blk_d/)
    a = 1.0
  end subroutine
end module

! The plugin's pass turns them into the markers Enzyme reads.
! CHECK-DAG: fir.global @_QMrulesEg {fir.directives = [{args = {shadow = @_QMrulesEg_d}, keyword = "shadow", prefix = "enzyme"}]} : f32
! CHECK-DAG: fir.global weak @__enzyme_shadow_global._QMrulesEg : tuple<!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>>
! CHECK-DAG: func.func @_QMrulesPf({{.*}}fir.directives = [{args = {augmented = @_QMrulesPf_aug, reverse = @_QMrulesPf_rev}, keyword = "custom_rule", prefix = "enzyme"}]
! CHECK-DAG: fir.global weak @__enzyme_register_gradient._QMrulesPf : tuple<!fir.boxproc<() -> ()>, !fir.boxproc<() -> ()>, !fir.boxproc<() -> ()>>
! CHECK-DAG: func.func @_QMrulesPtimer(){{.*}}fir.directives = [{args = {}, keyword = "inactive", prefix = "enzyme"}]{{.*}}llvm.passthrough = ["enzyme_inactive", "noinline"]
! CHECK-DAG: fir.global weak @__enzyme_inactivefn._QMrulesPtimer
! CHECK-DAG: fir.global weak @__enzyme_nofree._QMrulesPtimer
! CHECK-DAG: fir.global common @blk_({{.*}}fir.directives = [{args = {shadow = @blk_d_}, keyword = "shadow", prefix = "enzyme"}]
! CHECK-DAG: fir.global weak @__enzyme_shadow_global.blk_

! The module file keeps them, naming their subjects, for the units using it.
! MOD-DAG: !dir$ enzyme custom_rule(f, augmented=f_aug, reverse=f_rev)
! MOD-DAG: !dir$ enzyme shadow(g, shadow=g_d)
! MOD-DAG: !dir$ enzyme inactive(timer)
