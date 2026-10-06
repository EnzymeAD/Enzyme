! !DIR$ ENZYME directives, defined by the FlangEnzymeMLIR plugin through
! flang's plugin directives: flang resolves their names to symbols (here
! procedures defined later under CONTAINS), lowers them to `fir.directives`
! on their subject, and writes them into the module file.
!
! REQUIRES: flang_directives
! RUN: rm -rf %t && mkdir -p %t
! RUN: %fc -fc1 %flangFc1Directives -emit-fir -module-dir %t %s -o - | FileCheck %s
! RUN: FileCheck %s --check-prefix=MOD < %t/rules.mod

module rules
  implicit none
  public
  real :: g, g_d
  !dir$ enzyme custom_rule(f, augmented=f_aug, reverse=f_rev)
  !dir$ enzyme shadow(g, shadow=g_d)
contains
  subroutine f(x, y)
    real, intent(in) :: x
    real, intent(out) :: y
    y = 2.0 * x
  end subroutine f
  subroutine f_aug(x, dx, y, dy)
    real, intent(in) :: x, dx
    real, intent(out) :: y
    real, intent(inout) :: dy
    call f(x, y)
  end subroutine f_aug
  subroutine f_rev(x, dx, y, dy)
    real, intent(in) :: x, y
    real, intent(inout) :: dx, dy
    dx = dx + dy
    dy = 0.0
  end subroutine f_rev
  real function timer()
    !dir$ enzyme inactive
    !dir$ enzyme no_escaping_allocation
    timer = 0.0
  end function timer
  subroutine uses_common()
    real :: a, a_d
    ! allow(common-block)
    common /blk/ a
    ! allow(common-block)
    common /blk_d/ a_d
    !dir$ enzyme shadow(/blk/, shadow=/blk_d/)
    a = 1.0
  end subroutine uses_common
end module rules

! The plugin's pass turns them into the markers Enzyme reads.
! CHECK-DAG: fir.global @_QMrulesEg {fir.directives = [{args = {shadow = @_QMrulesEg_d}, keyword = "shadow", prefix = "enzyme"}]} : f32
! CHECK-DAG: fir.global weak @__enzyme_shadow_global._QMrulesEg : tuple<!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>>
! CHECK-DAG: func.func @_QMrulesPf({{.*}}fir.directives = [{args = {augmented = @_QMrulesPf_aug, reverse = @_QMrulesPf_rev}, keyword = "custom_rule", prefix = "enzyme"}]
! CHECK-DAG: fir.global weak @__enzyme_register_gradient._QMrulesPf : tuple<!fir.boxproc<() -> ()>, !fir.boxproc<() -> ()>, !fir.boxproc<() -> ()>>
! CHECK-DAG: func.func @_QMrulesPtimer(){{.*}}fir.directives = [{args = {}, keyword = "inactive", prefix = "enzyme"}, {args = {}, keyword = "no_escaping_allocation", prefix = "enzyme"}]{{.*}}llvm.passthrough = ["enzyme_inactive", "noinline"]
! CHECK-DAG: fir.global weak @__enzyme_inactivefn._QMrulesPtimer
! CHECK-DAG: fir.global weak @__enzyme_nofree._QMrulesPtimer
! CHECK-DAG: fir.global weak @__enzyme_no_escaping_allocation._QMrulesPtimer
! CHECK-DAG: fir.global common @blk_({{.*}}fir.directives = [{args = {shadow = @blk_d_}, keyword = "shadow", prefix = "enzyme"}]
! CHECK-DAG: fir.global weak @__enzyme_shadow_global.blk_

! The module file keeps them, naming their subjects, for the units using it.
! MOD-DAG: !dir$ enzyme custom_rule(f, augmented=f_aug, reverse=f_rev)
! MOD-DAG: !dir$ enzyme shadow(g, shadow=g_d)
! MOD-DAG: !dir$ enzyme inactive(timer)
! MOD-DAG: !dir$ enzyme no_escaping_allocation(timer)
