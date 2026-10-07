! The !$enzyme sentinel is the same directive as !DIR$ ENZYME: the module
! below, with its directives in either spelling (SENTINEL or not), lowers to
! the same FIR and registrations and writes the same module file. Compilers
! without the plugin read !$enzyme lines as comments. !$ followed by a blank
! remains OpenMP conditional compilation.
!
! REQUIRES: flang_directives
! RUN: rm -rf %t && mkdir -p %t/dir %t/sentinel %t/sentinel-omp
! RUN: %fc -fc1 %flangFc1Directives -cpp -emit-fir -module-dir %t/dir %s -o %t/dir.fir
! RUN: %fc -fc1 %flangFc1Directives -cpp -DSENTINEL -emit-fir -module-dir %t/sentinel %s -o %t/sentinel.fir
! RUN: FileCheck %s < %t/dir.fir
! RUN: FileCheck %s < %t/sentinel.fir
! RUN: diff %t/dir.fir %t/sentinel.fir
! RUN: diff %t/dir/sentinel_rules.mod %t/sentinel/sentinel_rules.mod
! RUN: FileCheck %s --check-prefix=MOD < %t/sentinel/sentinel_rules.mod
!
! With -fopenmp, !$enzyme is still the directive, and !$ x = ... is code.
! RUN: %fc -fc1 %flangFc1Directives -cpp -DSENTINEL -fopenmp -emit-fir -module-dir %t/sentinel-omp %s -o %t/sentinel-omp.fir
! RUN: FileCheck %s < %t/sentinel-omp.fir
! RUN: FileCheck %s --check-prefix=OMP < %t/sentinel-omp.fir
! RUN: FileCheck %s --check-prefix=NOOMP < %t/sentinel.fir
!
! Without the plugin: !$enzyme is a comment, !DIR$ ENZYME is not known.
! RUN: %fc -fc1 -cpp -DSENTINEL -fsyntax-only -module-dir %t %s 2>&1 | FileCheck %s --check-prefix=NOPLUGIN --allow-empty
! RUN: %fc -fc1 -cpp -fsyntax-only -module-dir %t %s 2>&1 | FileCheck %s --check-prefix=NOPLUGIN-DIR

module sentinel_rules
  implicit none
  public
  real :: g, g_d
  real :: a, a_d
  ! allow(common-block)
  common /blk/ a
  ! allow(common-block)
  common /blk_d/ a_d
  real :: c
  ! allow(common-block)
  common /consts/ c
#ifdef SENTINEL
  !$enzyme custom_rule(f, augmented=f_aug, reverse=f_rev)
  !$enzyme shadow(g, shadow=g_d)
  !$ENZYME shadow(/blk/, &
  !$enzyme&  shadow=/blk_d/)
  !$enzyme inactive(/consts/)
#else
  !dir$ enzyme custom_rule(f, augmented=f_aug, reverse=f_rev)
  !dir$ enzyme shadow(g, shadow=g_d)
  !DIR$ ENZYME shadow(/blk/, &
  !dir$&  shadow=/blk_d/)
  !dir$ enzyme inactive(/consts/)
#endif
contains
  subroutine f(x, y)
    real, intent(in) :: x
    real, intent(out) :: y
    y = 2.0 * x * c
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
    dx = dx + 2.0 * c * dy
    dy = 0.0
  end subroutine f_rev
  real function timer()
#ifdef SENTINEL
    !$enzyme inactive
#else
    !dir$ enzyme inactive
#endif
    timer = 0.0
  end function timer
  subroutine scale(x)
    real, intent(inout) :: x
!$ x = 3.0 * x
    x = a * x
  end subroutine scale
end module sentinel_rules

! CHECK-DAG: fir.global @_QMsentinel_rulesEg {fir.directives = [{args = {shadow = @_QMsentinel_rulesEg_d}, keyword = "shadow", prefix = "enzyme"}]} : f32
! CHECK-DAG: fir.global weak @__enzyme_shadow_global._QMsentinel_rulesEg
! CHECK-DAG: func.func @_QMsentinel_rulesPf({{.*}}fir.directives = [{args = {augmented = @_QMsentinel_rulesPf_aug, reverse = @_QMsentinel_rulesPf_rev}, keyword = "custom_rule", prefix = "enzyme"}]
! CHECK-DAG: fir.global weak @__enzyme_register_gradient._QMsentinel_rulesPf
! CHECK-DAG: fir.global common @blk_({{.*}}fir.directives = [{args = {shadow = @blk_d_}, keyword = "shadow", prefix = "enzyme"}]
! CHECK-DAG: fir.global weak @__enzyme_shadow_global.blk_
! CHECK-DAG: fir.global common @consts_({{.*}}fir.directives = [{args = {}, keyword = "inactive", prefix = "enzyme"}]
! CHECK-DAG: fir.global weak @__enzyme_inactive_global.consts_
! CHECK-DAG: func.func @_QMsentinel_rulesPtimer(){{.*}}fir.directives = [{args = {}, keyword = "inactive", prefix = "enzyme"}]
! CHECK-DAG: fir.global weak @__enzyme_inactivefn._QMsentinel_rulesPtimer

! OMP-LABEL: func.func @_QMsentinel_rulesPscale(
! OMP: arith.constant 3.000000e+00 : f32
! NOOMP-LABEL: func.func @_QMsentinel_rulesPscale(
! NOOMP-NOT: arith.constant 3.000000e+00 : f32
! NOOMP: return

! The module file has the canonical spelling.
! MOD-DAG: !dir$ enzyme custom_rule(f, augmented=f_aug, reverse=f_rev)
! MOD-DAG: !dir$ enzyme shadow(g, shadow=g_d)
! MOD-DAG: !dir$ enzyme shadow(/blk/, shadow=/blk_d/)
! MOD-DAG: !dir$ enzyme inactive(/consts/)
! MOD-DAG: !dir$ enzyme inactive(timer)

! NOPLUGIN-NOT: {{warning|error}}
! NOPLUGIN-DIR: warning: Unrecognized compiler directive was ignored
