! The !$enzyme sentinel in fixed form (-ffixed-form): c$enzyme, *$enzyme or !$enzyme from
! column 1. The sentinel is longer than columns 2-5, so the column after it
! (9) takes the place of column 6: blank on an initial line, the
! continuation mark on a continuation line. The directives lower and are
! written to the module file as their !DIR$ ENZYME spelling.
!
! REQUIRES: flang_directives
! RUN: rm -rf %t && mkdir -p %t/dir %t/sentinel %t/sentinel-omp
! RUN: %fc -fc1 %flangFc1Directives -ffixed-form -cpp -emit-fir \
! RUN:   -module-dir %t/dir %s -o %t/dir.fir
! RUN: %fc -fc1 %flangFc1Directives -ffixed-form -cpp -DSENTINEL -emit-fir \
! RUN:   -module-dir %t/sentinel %s -o %t/sentinel.fir
! RUN: FileCheck %s < %t/dir.fir
! RUN: FileCheck %s < %t/sentinel.fir
! RUN: FileCheck %s --check-prefix=NOOMP < %t/sentinel.fir
! RUN: diff %t/dir.fir %t/sentinel.fir
! RUN: diff %t/dir/fixed_rules.mod %t/sentinel/fixed_rules.mod
! RUN: %fc -fc1 %flangFc1Directives -ffixed-form -cpp -DSENTINEL -fopenmp \
! RUN:   -emit-fir -module-dir %t/sentinel-omp %s -o %t/sentinel-omp.fir
! RUN: FileCheck %s < %t/sentinel-omp.fir
! RUN: FileCheck %s --check-prefix=OMP < %t/sentinel-omp.fir
! RUN: %fc -fc1 -ffixed-form -cpp -DSENTINEL -fsyntax-only -module-dir %t \
! RUN:   %s 2>&1 | FileCheck %s --check-prefix=NOPLUGIN --allow-empty

      module fixed_rules
      implicit none
      real g, g_d
      real a, a_d
      common /blk/ a
      common /blk_d/ a_d
#ifdef SENTINEL
c$enzyme shadow(g, shadow=g_d)
*$ENZYME shadow(/blk/,
c$enzyme+ shadow=/blk_d/)
!$enzyme custom_rule(f, augmented=f_aug,
!$enzyme1 reverse=f_rev)
#else
!dir$ enzyme shadow(g, shadow=g_d)
!DIR$ ENZYME shadow(/blk/,
!dir$+ shadow=/blk_d/)
!dir$ enzyme custom_rule(f, augmented=f_aug,
!dir$1 reverse=f_rev)
#endif
      contains
      subroutine f(x, y)
      real x, y
      y = 2.0 * x
      end subroutine
      subroutine f_aug(x, dx, y, dy)
      real x, dx, y, dy
      call f(x, y)
      end subroutine
      subroutine f_rev(x, dx, y, dy)
      real x, dx, y, dy
      dx = dx + 2.0 * dy
      dy = 0.0
      end subroutine
      subroutine scale(x)
      real x
c$    x = 3.0 * x
      x = a * x
      end subroutine
      end module

! CHECK-DAG: fir.global @_QMfixed_rulesEg {fir.directives = [{args = {shadow = @_QMfixed_rulesEg_d}, keyword = "shadow", prefix = "enzyme"}]} : f32
! CHECK-DAG: fir.global weak @__enzyme_shadow_global._QMfixed_rulesEg
! CHECK-DAG: fir.global common @blk_({{.*}}fir.directives = [{args = {shadow = @blk_d_}, keyword = "shadow", prefix = "enzyme"}]
! CHECK-DAG: fir.global weak @__enzyme_shadow_global.blk_
! CHECK-DAG: func.func @_QMfixed_rulesPf({{.*}}fir.directives = [{args = {augmented = @_QMfixed_rulesPf_aug, reverse = @_QMfixed_rulesPf_rev}, keyword = "custom_rule", prefix = "enzyme"}]
! CHECK-DAG: fir.global weak @__enzyme_register_gradient._QMfixed_rulesPf

! OMP-LABEL: func.func @_QMfixed_rulesPscale(
! OMP: arith.constant 3.000000e+00 : f32
! NOOMP-LABEL: func.func @_QMfixed_rulesPscale(
! NOOMP-NOT: arith.constant 3.000000e+00 : f32
! NOOMP: return

! NOPLUGIN-NOT: {{warning|error}}
