! A directive whose subject is a generic interface applies to each of its
! specific procedures, also private ones, which a directive in another module
! cannot name. The unit with the directive declares them, by their external
! names, and registers each. A directive with procedure arguments
! (custom_rule) must name a specific procedure instead.
!
! REQUIRES: flang_directives
! RUN: rm -rf %t && mkdir -p %t
! RUN: %fc -fc1 %flangFc1Directives -cpp -DPART=1 -emit-fir \
! RUN:   -module-dir %t %s -o %t/lib.fir
! RUN: FileCheck %s --check-prefix=LIB < %t/lib.fir
! RUN: FileCheck %s --check-prefix=MOD < %t/gen_lib.mod
! RUN: %fc -fc1 %flangFc1Directives -cpp -DPART=2 -emit-fir \
! RUN:   -module-dir %t %s -o - | FileCheck %s --check-prefix=USER
! RUN: %fc -fc1 %flangFc1Directives -cpp -DPART=2 -emit-llvm \
! RUN:   -module-dir %t %s -o - | FileCheck %s --check-prefix=USER-LL
! RUN: not %fc -fc1 %flangFc1Directives -cpp -DPART=3 -fsyntax-only \
! RUN:   -module-dir %t %s 2>&1 | FileCheck %s --check-prefix=ERR

#if PART == 1
module gen_lib
  implicit none
  private
  public :: gen, other, rule_rev, rule_aug
  interface gen
    module procedure spec_r, spec_i
  end interface
  interface other
    module procedure other_a, other_b
  end interface
  ! In the module that defines the generic, too.
  !$enzyme no_escaping_allocation(other)
contains
  subroutine spec_r(x)
    real, intent(inout) :: x
    x = 2.0 * x
  end subroutine
  subroutine spec_i(n)
    integer, intent(inout) :: n
    n = n + 1
  end subroutine
  subroutine other_a(x)
    real, intent(inout) :: x
    x = x + 1.0
  end subroutine
  subroutine other_b(n)
    integer, intent(inout) :: n
    n = n - 1
  end subroutine
  subroutine rule_aug(x, dx)
    real :: x, dx
  end subroutine
  subroutine rule_rev(x, dx)
    real :: x, dx
  end subroutine
end module

! LIB-DAG: func.func @_QMgen_libPother_a({{.*}}fir.directives = [{args = {}, keyword = "no_escaping_allocation", prefix = "enzyme"}]
! LIB-DAG: func.func @_QMgen_libPother_b({{.*}}fir.directives = [{args = {}, keyword = "no_escaping_allocation", prefix = "enzyme"}]
! LIB-DAG: fir.global weak @__enzyme_no_escaping_allocation._QMgen_libPother_a
! LIB-DAG: fir.global weak @__enzyme_no_escaping_allocation._QMgen_libPother_b
! The module file names the specific procedures.
! MOD-DAG: !dir$ enzyme no_escaping_allocation(other_a)
! MOD-DAG: !dir$ enzyme no_escaping_allocation(other_b)
#endif

#if PART == 2
module gen_user
  use gen_lib, only: gen
  implicit none
  !$enzyme inactive(gen)
contains
  subroutine work(x)
    real, intent(inout) :: x
    x = x * x
  end subroutine
end module

! USER-DAG: func.func private @_QMgen_libPspec_r({{.*}}fir.directives = [{args = {}, keyword = "inactive", prefix = "enzyme"}]
! USER-DAG: func.func private @_QMgen_libPspec_i({{.*}}fir.directives = [{args = {}, keyword = "inactive", prefix = "enzyme"}]
! USER-DAG: fir.global weak @__enzyme_inactivefn._QMgen_libPspec_r
! USER-DAG: fir.global weak @__enzyme_inactivefn._QMgen_libPspec_i
! USER-DAG: fir.global weak @__enzyme_nofree._QMgen_libPspec_r
! USER-DAG: fir.global weak @__enzyme_nofree._QMgen_libPspec_i

! The private specific procedures are external symbols of gen_lib.
! USER-LL-DAG: @__enzyme_inactivefn._QMgen_libPspec_r = weak global { ptr } { ptr @_QMgen_libPspec_r }
! USER-LL-DAG: @__enzyme_inactivefn._QMgen_libPspec_i = weak global { ptr } { ptr @_QMgen_libPspec_i }
! USER-LL-DAG: declare void @_QMgen_libPspec_r(ptr {{.*}}) #[[ATTR:[0-9]+]]
! USER-LL-DAG: declare void @_QMgen_libPspec_i(ptr {{.*}})
#endif

#if PART == 3
module gen_errors
  use gen_lib, only: gen, rule_aug, rule_rev
  implicit none
  ! ERR: error: 'gen' is a generic interface; a 'enzyme custom_rule' directive must name one of its specific procedures
  !$enzyme custom_rule(gen, augmented=rule_aug, reverse=rule_rev)
contains
  subroutine f(x, dx)
    real :: x, dx
  end subroutine
  ! ERR: error: 'gen' is a generic interface; argument 'reverse' of a 'enzyme custom_rule' directive must name a specific procedure
  subroutine g(x)
    real :: x
    !$enzyme custom_rule(augmented=rule_aug, reverse=gen)
  end subroutine
end module
#endif
