! REQUIRES: flang
! RUN: %fc -fc1 -load %loadFlangEnzyme -plugin enzyme %s 2>&1 | FileCheck %s

! This exercises the FlangEnzyme frontend plugin sketch
! (Enzyme/Flang/EnzymeFlang.cpp). Loaded with `-fc1 -load ... -plugin enzyme`,
! the plugin walks the parse tree and reports every call to an Enzyme
! differentiation hook it finds. It does not perform differentiation here (that
! is done via the `-fpass-plugin` role of the same shared object); this only
! checks that the frontend plugin recognizes Enzyme usage.

subroutine diff_square(x, dx)
    implicit none
    ! Implicit-interface Enzyme hook (declared, not defined here). We use the
    ! `enzyme_autodiff` spelling since Fortran identifiers may not begin with an
    ! underscore; the plugin matches this the same way it would `__enzyme_autodiff`.
    external :: enzyme_autodiff
    real :: x, dx

    call enzyme_autodiff(x, dx)
end subroutine diff_square

! CHECK: enzyme: detected Enzyme differentiation call to 'enzyme_autodiff'
! CHECK: enzyme: 1 Enzyme differentiation call(s) found
