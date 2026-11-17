// RUN: %eopt --cir-raise-enzyme-calls --split-input-file --verify-diagnostics %s

!s32i = !cir.int<s, 32>
!void = !cir.void
module {
  cir.global "private" external @enzyme_bogus : !s32i
  cir.func @sq(%x: !cir.double) -> !cir.double {
    cir.return %x : !cir.double
  }
  cir.func private @__enzyme_autodiff(!cir.ptr<!void>, ...) -> !cir.double
  cir.func @d(%x: !cir.double) -> !cir.double {
    %f = cir.get_global @sq : !cir.ptr<!cir.func<(!cir.double) -> !cir.double>>
    %fp = cir.cast bitcast %f : !cir.ptr<!cir.func<(!cir.double) -> !cir.double>> -> !cir.ptr<!void>
    %mg = cir.get_global @enzyme_bogus : !cir.ptr<!s32i>
    %m = cir.load %mg : !cir.ptr<!s32i>, !s32i
    // expected-error @below {{unknown Enzyme marker enzyme_bogus}}
    %r = cir.call @__enzyme_autodiff(%fp, %m, %x) : (!cir.ptr<!void>, !s32i, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// -----

!s32i = !cir.int<s, 32>
!void = !cir.void
module {
  cir.global "private" external @enzyme_width : !s32i
  cir.func @sq(%x: !cir.double) -> !cir.double {
    cir.return %x : !cir.double
  }
  cir.func private @__enzyme_autodiff(!cir.ptr<!void>, ...) -> !cir.double
  cir.func @d(%x: !cir.double, %w: !s32i) -> !cir.double {
    %f = cir.get_global @sq : !cir.ptr<!cir.func<(!cir.double) -> !cir.double>>
    %fp = cir.cast bitcast %f : !cir.ptr<!cir.func<(!cir.double) -> !cir.double>> -> !cir.ptr<!void>
    %mg = cir.get_global @enzyme_width : !cir.ptr<!s32i>
    %m = cir.load %mg : !cir.ptr<!s32i>, !s32i
    // expected-error @below {{enzyme_width is not supported on ClangIR yet}}
    %r = cir.call @__enzyme_autodiff(%fp, %m, %w, %x) : (!cir.ptr<!void>, !s32i, !s32i, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// -----

!void = !cir.void
module {
  cir.func @mul(%x: !cir.double, %y: !cir.double) -> !cir.double {
    cir.return %x : !cir.double
  }
  cir.func private @__enzyme_autodiff(!cir.ptr<!void>, ...) -> !cir.double
  cir.func @d(%x: !cir.double) -> !cir.double {
    %f = cir.get_global @mul : !cir.ptr<!cir.func<(!cir.double, !cir.double) -> !cir.double>>
    %fp = cir.cast bitcast %f : !cir.ptr<!cir.func<(!cir.double, !cir.double) -> !cir.double>> -> !cir.ptr<!void>
    // expected-error @below {{too few arguments for @mul, which takes 2}}
    %r = cir.call @__enzyme_autodiff(%fp, %x) : (!cir.ptr<!void>, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// -----

!void = !cir.void
module {
  cir.func @sq(%x: !cir.double) -> !cir.double {
    cir.return %x : !cir.double
  }
  cir.func private @__enzyme_autodiff(!cir.ptr<!void>, ...) -> !cir.double
  cir.func @d(%x: !cir.double) -> !cir.double {
    %f = cir.get_global @sq : !cir.ptr<!cir.func<(!cir.double) -> !cir.double>>
    %fp = cir.cast bitcast %f : !cir.ptr<!cir.func<(!cir.double) -> !cir.double>> -> !cir.ptr<!void>
    // expected-error @below {{too many arguments for @sq, which takes 1}}
    %r = cir.call @__enzyme_autodiff(%fp, %x, %x) : (!cir.ptr<!void>, !cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// -----

!s32i = !cir.int<s, 32>
!void = !cir.void
module {
  cir.global "private" external @enzyme_dup : !s32i
  cir.func @ld(%p: !cir.ptr<!cir.double>) -> !cir.double {
    %v = cir.load %p : !cir.ptr<!cir.double>, !cir.double
    cir.return %v : !cir.double
  }
  cir.func private @__enzyme_autodiff(!cir.ptr<!void>, ...) -> !cir.double
  cir.func @d(%p: !cir.ptr<!cir.double>) -> !cir.double {
    %f = cir.get_global @ld : !cir.ptr<!cir.func<(!cir.ptr<!cir.double>) -> !cir.double>>
    %fp = cir.cast bitcast %f : !cir.ptr<!cir.func<(!cir.ptr<!cir.double>) -> !cir.double>> -> !cir.ptr<!void>
    %mg = cir.get_global @enzyme_dup : !cir.ptr<!s32i>
    %m = cir.load %mg : !cir.ptr<!s32i>, !s32i
    // expected-error @below {{argument 0 needs a shadow after it}}
    %r = cir.call @__enzyme_autodiff(%fp, %m, %p) : (!cir.ptr<!void>, !s32i, !cir.ptr<!cir.double>) -> !cir.double
    cir.return %r : !cir.double
  }
}

// -----

!s32i = !cir.int<s, 32>
!void = !cir.void
module {
  cir.global "private" external @enzyme_out : !s32i
  cir.func @sq(%x: !cir.double) -> !cir.double {
    cir.return %x : !cir.double
  }
  cir.func private @__enzyme_fwddiff(!cir.ptr<!void>, ...) -> !cir.double
  cir.func @f(%x: !cir.double) -> !cir.double {
    %f = cir.get_global @sq : !cir.ptr<!cir.func<(!cir.double) -> !cir.double>>
    %fp = cir.cast bitcast %f : !cir.ptr<!cir.func<(!cir.double) -> !cir.double>> -> !cir.ptr<!void>
    %mg = cir.get_global @enzyme_out : !cir.ptr<!s32i>
    %m = cir.load %mg : !cir.ptr<!s32i>, !s32i
    // expected-error @below {{enzyme_out is not valid in forward mode}}
    %r = cir.call @__enzyme_fwddiff(%fp, %m, %x) : (!cir.ptr<!void>, !s32i, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// -----

!void = !cir.void
module {
  cir.func private @ext(!cir.double) -> !cir.double
  cir.func private @__enzyme_autodiff(!cir.ptr<!void>, ...) -> !cir.double
  cir.func @d(%x: !cir.double) -> !cir.double {
    %f = cir.get_global @ext : !cir.ptr<!cir.func<(!cir.double) -> !cir.double>>
    %fp = cir.cast bitcast %f : !cir.ptr<!cir.func<(!cir.double) -> !cir.double>> -> !cir.ptr<!void>
    // expected-error @below {{cannot differentiate @ext: it has no body in this module}}
    %r = cir.call @__enzyme_autodiff(%fp, %x) : (!cir.ptr<!void>, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// -----

!void = !cir.void
module {
  cir.func private @__enzyme_autodiff(!cir.ptr<!void>, ...) -> !cir.double
  cir.func @d(%fp: !cir.ptr<!void>, %x: !cir.double) -> !cir.double {
    // expected-error @below {{the first argument of __enzyme_autodiff must be the address of a function defined in this module}}
    %r = cir.call @__enzyme_autodiff(%fp, %x) : (!cir.ptr<!void>, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}
