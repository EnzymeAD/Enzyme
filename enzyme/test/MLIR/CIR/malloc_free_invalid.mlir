// RUN: %eopt --enzyme --verify-diagnostics %s

// In reverse mode the adjoint of the copy realloc makes needs the size of the
// old allocation, which realloc is not given: it has to come from a malloc,
// calloc, realloc or null in this function, and a pointer argument does not.

!u64i = !cir.int<u, 64>
!void = !cir.void
module {
  cir.func private @realloc(!cir.ptr<!void>, !u64i) -> !cir.ptr<!void>
  cir.func private @free(!cir.ptr<!void>)
  cir.func @regrow(%p: !cir.ptr<!cir.double>, %x: !cir.double) -> !cir.double {
    %n = cir.const #cir.int<16> : !u64i
    %m = cir.cast bitcast %p : !cir.ptr<!cir.double> -> !cir.ptr<!void>
    // expected-error @below {{cannot differentiate a realloc in reverse mode unless what it resizes comes from malloc, calloc, realloc or null in this function}}
    %r = cir.call @realloc(%m, %n) : (!cir.ptr<!void>, !u64i) -> !cir.ptr<!void>
    %t = cir.cast bitcast %r : !cir.ptr<!void> -> !cir.ptr<!cir.double>
    %sq = cir.fmul %x, %x : !cir.double
    cir.store %sq, %t : !cir.double, !cir.ptr<!cir.double>
    %y = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    cir.call @free(%r) : (!cir.ptr<!void>) -> ()
    cir.return %y : !cir.double
  }
  cir.func @rev(%p: !cir.ptr<!cir.double>, %dp: !cir.ptr<!cir.double>, %x: !cir.double, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @regrow(%p, %dp, %x, %seed) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}
