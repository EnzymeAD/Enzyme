// RUN: %eopt --enzyme --split-input-file --verify-diagnostics %s

// Storing an active value into a global needs somewhere to keep its tangent,
// and a global defined in another module cannot get one here.
module {
  cir.global "private" external @g : !cir.double
  cir.func @write(%x: !cir.double) -> !cir.double {
    // expected-error @+1 {{global @g is active and defined outside this module, so it needs a shadow global named by an enzyme_shadow attribute}}
    %0 = cir.get_global @g : !cir.ptr<!cir.double>
    cir.store %x, %0 : !cir.double, !cir.ptr<!cir.double>
    cir.return %x : !cir.double
  }
  cir.func @dwrite(%x: !cir.double, %dx: !cir.double) -> !cir.double {
    %r = enzyme.fwddiff @write(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// -----

!s32i = !cir.int<s, 32>
module {
  cir.global external @g = #cir.fp<3.000000e+00> : !cir.double {enzyme_shadow = @g_shadow}
  cir.global external @g_shadow = #cir.int<0> : !s32i
  cir.func @write(%x: !cir.double) -> !cir.double {
    // expected-error @+1 {{enzyme_shadow of @g must name a cir.global of type '!cir.double'}}
    %0 = cir.get_global @g : !cir.ptr<!cir.double>
    cir.store %x, %0 : !cir.double, !cir.ptr<!cir.double>
    cir.return %x : !cir.double
  }
  cir.func @dwrite(%x: !cir.double, %dx: !cir.double) -> !cir.double {
    %r = enzyme.fwddiff @write(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// -----

// A public global may be written with an active value outside this module, so
// even reading it needs a shadow.
module {
  cir.global "private" external @g : !cir.double
  cir.func @read(%x: !cir.double) -> !cir.double {
    // expected-error @+1 {{global @g is active and defined outside this module, so it needs a shadow global named by an enzyme_shadow attribute}}
    %0 = cir.get_global @g : !cir.ptr<!cir.double>
    %1 = cir.load %0 : !cir.ptr<!cir.double>, !cir.double
    %2 = cir.fmul %x, %1 : !cir.double
    cir.return %2 : !cir.double
  }
  cir.func @dread(%x: !cir.double, %dr: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @read(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}
