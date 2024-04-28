// RUN: %eopt --enzyme %s -allow-unregistered-dialect | FileCheck %s

module {
  func.func @inactive(%x : f64) -> f64 {
    // We don't have an interface implementation for "foo",
    // but we can see it's inactive from its lack of operands
    // and results.
    "test.foo"() : () -> ()
    return %x : f64
  }
  func.func @diff(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @inactive(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> (f64)
    return %r : f64
  }
}

// Just check that we didn't trigger the error on there not being an interface
// implementation.
// CHECK-LABEL: func private @fwddiffeinactive
// CHECK: "test.foo"()
