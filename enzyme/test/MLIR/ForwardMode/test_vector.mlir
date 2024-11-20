// RUN: %eopt --enzyme %s | FileCheck %s

// TODO: actually test

module {
  func.func @square(%x : f64) -> f64{
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @dsq(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>], width=2 } : (f64, f64) -> (f64)
    return %r : f64
  }
}
