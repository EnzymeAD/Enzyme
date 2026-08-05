// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

// What enzyme.fwddiff names is not always a func.func. Anything raised from
// LLVM gives an llvm.func, and forward mode turned that down twice over: the
// verifier looked the callee up as a func.func, and the pass wrote a func.call
// to whatever came back. enzyme.autodiff has always taken either.

module {
  llvm.func @square(%x: f64) -> f64 {
    %r = arith.mulf %x, %x : f64
    llvm.return %r : f64
  }

  llvm.func @dsquare(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> f64
    llvm.return %r : f64
  }
}

// The derivative is an llvm.func too, and is called as one.

// CHECK:       llvm.func @dsquare(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64
// CHECK:         %[[c:.+]] = llvm.call @fwddiffesquare(%[[x]], %[[dx]]) : (f64, f64) -> f64
// CHECK:         llvm.return %[[c]] : f64

// CHECK:       llvm.func @fwddiffesquare(%[[px:.+]]: f64, %[[pdx:.+]]: f64) -> f64
// CHECK:         %[[a:.+]] = arith.mulf %[[pdx]], %[[px]] fastmath<fast> : f64
// CHECK:         %[[b:.+]] = arith.mulf %[[pdx]], %[[px]] fastmath<fast> : f64
// CHECK:         %[[s:.+]] = arith.addf %[[a]], %[[b]] fastmath<fast> : f64
// CHECK:         llvm.return %[[s]] : f64
