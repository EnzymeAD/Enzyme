// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

module {
  llvm.func @square(%x: f64) -> f64 {
    %next = arith.mulf %x, %x : f64
    llvm.return %next : f64
  }

  func.func @dsquare(%x: f64, %dr: f64) -> f64 {
    %dx = enzyme.autodiff @square(%x, %dr)
      {
        activity=[#enzyme<activity enzyme_active>],
        ret_activity=[#enzyme<activity enzyme_activenoneed>]
      } : (f64, f64) -> f64
    return %dx : f64
  }

// CHECK: llvm.func @diffesquare(%arg0: f64, %arg1: f64) -> f64 attributes {sym_visibility = "private"} {
// CHECK-NEXT:    %0 = arith.mulf %arg1, %arg0 : f64
// CHECK-NEXT:    %1 = arith.mulf %arg1, %arg0 : f64
// CHECK-NEXT:    %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:    llvm.return %2 : f64
// CHECK-NEXT:  }
}

// -----

llvm.func @multireturn(%x: f64, %y: f64) -> f64 {
  %0 = arith.mulf %x, %y : f64
  llvm.return %0 : f64
}

func.func @dmultireturn(%x: f64, %y: f64, %dr: f64) -> (f64, f64) {
  %res = enzyme.autodiff @multireturn(%x, %y, %dr)
    {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64, f64) -> !llvm.struct<(f64, f64)>
  %fst = llvm.extractvalue %res[0] : !llvm.struct<(f64, f64)>
  %snd = llvm.extractvalue %res[1] : !llvm.struct<(f64, f64)>
  return %fst, %snd : f64, f64
}

// CHECK: llvm.func @diffemultireturn(%arg0: f64, %arg1: f64, %arg2: f64) -> !llvm.struct<(f64, f64)> attributes {sym_visibility = "private"} {
// CHECK-NEXT:    %0 = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-NEXT:    %1 = arith.mulf %arg2, %arg1 : f64
// CHECK-NEXT:    %2 = arith.mulf %arg2, %arg0 : f64
// CHECK-NEXT:    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:    llvm.return %4 : !llvm.struct<(f64, f64)>
// CHECK-NEXT:  }
