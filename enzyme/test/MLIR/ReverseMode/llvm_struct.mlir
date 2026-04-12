// RUN: %eopt --outline-enzyme-regions --enzyme %s | FileCheck %s

llvm.func @compute(%arg0: !llvm.ptr, %arg1: f64) -> f64 {
  %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(i64, i64)>
  %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, i64)>
  %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
  %3 = llvm.load %2 : !llvm.ptr -> f64
  %4 = llvm.fmul %3, %arg1 : f64
  llvm.return %4 : f64
}

llvm.func @kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: f64, %arg3: f64) {
  enzyme.autodiff_region(%arg0, %arg1, %arg2, %arg3) {
  ^bb0(%a0: !llvm.ptr, %a1: f64):
    %0 = llvm.load %a0 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, i64)>
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %3 = llvm.load %2 : !llvm.ptr -> f64
    %4 = llvm.fmul %3, %a1 : f64
    enzyme.yield %4 : f64
  } attributes {
    activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>],
    fn = "compute"
  } : (!llvm.ptr, !llvm.ptr, f64, f64) -> f64
  llvm.return
}

// CHECK:      llvm.func @kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: f64, %arg3: f64) {
// CHECK-NEXT:   func.call @diffekernel_to_diff0(%arg0, %arg1, %arg2, %arg3) : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }

// CHECK:      func.func @kernel_to_diff0(%arg0: !llvm.ptr, %arg1: f64) -> f64 {
// CHECK-NEXT:   %[[V0:.+]] = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[V1:.+]] = llvm.extractvalue %[[V0]][0] : !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[V2:.+]] = llvm.inttoptr %[[V1]] : i64 to !llvm.ptr
// CHECK-NEXT:   %[[V3:.+]] = llvm.load %[[V2]] : !llvm.ptr -> f64
// CHECK-NEXT:   %[[V4:.+]] = llvm.fmul %[[V3]], %arg1 : f64
// CHECK-NEXT:   return %[[V4]] : f64
// CHECK-NEXT: }

// CHECK:      func.func private @diffekernel_to_diff0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: f64, %arg3: f64) {
// CHECK-NEXT:   %[[GINIT:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<!llvm.struct<(i64, i64)>>
// CHECK-NEXT:   %[[POISON:.+]] = llvm.mlir.poison : !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[Z0:.+]] = arith.constant 0 : i64
// CHECK-NEXT:   %[[S0:.+]] = llvm.insertvalue %[[Z0]], %[[POISON]][0] : !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[Z1:.+]] = arith.constant 0 : i64
// CHECK-NEXT:   %[[S1:.+]] = llvm.insertvalue %[[Z1]], %[[S0]][1] : !llvm.struct<(i64, i64)>
// CHECK-NEXT:   "enzyme.set"(%[[GINIT]], %[[S1]]) : (!enzyme.Gradient<!llvm.struct<(i64, i64)>>, !llvm.struct<(i64, i64)>) -> ()
// CHECK-NEXT:   %[[CACHE:.+]] = "enzyme.init"() : () -> !enzyme.Cache<!llvm.ptr>
// CHECK-NEXT:   "enzyme.push"(%[[CACHE]], %arg1) : (!enzyme.Cache<!llvm.ptr>, !llvm.ptr) -> ()
// CHECK-NEXT:   %[[LD0:.+]] = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[EV:.+]] = llvm.extractvalue %[[LD0]][0] : !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[ITP:.+]] = llvm.inttoptr %[[EV]] : i64 to !llvm.ptr
// CHECK-NEXT:   %[[LD1:.+]] = llvm.load %[[ITP]] : !llvm.ptr -> f64
// CHECK-NEXT:   %[[MUL:.+]] = llvm.fmul %[[LD1]], %arg2 : f64
// CHECK-NEXT:   cf.br ^bb1
// CHECK:      ^bb1:
// CHECK-NEXT:   %[[GVAL:.+]] = "enzyme.get"(%[[GINIT]]) : (!enzyme.Gradient<!llvm.struct<(i64, i64)>>) -> !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[SPTR:.+]] = "enzyme.pop"(%[[CACHE]]) : (!enzyme.Cache<!llvm.ptr>) -> !llvm.ptr
// CHECK-NEXT:   %[[SLD:.+]] = llvm.load %[[SPTR]] : !llvm.ptr -> !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[P2:.+]] = llvm.mlir.poison : !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[A0:.+]] = llvm.extractvalue %[[SLD]][0] : !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[B0:.+]] = llvm.extractvalue %[[GVAL]][0] : !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[ADD0:.+]] = arith.addi %[[A0]], %[[B0]] : i64
// CHECK-NEXT:   %[[R0:.+]] = llvm.insertvalue %[[ADD0]], %[[P2]][0] : !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[A1:.+]] = llvm.extractvalue %[[SLD]][1] : !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[B1:.+]] = llvm.extractvalue %[[GVAL]][1] : !llvm.struct<(i64, i64)>
// CHECK-NEXT:   %[[ADD1:.+]] = arith.addi %[[A1]], %[[B1]] : i64
// CHECK-NEXT:   %[[R1:.+]] = llvm.insertvalue %[[ADD1]], %[[R0]][1] : !llvm.struct<(i64, i64)>
// CHECK-NEXT:   llvm.store %[[R1]], %[[SPTR]] : !llvm.struct<(i64, i64)>, !llvm.ptr
// CHECK-NEXT:   return
// CHECK-NEXT: }
