// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : !llvm.ptr, %y : !llvm.ptr) {
    %u = llvm.mlir.poison : !llvm.struct<(ptr, ptr)>
    %a0 = llvm.insertvalue %x, %u[0] : !llvm.struct<(ptr, ptr)>
    %a1 = llvm.insertvalue %y, %a0[1] : !llvm.struct<(ptr, ptr)>
    %px = llvm.extractvalue %a1[0] : !llvm.struct<(ptr, ptr)>
    %py = llvm.extractvalue %a1[1] : !llvm.struct<(ptr, ptr)>
    %v = llvm.load %px : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %py : f64, !llvm.ptr
    return
  }

  func.func @dsquare(%x : !llvm.ptr, %dx : !llvm.ptr, %y : !llvm.ptr, %dy : !llvm.ptr) {
    enzyme.fwddiff @square(%x, %dx, %y, %dy) {
      activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
      ret_activity=[]
    } : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK:  func.func private @fwddiffesquare(%[[x:.+]]: !llvm.ptr, %[[dx:.+]]: !llvm.ptr, %[[y:.+]]: !llvm.ptr, %[[dy:.+]]: !llvm.ptr) {
// CHECK:    %[[sa0:.+]] = llvm.insertvalue %[[dx]], %{{.+}}[0] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[pa0:.+]] = llvm.insertvalue %[[x]], %{{.+}}[0] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[sa1:.+]] = llvm.insertvalue %[[dy]], %[[sa0]][1] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[pa1:.+]] = llvm.insertvalue %[[y]], %[[pa0]][1] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[spx:.+]] = llvm.extractvalue %[[sa1]][0] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[ppx:.+]] = llvm.extractvalue %[[pa1]][0] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[spy:.+]] = llvm.extractvalue %[[sa1]][1] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[ppy:.+]] = llvm.extractvalue %[[pa1]][1] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[dv:.+]] = llvm.load %[[spx]] : !llvm.ptr -> f64
// CHECK:    %[[v:.+]] = llvm.load %[[ppx]] : !llvm.ptr -> f64
// CHECK:    %[[l:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK:    %[[r:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK:    %[[ds:.+]] = arith.addf %[[l]], %[[r]] fastmath<fast> : f64
// CHECK:    %[[s:.+]] = arith.mulf %[[v]], %[[v]] : f64
// CHECK:    llvm.store %[[ds]], %[[spy]] : f64, !llvm.ptr
// CHECK:    llvm.store %[[s]], %[[ppy]] : f64, !llvm.ptr
