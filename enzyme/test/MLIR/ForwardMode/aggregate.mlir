// RUN: %eopt --enzyme --split-input-file %s | FileCheck %s
// RUN: %eopt --pass-pipeline="builtin.module(enzyme{dataflow})" --split-input-file %s | FileCheck %s

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
    enzyme.fwddiff @square(%x, %dx, %y, %dy) <{
      activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity=[]
    }> : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL: func.func private @fwddiffesquare
// CHECK-SAME: (%[[x:.+]]: !llvm.ptr, %[[dx:.+]]: !llvm.ptr, %[[y:.+]]: !llvm.ptr, %[[dy:.+]]: !llvm.ptr) {
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

// -----

module {
  func.func @capture_arg(%cap : !llvm.struct<(ptr, ptr)>) {
    %px = llvm.extractvalue %cap[0] : !llvm.struct<(ptr, ptr)>
    %py = llvm.extractvalue %cap[1] : !llvm.struct<(ptr, ptr)>
    %v = llvm.load %px : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %py : f64, !llvm.ptr
    return
  }

  func.func @dcapture_arg(%cap : !llvm.struct<(ptr, ptr)>, %dcap : !llvm.struct<(ptr, ptr)>) {
    enzyme.fwddiff @capture_arg(%cap, %dcap) <{
      activity=[#enzyme.activity<enzyme_dup>],
      ret_activity=[]
    }> : (!llvm.struct<(ptr, ptr)>, !llvm.struct<(ptr, ptr)>) -> ()
    return
  }
}

// CHECK-LABEL: func.func private @fwddiffecapture_arg
// CHECK-SAME: (%[[cap:.+]]: !llvm.struct<(ptr, ptr)>, %[[dcap:.+]]: !llvm.struct<(ptr, ptr)>) {
// CHECK:    %[[spx:.+]] = llvm.extractvalue %[[dcap]][0] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[ppx:.+]] = llvm.extractvalue %[[cap]][0] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[spy:.+]] = llvm.extractvalue %[[dcap]][1] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[ppy:.+]] = llvm.extractvalue %[[cap]][1] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[dv:.+]] = llvm.load %[[spx]] : !llvm.ptr -> f64
// CHECK:    %[[v:.+]] = llvm.load %[[ppx]] : !llvm.ptr -> f64
// CHECK:    %[[l:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK:    %[[r:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK:    %[[ds:.+]] = arith.addf %[[l]], %[[r]] fastmath<fast> : f64
// CHECK:    %[[s:.+]] = arith.mulf %[[v]], %[[v]] : f64
// CHECK:    llvm.store %[[ds]], %[[spy]] : f64, !llvm.ptr
// CHECK:    llvm.store %[[s]], %[[ppy]] : f64, !llvm.ptr

// -----

module {
  func.func @capture_load(%cap_ptr : !llvm.ptr) {
    %cap = llvm.load %cap_ptr : !llvm.ptr -> !llvm.struct<(ptr, ptr)>
    %px = llvm.extractvalue %cap[0] : !llvm.struct<(ptr, ptr)>
    %py = llvm.extractvalue %cap[1] : !llvm.struct<(ptr, ptr)>
    %v = llvm.load %px : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %py : f64, !llvm.ptr
    return
  }

  func.func @dcapture_load(%cap_ptr : !llvm.ptr, %dcap_ptr : !llvm.ptr) {
    enzyme.fwddiff @capture_load(%cap_ptr, %dcap_ptr) <{
      activity=[#enzyme.activity<enzyme_dup>],
      ret_activity=[]
    }> : (!llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL: func.func private @fwddiffecapture_load
// CHECK-SAME: (%[[cp:.+]]: !llvm.ptr, %[[dcp:.+]]: !llvm.ptr) {
// CHECK:    %[[dcap:.+]] = llvm.load %[[dcp]] : !llvm.ptr -> !llvm.struct<(ptr, ptr)>
// CHECK:    %[[cap:.+]] = llvm.load %[[cp]] : !llvm.ptr -> !llvm.struct<(ptr, ptr)>
// CHECK:    %[[spx:.+]] = llvm.extractvalue %[[dcap]][0] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[ppx:.+]] = llvm.extractvalue %[[cap]][0] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[spy:.+]] = llvm.extractvalue %[[dcap]][1] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[ppy:.+]] = llvm.extractvalue %[[cap]][1] : !llvm.struct<(ptr, ptr)>
// CHECK:    %[[dv:.+]] = llvm.load %[[spx]] : !llvm.ptr -> f64
// CHECK:    %[[v:.+]] = llvm.load %[[ppx]] : !llvm.ptr -> f64
// CHECK:    %[[l:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK:    %[[r:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK:    %[[ds:.+]] = arith.addf %[[l]], %[[r]] fastmath<fast> : f64
// CHECK:    %[[s:.+]] = arith.mulf %[[v]], %[[v]] : f64
// CHECK:    llvm.store %[[ds]], %[[spy]] : f64, !llvm.ptr
// CHECK:    llvm.store %[[s]], %[[ppy]] : f64, !llvm.ptr

// -----

module {
  func.func @array_pack(%x : !llvm.ptr, %y : !llvm.ptr) {
    %u = llvm.mlir.poison : !llvm.array<2 x ptr>
    %a0 = llvm.insertvalue %x, %u[0] : !llvm.array<2 x ptr>
    %a1 = llvm.insertvalue %y, %a0[1] : !llvm.array<2 x ptr>
    %px = llvm.extractvalue %a1[0] : !llvm.array<2 x ptr>
    %py = llvm.extractvalue %a1[1] : !llvm.array<2 x ptr>
    %v = llvm.load %px : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %py : f64, !llvm.ptr
    return
  }

  func.func @darray_pack(%x : !llvm.ptr, %dx : !llvm.ptr, %y : !llvm.ptr, %dy : !llvm.ptr) {
    enzyme.fwddiff @array_pack(%x, %dx, %y, %dy) <{
      activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity=[]
    }> : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL: func.func private @fwddiffearray_pack
// CHECK-SAME: (%[[x:.+]]: !llvm.ptr, %[[dx:.+]]: !llvm.ptr, %[[y:.+]]: !llvm.ptr, %[[dy:.+]]: !llvm.ptr) {
// CHECK:    %[[sa0:.+]] = llvm.insertvalue %[[dx]], %{{.+}}[0] : !llvm.array<2 x ptr>
// CHECK:    %[[pa0:.+]] = llvm.insertvalue %[[x]], %{{.+}}[0] : !llvm.array<2 x ptr>
// CHECK:    %[[sa1:.+]] = llvm.insertvalue %[[dy]], %[[sa0]][1] : !llvm.array<2 x ptr>
// CHECK:    %[[pa1:.+]] = llvm.insertvalue %[[y]], %[[pa0]][1] : !llvm.array<2 x ptr>
// CHECK:    %[[spx:.+]] = llvm.extractvalue %[[sa1]][0] : !llvm.array<2 x ptr>
// CHECK:    %[[ppx:.+]] = llvm.extractvalue %[[pa1]][0] : !llvm.array<2 x ptr>
// CHECK:    %[[spy:.+]] = llvm.extractvalue %[[sa1]][1] : !llvm.array<2 x ptr>
// CHECK:    %[[ppy:.+]] = llvm.extractvalue %[[pa1]][1] : !llvm.array<2 x ptr>
// CHECK:    %[[dv:.+]] = llvm.load %[[spx]] : !llvm.ptr -> f64
// CHECK:    %[[v:.+]] = llvm.load %[[ppx]] : !llvm.ptr -> f64
// CHECK:    %[[l:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK:    %[[r:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK:    %[[ds:.+]] = arith.addf %[[l]], %[[r]] fastmath<fast> : f64
// CHECK:    %[[s:.+]] = arith.mulf %[[v]], %[[v]] : f64
// CHECK:    llvm.store %[[ds]], %[[spy]] : f64, !llvm.ptr
// CHECK:    llvm.store %[[s]], %[[ppy]] : f64, !llvm.ptr
