// RUN: %eopt --print-activity-analysis='use-annotations' --split-input-file %s | FileCheck %s

// CHECK-LABEL: processing function @load_nested
// CHECK: forward value origins:
// CHECK:      distinct[0]<#enzyme.pseudoclass<@load_nested(1, 0)>> originates from [#enzyme.argorigin<@load_nested(0)>, #enzyme.argorigin<@load_nested(1)>]
func.func @load_nested(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %data = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
  %val = llvm.load %data : !llvm.ptr -> f64
  llvm.store %val, %arg1 : f64, !llvm.ptr
  return
}

// Since we're not keeping track of origin depth, just passing %alloc to @load_nested
// means we need to union the origins of %alloc with everything %alloc points to
// (%inner in this case)
// CHECK-LABEL: processing function @pass_pointer_to
// CHECK: forward value origins:
// CHECK:      distinct[0]<#enzyme.pseudoclass<@pass_pointer_to(2, 0)>> originates from [#enzyme.argorigin<@pass_pointer_to(0)>, #enzyme.argorigin<@pass_pointer_to(1)>, #enzyme.argorigin<@pass_pointer_to(2)>]
func.func @pass_pointer_to(%arg0: f64, %alloc: !llvm.ptr, %out: !llvm.ptr) {
  %one = llvm.mlir.constant (1) : i64
  %inner = llvm.load %alloc : !llvm.ptr -> !llvm.ptr
  llvm.store %arg0, %inner : f64, !llvm.ptr
  func.call @load_nested(%alloc, %out) : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func @load_double_nested(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %data = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
  %val = llvm.load %data : !llvm.ptr -> f64
  // Rather than storing to %arg1, store to what %arg1 points to.
  %out_data = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
  llvm.store %val, %out_data : f64, !llvm.ptr
  return
}

// TODO: This is an issue where we don't see what %out points to, but it points to something.
// Probably need to fix the summarized func processing in p2p.
func.func @pass_pointer_to(%arg0: f64, %alloc: !llvm.ptr, %out: !llvm.ptr) {
  %one = llvm.mlir.constant (1) : i64
  %inner = llvm.load %alloc : !llvm.ptr -> !llvm.ptr
  llvm.store %arg0, %inner : f64, !llvm.ptr
  func.call @load_double_nested(%alloc, %out) : (!llvm.ptr, !llvm.ptr) -> ()
  return
}
