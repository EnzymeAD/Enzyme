// RUN: %eopt --print-activity-analysis='use-annotations' --split-input-file %s | FileCheck %s

// CHECK-LABEL: processing function @sparse_callee
// CHECK: "fadd"(#0)
// CHECK:   sources: [#enzyme.argorigin<@sparse_callee(0)>]
// CHECK:   sinks:   [#enzyme.retorigin<@sparse_callee(1)>]
func.func @sparse_callee(%arg0: f64) -> (f64, f64) {
  %zero = llvm.mlir.constant (0.0) : f64
  %0 = llvm.fadd %arg0, %arg0 {tag = "fadd"} : f64
  return %zero, %0 : f64, f64
}

// CHECK-LABEL: processing function @sparse_caller
// CHECK: "fmul"(#0)
// CHECK:   sources: [#enzyme.argorigin<@sparse_caller(1)>]
// CHECK:   sinks:   [#enzyme.retorigin<@sparse_caller(0)>]
func.func @sparse_caller(%unused: i64, %arg0: f64) -> f64 {
  %0 = llvm.fmul %arg0, %arg0 {tag = "fmul"} : f64
  %zero, %1 = call @sparse_callee(%0) : (f64) -> (f64, f64)
  return %1 : f64
}

// -----

func.func @aliased_callee(%arg0: !llvm.ptr) -> !llvm.ptr {
  %c0 = llvm.mlir.constant (0) : i64
  %0 = llvm.getelementptr inbounds %arg0[%c0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  return %0 : !llvm.ptr
}

// Test propagation of aliasing through function calls
// CHECK-LABEL: processing function @loadstore
// CHECK: "alloca"(#0)
// CHECK:   sources: [#enzyme.argorigin<@loadstore(0)>]
// CHECK:   sinks:   [#enzyme.retorigin<@loadstore(0)>]
func.func @loadstore(%arg0: f64) -> f64 {
  %c1 = llvm.mlir.constant (1) : i64
  %ptr = llvm.alloca %c1 x f64 {tag = "alloca"} : (i64) -> !llvm.ptr
  %ptr2 = call @aliased_callee(%ptr) : (!llvm.ptr) -> !llvm.ptr
  llvm.store %arg0, %ptr2 : f64, !llvm.ptr
  %0 = llvm.load %ptr : !llvm.ptr -> f64
  return %0 : f64
}

// -----

llvm.func local_unnamed_addr @malloc(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["allockind", "9"], ["allocsize", "4294967295"], ["alloc-family", "malloc"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], sym_visibility = "private", target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>}

func.func @returnptr(%arg0: f64) -> !llvm.ptr {
  %c8 = llvm.mlir.constant (8) : i64
  %ptr = llvm.call @malloc(%c8) {tag = "malloc"} : (i64) -> !llvm.ptr
  llvm.store %arg0, %ptr : f64, !llvm.ptr
  return %ptr : !llvm.ptr
}

// CHECK-LABEL: processing function @loadstore
// CHECK: "loaded"(#0)
// CHECK:   sources: [#enzyme.argorigin<@loadstore(0)>]
// CHECK:   sinks:   [#enzyme.retorigin<@loadstore(0)>]
func.func @loadstore(%arg0: f64) -> f64 {
  %ptr = call @returnptr(%arg0) : (f64) -> !llvm.ptr
  %val = llvm.load %ptr {tag = "loaded"} : !llvm.ptr -> f64
  return %val : f64
}

// -----

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
// CHECK: "val"(#0)
// CHECK:   sources: [#enzyme.argorigin<@pass_pointer_to(0)>]
// CHECK:   sinks:   [#enzyme.argorigin<@pass_pointer_to(1)>, #enzyme.argorigin<@pass_pointer_to(2)>]
// CHECK: "inner"(#0)
// CHECK:   sources: [#enzyme.argorigin<@pass_pointer_to(0)>, #enzyme.argorigin<@pass_pointer_to(1)>]
// CHECK:   sinks:   [#enzyme.argorigin<@pass_pointer_to(1)>, #enzyme.argorigin<@pass_pointer_to(2)>]
func.func @pass_pointer_to(%arg0: f64, %alloc: !llvm.ptr, %out: !llvm.ptr) {
  %one = llvm.mlir.constant (1) : i64
  %val = llvm.fmul %arg0, %arg0 {tag = "val"} : f64
  %inner = llvm.load %alloc {tag = "inner"} : !llvm.ptr -> !llvm.ptr
  llvm.store %val, %inner : f64, !llvm.ptr
  func.call @load_nested(%alloc, %out) : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func @callee(%val: f64, %out: !llvm.ptr) {
  llvm.store %val, %out : f64, !llvm.ptr
  return
}

// CHECK-LABEL: processing function @caller
// CHECK: "square"(#0)
// CHECK:   sources: [#enzyme.argorigin<@caller(1)>]
// CHECK:   sinks:   [#enzyme.argorigin<@caller(2)>]
func.func @caller(%unused: i32, %val: f64, %out: !llvm.ptr) {
  %square = llvm.fmul %val, %val {tag = "square"} : f64
  call @callee(%square, %out) : (f64, !llvm.ptr) -> ()
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
