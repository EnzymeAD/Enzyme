// RUN: %eopt --inline-enzyme-regions --split-input-file %s | FileCheck %s

func.func @square(%x: f64) -> f64 {
  %next = arith.mulf %x, %x : f64
  return %next : f64
}

func.func @dsquare(%x: f64, %dr: f64) -> f64 {
  %r = enzyme.autodiff @square(%x, %dr)
    {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
  return %r : f64
}

// CHECK:  func.func @dsquare(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %0 = enzyme.autodiff_region(%arg0, %arg1) {
// CHECK-NEXT:    ^bb0(%arg2: f64):
// CHECK-NEXT:      %1 = arith.mulf %arg2, %arg2 : f64
// CHECK-NEXT:      enzyme.yield %1 : f64
// CHECK-NEXT:    } attributes {activity = [#enzyme<activity enzyme_active>], fn = "square", fn_attrs = {}, ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (f64, f64) -> f64
// CHECK-NEXT:    return %0 : f64
// CHECK-NEXT:  }

// -----

llvm.func internal @_Z6squarePfS_(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_86"]], target_cpu = "sm_86", target_features = #llvm.target_features<["+ptx88", "+sm_86"]>, will_return, sym_visibility = "private"} {
  %0 = llvm.mlir.constant(5.600000e+00 : f64) : f64
  %1 = nvvm.read.ptx.sreg.tid.x : i32
  %2 = llvm.zext nneg %1 : i32 to i64
  %3 = llvm.getelementptr inbounds|nuw %arg0[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %4 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> f32
  %5 = llvm.fpext %4 : f32 to f64
  %6 = llvm.fmul %5, %0 {fastmathFlags = #llvm.fastmath<contract>} : f64
  %7 = llvm.fptrunc %6 : f64 to f32
  %8 = llvm.getelementptr inbounds|nuw %arg1[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %7, %8 {alignment = 4 : i64} : f32, !llvm.ptr
  llvm.return
}

llvm.func internal @d_Z6squarePfS_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
  enzyme.autodiff @_Z6squarePfS_(%arg0, %arg1, %arg2, %arg3)
    {
      activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
      ret_activity=[]
    } : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// Make sure that function attributes are preserved
// CHECK: llvm.func internal @d_Z6squarePfS_
// CHECK:    enzyme.autodiff_region(%arg0, %arg1, %arg2, %arg3) {
// CHECK:    } attributes {activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], fn = "_Z6squarePfS_", fn_attrs = {CConv = #llvm.cconv<ccc>, arg_attrs = [{llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}], dso_local, frame_pointer = #llvm.framePointerKind<all>, linkage = #llvm.linkage<internal>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_86"]], sym_visibility = "private", target_cpu = "sm_86", target_features = #llvm.target_features<["+ptx88", "+sm_86"]>, unnamed_addr = 0 : i64, visibility_ = 0 : i64, will_return}, ret_activity = []} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> (
