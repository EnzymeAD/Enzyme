// RUN: %eopt --outline-enzyme-regions --split-input-file %s | FileCheck %s

func.func @to_outline(%26: !llvm.ptr, %27: !llvm.ptr, %28: !llvm.ptr, %29: !llvm.ptr) {
  %cst = arith.constant 5.6 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c1, %c1, %c1) step (%c1, %c1, %c1) {
    scf.parallel (%arg5, %arg6, %arg7) = (%c0, %c0, %c0) to (%c4, %c1, %c1) step (%c1, %c1, %c1) {
      memref.alloca_scope  {
        scf.execute_region {
          enzyme.autodiff_region(%26, %27, %28, %29) {
          ^bb0(%arg8: !llvm.ptr, %arg9: !llvm.ptr):
            %63 = arith.index_castui %arg5 : index to i64
            %64 = llvm.getelementptr inbounds|nuw %arg8[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %65 = llvm.load %64 invariant {alignment = 4 : i64} : !llvm.ptr -> f32
            %66 = arith.extf %65 : f32 to f64
            %67 = arith.mulf %66, %cst {fastmathFlags = #llvm.fastmath<contract>} : f64
            %68 = arith.truncf %67 : f64 to f32
            %69 = llvm.getelementptr inbounds|nuw %arg9[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            llvm.store %68, %69 {alignment = 4 : i64} : f32, !llvm.ptr
            enzyme.yield
          } attributes {activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], fn = "outlined_func", ret_activity = []} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
          scf.yield
        }
      }
      scf.reduce 
    }
    scf.reduce 
  }
  return
}

// CHECK: func.func @outlined_func(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: index, %arg3: f64) {
// CHECK-NEXT:    %0 = arith.index_castui %arg2 : index to i64
// CHECK-NEXT:    %1 = llvm.getelementptr inbounds|nuw %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:    %2 = llvm.load %1 invariant {alignment = 4 : i64} : !llvm.ptr -> f32
// CHECK-NEXT:    %3 = arith.extf %2 : f32 to f64
// CHECK-NEXT:    %4 = arith.mulf %3, %arg3 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:    %5 = arith.truncf %4 : f64 to f32
// CHECK-NEXT:    %6 = llvm.getelementptr inbounds|nuw %arg1[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:    llvm.store %5, %6 {alignment = 4 : i64} : f32, !llvm.ptr
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

llvm.func internal @d_Z6squarePfS_(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
  %0 = llvm.mlir.constant(5.600000e+00 : f64) : f64
  enzyme.autodiff_region(%arg0, %arg1, %arg2, %arg3) {
  ^bb0(%arg4: !llvm.ptr, %arg5: !llvm.ptr):
    %1 = nvvm.read.ptx.sreg.tid.x : i32
    %2 = llvm.zext nneg %1 : i32 to i64
    %3 = llvm.getelementptr inbounds|nuw %arg4[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %4 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> f32
    %5 = llvm.fpext %4 : f32 to f64
    %6 = llvm.fmul %5, %0 {fastmathFlags = #llvm.fastmath<contract>} : f64
    %7 = llvm.fptrunc %6 : f64 to f32
    %8 = llvm.getelementptr inbounds|nuw %arg5[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %7, %8 {alignment = 4 : i64} : f32, !llvm.ptr
    enzyme.yield
  } attributes {activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], fn = "_Z6squarePfS_", fn_attrs = {CConv = #llvm.cconv<ccc>, arg_attrs = [{llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}], dso_local, frame_pointer = #llvm.framePointerKind<all>, linkage = #llvm.linkage<internal>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_86"]], sym_name = "_Z6squarePfS_", sym_visibility = "private", target_cpu = "sm_86", target_features = #llvm.target_features<["+ptx88", "+sm_86"]>, unnamed_addr = 0 : i64, visibility_ = 0 : i64, will_return}, ret_activity = []} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// Attributes should be passed back to the outlined function
// CHECK: func.func private @_Z6squarePfS_(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg2: f64) attributes {CConv = #llvm.cconv<ccc>, dso_local, frame_pointer = #llvm.framePointerKind<all>, linkage = #llvm.linkage<internal>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_86"]], target_cpu = "sm_86", target_features = #llvm.target_features<["+ptx88", "+sm_86"]>, unnamed_addr = 0 : i64, visibility_ = 0 : i64, will_return} {

// -----

func.func @outline_multi(%x: f64, %dr: f64) -> (f64, f64) {
  %r0 = enzyme.autodiff_region(%x, %dr) {
  ^bb0(%arg0: f64):
    %sq = arith.mulf %arg0, %arg0 : f64
    enzyme.yield %sq : f64
  } attributes {activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>]} : (f64, f64) -> f64

  %r1 = enzyme.autodiff_region(%x, %dr) {
  ^bb0(%arg0: f64):
    %add = arith.addf %arg0, %arg0 : f64
    enzyme.yield %add : f64
  } attributes {activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>]} : (f64, f64) -> f64
  return %r0, %r1 : f64, f64
}

// CHECK: func.func @outline_multi(%arg0: f64, %arg1: f64) -> (f64, f64) {
// CHECK-NEXT:    %0 = enzyme.autodiff @outline_multi_to_diff0(%arg0, %arg1) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (f64, f64) -> f64
// CHECK-NEXT:    %1 = enzyme.autodiff @outline_multi_to_diff1(%arg0, %arg1) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (f64, f64) -> f64
// CHECK-NEXT:    return %0, %1 : f64, f64
// CHECK-NEXT:  }

// CHECK: func.func @outline_multi_to_diff1(%arg0: f64) -> f64 {
// CHECK-NEXT:    %0 = arith.addf %arg0, %arg0 : f64
// CHECK-NEXT:    return %0 : f64
// CHECK-NEXT:  }

// CHECK: func.func @outline_multi_to_diff0(%arg0: f64) -> f64 {
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg0 : f64
// CHECK-NEXT:    return %0 : f64
// CHECK-NEXT:  }
