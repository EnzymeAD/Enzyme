// RUN: %eopt --print-activity-analysis='use-annotations' --split-input-file %s | FileCheck %s

#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "mat_mult">
#alias_scope_domain1 = #llvm.alias_scope_domain<id = distinct[1]<>, description = "mat_mult">
#alias_scope_domain2 = #llvm.alias_scope_domain<id = distinct[2]<>, description = "mat_mult">
#alias_scope_domain3 = #llvm.alias_scope_domain<id = distinct[3]<>, description = "mat_mult">
#alias_scope_domain4 = #llvm.alias_scope_domain<id = distinct[4]<>, description = "mat_mult">
#alias_scope_domain5 = #llvm.alias_scope_domain<id = distinct[5]<>, description = "mat_mult">
#alias_scope_domain6 = #llvm.alias_scope_domain<id = distinct[6]<>, description = "mat_mult">
#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#alias_scope = #llvm.alias_scope<id = distinct[7]<>, domain = #alias_scope_domain, description = "mat_mult: %rhs">
#alias_scope1 = #llvm.alias_scope<id = distinct[8]<>, domain = #alias_scope_domain, description = "mat_mult: %lhs">
#alias_scope2 = #llvm.alias_scope<id = distinct[9]<>, domain = #alias_scope_domain, description = "mat_mult: %out">
#alias_scope3 = #llvm.alias_scope<id = distinct[10]<>, domain = #alias_scope_domain1, description = "mat_mult: %lhs">
#alias_scope4 = #llvm.alias_scope<id = distinct[11]<>, domain = #alias_scope_domain1, description = "mat_mult: %rhs">
#alias_scope5 = #llvm.alias_scope<id = distinct[12]<>, domain = #alias_scope_domain1, description = "mat_mult: %out">
#alias_scope6 = #llvm.alias_scope<id = distinct[13]<>, domain = #alias_scope_domain2, description = "mat_mult: %lhs">
#alias_scope7 = #llvm.alias_scope<id = distinct[14]<>, domain = #alias_scope_domain2, description = "mat_mult: %rhs">
#alias_scope8 = #llvm.alias_scope<id = distinct[15]<>, domain = #alias_scope_domain2, description = "mat_mult: %out">
#alias_scope9 = #llvm.alias_scope<id = distinct[16]<>, domain = #alias_scope_domain3, description = "mat_mult: %out">
#alias_scope10 = #llvm.alias_scope<id = distinct[17]<>, domain = #alias_scope_domain3, description = "mat_mult: %lhs">
#alias_scope11 = #llvm.alias_scope<id = distinct[18]<>, domain = #alias_scope_domain3, description = "mat_mult: %rhs">
#alias_scope12 = #llvm.alias_scope<id = distinct[19]<>, domain = #alias_scope_domain4, description = "mat_mult: %lhs">
#alias_scope13 = #llvm.alias_scope<id = distinct[20]<>, domain = #alias_scope_domain4, description = "mat_mult: %out">
#alias_scope14 = #llvm.alias_scope<id = distinct[21]<>, domain = #alias_scope_domain4, description = "mat_mult: %rhs">
#alias_scope15 = #llvm.alias_scope<id = distinct[22]<>, domain = #alias_scope_domain5, description = "mat_mult: %lhs">
#alias_scope16 = #llvm.alias_scope<id = distinct[23]<>, domain = #alias_scope_domain5, description = "mat_mult: %rhs">
#alias_scope17 = #llvm.alias_scope<id = distinct[24]<>, domain = #alias_scope_domain5, description = "mat_mult: %out">
#alias_scope18 = #llvm.alias_scope<id = distinct[25]<>, domain = #alias_scope_domain6, description = "mat_mult: %lhs">
#alias_scope19 = #llvm.alias_scope<id = distinct[26]<>, domain = #alias_scope_domain6, description = "mat_mult: %rhs">
#alias_scope20 = #llvm.alias_scope<id = distinct[27]<>, domain = #alias_scope_domain6, description = "mat_mult: %out">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "any pointer", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "double", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_type_desc4 = #llvm.tbaa_type_desc<id = "Matrix", members = {<#tbaa_type_desc1, 0>, <#tbaa_type_desc1, 4>, <#tbaa_type_desc2, 8>}>
#tbaa_tag3 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_tag4 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc1, offset = 4>
#tbaa_tag5 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc2, offset = 8>

llvm.func local_unnamed_addr @malloc(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["allockind", "9"], ["allocsize", "4294967295"], ["alloc-family", "malloc"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], sym_visibility = "private", target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>}
llvm.func local_unnamed_addr @free(!llvm.ptr {llvm.allocptr, llvm.nocapture, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nounwind", "willreturn", ["allockind", "4"], ["alloc-family", "malloc"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], sym_visibility = "private", target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>}

// CHECK-LABEL: processing function @to_pose_params
// CHECK: p2p summary:
// CHECK-NEXT:    distinct[0]<"arg-to_pose_params-3"> -> [distinct[0]<"arg-to_pose_params-3-deref">, distinct[1]<"fresh-pose_params_malloc">]
// CHECK-NEXT:    distinct[0]<"fresh-pose_params_malloc"> -> []
llvm.func local_unnamed_addr @to_pose_params(
  %arg0: i32 {llvm.noundef},
  %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly},
  %arg2: !llvm.ptr {llvm.nocapture, llvm.readnone},
  %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}
) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
  %0 = llvm.mlir.constant(3 : i32) : i32
  %1 = llvm.mlir.constant(0 : i64) : i64
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.mlir.constant(2 : i32) : i32
  %4 = llvm.mlir.zero : !llvm.ptr
  %5 = llvm.mlir.constant(-3 : i32) : i32
  %6 = llvm.mlir.constant(3 : i64) : i64
  %7 = llvm.mlir.constant(0 : i8) : i8
  %8 = llvm.mlir.constant(6 : i64) : i64
  %9 = llvm.mlir.constant(24 : i64) : i64
  %10 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %11 = llvm.mlir.constant(1 : i64) : i64
  %12 = llvm.mlir.constant(0 : i32) : i32
  %13 = llvm.mlir.constant(5 : i32) : i32
  %14 = llvm.mlir.constant(6 : i32) : i32
  %15 = llvm.add %arg0, %0 overflow<nsw>  : i32
  %16 = llvm.load %arg3 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %17 = llvm.getelementptr inbounds %arg3[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %18 = llvm.load %17 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %19 = llvm.mul %18, %16 overflow<nsw>  : i32
  %20 = llvm.mul %15, %0 overflow<nsw>  : i32
  %21 = llvm.icmp "eq" %19, %20 : i32
  llvm.cond_br %21, ^bb6, ^bb1
^bb1:  // pred: ^bb0
  %22 = llvm.getelementptr inbounds %arg3[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %23 = llvm.load %22 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %24 = llvm.icmp "eq" %23, %4 : !llvm.ptr
  llvm.cond_br %24, ^bb3, ^bb2
^bb2:  // pred: ^bb1
  llvm.call @free(%23) : (!llvm.ptr) -> ()
  llvm.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  %25 = llvm.icmp "sgt" %arg0, %5 : i32
  llvm.cond_br %25, ^bb4, ^bb5(%4 : !llvm.ptr)
^bb4:  // pred: ^bb3
  %26 = llvm.zext %20 : i32 to i64
  %27 = llvm.shl %26, %6 overflow<nsw, nuw>  : i64
  %28 = llvm.call @malloc(%27) {tag = "pose_params_malloc"} : (i64) -> !llvm.ptr
  llvm.br ^bb5(%28 : !llvm.ptr)
^bb5(%29: !llvm.ptr):  // 2 preds: ^bb3, ^bb4
  llvm.store %29, %22 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb6
^bb6:  // 2 preds: ^bb0, ^bb5
  llvm.store %15, %17 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
  llvm.store %0, %arg3 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
  %30 = llvm.icmp "sgt" %arg0, %5 : i32
  %31 = llvm.getelementptr inbounds %arg3[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %32 = llvm.load %31 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  llvm.cond_br %30, ^bb7, ^bb8
^bb7:  // pred: ^bb6
  %33 = llvm.zext %20 : i32 to i64
  %34 = llvm.shl %33, %6 overflow<nsw, nuw>  : i64
  "llvm.intr.memset"(%32, %7, %34) <{isVolatile = false, tbaa = [#tbaa_tag1]}> : (!llvm.ptr, i8, i64) -> ()
  llvm.br ^bb8
^bb8:  // 2 preds: ^bb6, ^bb7
  %35 = llvm.getelementptr %32[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %36 = llvm.getelementptr %32[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  "llvm.intr.memcpy"(%32, %arg1, %9) <{isVolatile = false, tbaa = [#tbaa_tag1]}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %37 = llvm.getelementptr %arg1[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb9(%1 : i64)
^bb9(%38: i64):  // 2 preds: ^bb8, ^bb9
  %39 = llvm.getelementptr %35[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %10, %39 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %40 = llvm.getelementptr %37[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %41 = llvm.load %40 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %42 = llvm.getelementptr %36[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %41, %42 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %43 = llvm.add %38, %11 overflow<nsw, nuw>  : i64
  %44 = llvm.icmp "eq" %43, %6 : i64
  llvm.cond_br %44, ^bb10(%12, %13, %14 : i32, i32, i32), ^bb9(%43 : i64) {loop_annotation = #loop_annotation}
^bb10(%45: i32, %46: i32, %47: i32):  // 2 preds: ^bb9, ^bb14
  %48 = llvm.sext %46 : i32 to i64
  %49 = llvm.add %46, %0  : i32
  llvm.br ^bb11(%48, %3, %47 : i64, i32, i32)
^bb11(%50: i64, %51: i32, %52: i32):  // 2 preds: ^bb10, ^bb13
  %53 = llvm.sext %52 : i32 to i64
  %54 = llvm.getelementptr inbounds %arg1[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %55 = llvm.load %54 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %56 = llvm.mul %50, %6 overflow<nsw>  : i64
  %57 = llvm.getelementptr inbounds %32[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %55, %57 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %58 = llvm.add %52, %2 overflow<nsw>  : i32
  %59 = llvm.icmp "eq" %51, %3 : i32
  llvm.cond_br %59, ^bb12, ^bb13(%58 : i32)
^bb12:  // pred: ^bb11
  %60 = llvm.sext %58 : i32 to i64
  %61 = llvm.getelementptr inbounds %arg1[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %62 = llvm.load %61 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %63 = llvm.getelementptr %57[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %62, %63 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %64 = llvm.add %52, %3 overflow<nsw>  : i32
  llvm.br ^bb13(%64 : i32)
^bb13(%65: i32):  // 2 preds: ^bb11, ^bb12
  %66 = llvm.add %50, %11 overflow<nsw>  : i64
  %67 = llvm.add %51, %2 overflow<nsw, nuw>  : i32
  %68 = llvm.trunc %66 : i64 to i32
  %69 = llvm.icmp "eq" %49, %68 : i32
  llvm.cond_br %69, ^bb14, ^bb11(%66, %67, %65 : i64, i32, i32) {loop_annotation = #loop_annotation}
^bb14:  // pred: ^bb13
  %70 = llvm.trunc %50 : i64 to i32
  %71 = llvm.add %70, %3 overflow<nsw>  : i32
  %72 = llvm.add %45, %2 overflow<nsw, nuw>  : i32
  %73 = llvm.icmp "eq" %72, %13 : i32
  llvm.cond_br %73, ^bb15, ^bb10(%72, %71, %65 : i32, i32, i32) {loop_annotation = #loop_annotation}
^bb15:  // pred: ^bb14
  llvm.return
}

// CHECK-LABEL: processing function @euler_angles_to_rotation_matrix
// CHECK: p2p summary:
// CHECK-NEXT:    distinct[0]<"arg-euler_angles_to_rotation_matrix-1"> -> [distinct[0]<"arg-euler_angles_to_rotation_matrix-1-deref">, distinct[1]<"fresh-euler_angles_malloc">]
// CHECK-NEXT:    distinct[0]<"fresh-euler_angles_malloc"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-malloc_RX"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-malloc_RY"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-malloc_RZ"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-malloc_tmp"> -> []
llvm.func local_unnamed_addr @euler_angles_to_rotation_matrix(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
  %0 = llvm.mlir.constant(2 : i64) : i64
  %1 = llvm.mlir.constant(1 : i64) : i64
  %2 = llvm.mlir.constant(72 : i64) : i64
  %3 = llvm.mlir.constant(0 : i64) : i64
  %4 = llvm.mlir.constant(3 : i64) : i64
  %5 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %6 = llvm.mlir.constant(0.000000e+00 : f64) : f64
  %7 = llvm.mlir.constant(4 : i64) : i64
  %8 = llvm.mlir.constant(5 : i64) : i64
  %9 = llvm.mlir.constant(7 : i64) : i64
  %10 = llvm.mlir.constant(8 : i64) : i64
  %11 = llvm.mlir.constant(6 : i64) : i64
  %12 = llvm.mlir.constant(1 : i32) : i32
  %13 = llvm.mlir.constant(9 : i32) : i32
  %14 = llvm.mlir.constant(2 : i32) : i32
  %15 = llvm.mlir.zero : !llvm.ptr
  %16 = llvm.mlir.constant(3 : i32) : i32
  %17 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %18 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %19 = llvm.load %18 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %20 = llvm.getelementptr inbounds %arg0[%1] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %21 = llvm.load %20 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %22 = llvm.call @malloc(%2) {tag = "malloc_RX"} : (i64) -> !llvm.ptr
  %23 = llvm.call @malloc(%2) {tag = "malloc_RY"} : (i64) -> !llvm.ptr
  %24 = llvm.call @malloc(%2) {tag = "malloc_RZ"} : (i64) -> !llvm.ptr
  llvm.br ^bb1(%3 : i64)
^bb1(%25: i64):  // 2 preds: ^bb0, ^bb3
  %26 = llvm.mul %25, %4 overflow<nsw, nuw>  : i64
  %27 = llvm.getelementptr %22[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb2(%3 : i64)
^bb2(%28: i64):  // 2 preds: ^bb1, ^bb2
  %29 = llvm.icmp "eq" %25, %28 : i64
  %30 = llvm.select %29, %5, %6 : i1, f64
  %31 = llvm.getelementptr %27[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %30, %31 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %32 = llvm.add %28, %1 overflow<nsw, nuw>  : i64
  %33 = llvm.icmp "eq" %32, %4 : i64
  llvm.cond_br %33, ^bb3, ^bb2(%32 : i64) {loop_annotation = #loop_annotation}
^bb3:  // pred: ^bb2
  %34 = llvm.add %25, %1 overflow<nsw, nuw>  : i64
  %35 = llvm.icmp "eq" %34, %4 : i64
  llvm.cond_br %35, ^bb4, ^bb1(%34 : i64) {loop_annotation = #loop_annotation}
^bb4:  // pred: ^bb3
  %36 = llvm.intr.cos(%17)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
  %37 = llvm.getelementptr %22[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %36, %37 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %38 = llvm.intr.sin(%17)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
  %39 = llvm.getelementptr %22[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %38, %39 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %40 = llvm.fneg %38  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %41 = llvm.getelementptr inbounds %22[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %40, %41 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %42 = llvm.getelementptr %22[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %36, %42 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  llvm.br ^bb5(%3 : i64)
^bb5(%43: i64):  // 2 preds: ^bb4, ^bb7
  %44 = llvm.mul %43, %4 overflow<nsw, nuw>  : i64
  %45 = llvm.getelementptr %23[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb6(%3 : i64)
^bb6(%46: i64):  // 2 preds: ^bb5, ^bb6
  %47 = llvm.icmp "eq" %43, %46 : i64
  %48 = llvm.select %47, %5, %6 : i1, f64
  %49 = llvm.getelementptr %45[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %48, %49 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %50 = llvm.add %46, %1 overflow<nsw, nuw>  : i64
  %51 = llvm.icmp "eq" %50, %4 : i64
  llvm.cond_br %51, ^bb7, ^bb6(%50 : i64) {loop_annotation = #loop_annotation}
^bb7:  // pred: ^bb6
  %52 = llvm.add %43, %1 overflow<nsw, nuw>  : i64
  %53 = llvm.icmp "eq" %52, %4 : i64
  llvm.cond_br %53, ^bb8, ^bb5(%52 : i64) {loop_annotation = #loop_annotation}
^bb8:  // pred: ^bb7
  %54 = llvm.intr.cos(%19)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
  llvm.store %54, %23 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %55 = llvm.intr.sin(%19)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
  %56 = llvm.getelementptr inbounds %23[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %55, %56 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %57 = llvm.fneg %55  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %58 = llvm.getelementptr inbounds %23[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %57, %58 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %59 = llvm.getelementptr %23[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %54, %59 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  llvm.br ^bb9(%3 : i64)
^bb9(%60: i64):  // 2 preds: ^bb8, ^bb11
  %61 = llvm.mul %60, %4 overflow<nsw, nuw>  : i64
  %62 = llvm.getelementptr %24[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb10(%3 : i64)
^bb10(%63: i64):  // 2 preds: ^bb9, ^bb10
  %64 = llvm.icmp "eq" %60, %63 : i64
  %65 = llvm.select %64, %5, %6 : i1, f64
  %66 = llvm.getelementptr %62[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %65, %66 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %67 = llvm.add %63, %1 overflow<nsw, nuw>  : i64
  %68 = llvm.icmp "eq" %67, %4 : i64
  llvm.cond_br %68, ^bb11, ^bb10(%67 : i64) {loop_annotation = #loop_annotation}
^bb11:  // pred: ^bb10
  %69 = llvm.add %60, %1 overflow<nsw, nuw>  : i64
  %70 = llvm.icmp "eq" %69, %4 : i64
  llvm.cond_br %70, ^bb12, ^bb9(%69 : i64) {loop_annotation = #loop_annotation}
^bb12:  // pred: ^bb11
  %71 = llvm.intr.cos(%21)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
  llvm.store %71, %24 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %72 = llvm.intr.sin(%21)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
  %73 = llvm.getelementptr inbounds %24[%1] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %72, %73 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %74 = llvm.fneg %72  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %75 = llvm.getelementptr inbounds %24[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %74, %75 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %76 = llvm.getelementptr %24[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %71, %76 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %77 = llvm.call @malloc(%2) {tag = "malloc_tmp"} : (i64) -> !llvm.ptr
  llvm.br ^bb13(%3 : i64)
^bb13(%78: i64):  // 2 preds: ^bb12, ^bb17
  %79 = llvm.getelementptr inbounds %24[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %80 = llvm.getelementptr %77[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %81 = llvm.load %79 {alignment = 8 : i64, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  llvm.br ^bb14(%3 : i64)
^bb14(%82: i64):  // 2 preds: ^bb13, ^bb16
  %83 = llvm.mul %82, %4 overflow<nsw, nuw>  : i64
  %84 = llvm.getelementptr inbounds %23[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %85 = llvm.load %84 {alignment = 8 : i64, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %86 = llvm.fmul %85, %81  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %87 = llvm.getelementptr %80[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb15(%1, %86 : i64, f64)
^bb15(%88: i64, %89: f64):  // 2 preds: ^bb14, ^bb15
  %90 = llvm.mul %88, %4 overflow<nsw, nuw>  : i64
  %91 = llvm.getelementptr %79[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %92 = llvm.load %91 {alignment = 8 : i64, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %93 = llvm.getelementptr %84[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %94 = llvm.load %93 {alignment = 8 : i64, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %95 = llvm.fmul %94, %92  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %96 = llvm.fadd %95, %89  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %97 = llvm.add %88, %1 overflow<nsw, nuw>  : i64
  %98 = llvm.icmp "eq" %97, %4 : i64
  llvm.cond_br %98, ^bb16, ^bb15(%97, %96 : i64, f64) {loop_annotation = #loop_annotation}
^bb16:  // pred: ^bb15
  llvm.store %96, %87 {alignment = 8 : i64, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %99 = llvm.add %82, %1 overflow<nsw, nuw>  : i64
  %100 = llvm.icmp "eq" %99, %4 : i64
  llvm.cond_br %100, ^bb17, ^bb14(%99 : i64) {loop_annotation = #loop_annotation}
^bb17:  // pred: ^bb16
  %101 = llvm.add %78, %1 overflow<nsw, nuw>  : i64
  %102 = llvm.icmp "eq" %101, %4 : i64
  llvm.cond_br %102, ^bb18, ^bb13(%101 : i64) {loop_annotation = #loop_annotation}
^bb18:  // pred: ^bb17
  llvm.intr.experimental.noalias.scope.decl #alias_scope9
  %103 = llvm.load %arg1 {lookatme, alias_scopes = [#alias_scope9], alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %104 = llvm.getelementptr inbounds %arg1[%3, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %105 = llvm.load %104 {alias_scopes = [#alias_scope9], alignment = 4 : i64, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %106 = llvm.mul %105, %103 overflow<nsw>  : i32
  %107 = llvm.icmp "eq" %106, %13 : i32
  %108 = llvm.getelementptr inbounds %arg1[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %109 = llvm.load %108 {alias_scopes = [#alias_scope9], alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  llvm.cond_br %107, ^bb22(%109 : !llvm.ptr), ^bb19
^bb19:  // pred: ^bb18
  %110 = llvm.icmp "eq" %109, %15 : !llvm.ptr
  llvm.cond_br %110, ^bb21, ^bb20
^bb20:  // pred: ^bb19
  llvm.call @free(%109) {noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9]} : (!llvm.ptr) -> ()
  llvm.br ^bb21
^bb21:  // 2 preds: ^bb19, ^bb20
  %111 = llvm.call @malloc(%2) {tag = "euler_angles_malloc"} : (i64) -> !llvm.ptr
  llvm.store %111, %108 {alias_scopes = [#alias_scope9], alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb22(%111 : !llvm.ptr)
^bb22(%112: !llvm.ptr):  // 2 preds: ^bb18, ^bb21
  llvm.store %16, %104 {alias_scopes = [#alias_scope9], alignment = 4 : i64, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
  llvm.store %16, %arg1 {alias_scopes = [#alias_scope9], alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
  llvm.br ^bb23(%3 : i64)
^bb23(%113: i64):  // 2 preds: ^bb22, ^bb27
  %114 = llvm.getelementptr inbounds %77[%113] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %115 = llvm.getelementptr %112[%113] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %116 = llvm.load %114 {alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  llvm.br ^bb24(%3 : i64)
^bb24(%117: i64):  // 2 preds: ^bb23, ^bb26
  %118 = llvm.mul %117, %4 overflow<nsw, nuw>  : i64
  %119 = llvm.getelementptr inbounds %22[%118] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %120 = llvm.load %119 {readme, alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %121 = llvm.fmul %120, %116  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %122 = llvm.getelementptr %115[%118] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  // bm
  llvm.store %121, %122 {problematic, alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  llvm.br ^bb25(%1, %121 : i64, f64)
^bb25(%123: i64, %124: f64):  // 2 preds: ^bb24, ^bb25
  %125 = llvm.mul %123, %4 overflow<nsw, nuw>  : i64
  %126 = llvm.getelementptr %114[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %127 = llvm.load %126 {alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %128 = llvm.getelementptr %119[%123] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %129 = llvm.load %128 {alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %130 = llvm.fmul %129, %127  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %131 = llvm.fadd %130, %124  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %132 = llvm.add %123, %1 overflow<nsw, nuw>  : i64
  %133 = llvm.icmp "eq" %132, %4 : i64
  llvm.cond_br %133, ^bb26, ^bb25(%132, %131 : i64, f64) {loop_annotation = #loop_annotation}
^bb26:  // pred: ^bb25
  // bm2
  llvm.store %131, %122 {alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %134 = llvm.add %117, %1 overflow<nsw, nuw>  : i64
  %135 = llvm.icmp "eq" %134, %4 : i64
  llvm.cond_br %135, ^bb27, ^bb24(%134 : i64) {loop_annotation = #loop_annotation}
^bb27:  // pred: ^bb26
  %136 = llvm.add %113, %1 overflow<nsw, nuw>  : i64
  %137 = llvm.icmp "eq" %136, %4 : i64
  llvm.cond_br %137, ^bb28, ^bb23(%136 : i64) {loop_annotation = #loop_annotation}
^bb28:  // pred: ^bb27
  llvm.call @free(%22) : (!llvm.ptr) -> ()
  llvm.call @free(%23) : (!llvm.ptr) -> ()
  llvm.call @free(%24) : (!llvm.ptr) -> ()
  llvm.call @free(%77) : (!llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: processing function @relatives_to_absolutes
// CHECK: p2p summary:
// CHECK-NEXT:    distinct[0]<"arg-relatives_to_absolutes-1"> -> [distinct[0]<"arg-relatives_to_absolutes-1-deref">]
// CHECK-NEXT:    distinct[0]<"arg-relatives_to_absolutes-3"> -> [distinct[0]<"arg-relatives_to_absolutes-3-deref">, distinct[1]<"fresh-rta1">, distinct[2]<"fresh-rta2">]
// CHECK-NEXT:    distinct[0]<"fresh-rta1"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-rta2"> -> []
llvm.func local_unnamed_addr @relatives_to_absolutes(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(0 : i64) : i64
  %2 = llvm.mlir.constant(-1 : i32) : i32
  %3 = llvm.mlir.constant(1 : i32) : i32
  %4 = llvm.mlir.constant(2 : i32) : i32
  %5 = llvm.mlir.zero : !llvm.ptr
  %6 = llvm.mlir.constant(3 : i64) : i64
  %7 = llvm.mlir.constant(1 : i64) : i64
  %8 = llvm.icmp "sgt" %arg0, %0 : i32
  llvm.cond_br %8, ^bb1, ^bb23
^bb1:  // pred: ^bb0
  %9 = llvm.zext %arg0 : i32 to i64
  llvm.br ^bb2(%1 : i64)
^bb2(%10: i64):  // 2 preds: ^bb1, ^bb22
  %11 = llvm.getelementptr inbounds %arg2[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %12 = llvm.load %11 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
  %13 = llvm.icmp "eq" %12, %2 : i32
  llvm.cond_br %13, ^bb3, ^bb8
^bb3:  // pred: ^bb2
  %14 = llvm.getelementptr inbounds %arg3[%10] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %15 = llvm.getelementptr inbounds %arg1[%10] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %16 = llvm.getelementptr inbounds %arg3[%10, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %17 = llvm.load %16 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %18 = llvm.icmp "eq" %17, %5 : !llvm.ptr
  llvm.cond_br %18, ^bb5, ^bb4
^bb4:  // pred: ^bb3
  llvm.call @free(%17) : (!llvm.ptr) -> ()
  llvm.br ^bb5
^bb5:  // 2 preds: ^bb3, ^bb4
  %19 = llvm.getelementptr inbounds %arg1[%10, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %20 = llvm.load %19 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %21 = llvm.getelementptr inbounds %arg3[%10, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.store %20, %21 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
  %22 = llvm.load %15 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  llvm.store %22, %14 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
  %23 = llvm.mul %22, %20 overflow<nsw>  : i32
  %24 = llvm.sext %23 : i32 to i64
  %25 = llvm.shl %24, %6 overflow<nsw>  : i64
  %26 = llvm.call @malloc(%25) {tag = "rta1"} : (i64) -> !llvm.ptr
  llvm.store %26, %16 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  %27 = llvm.icmp "sgt" %23, %0 : i32
  llvm.cond_br %27, ^bb6, ^bb22
^bb6:  // pred: ^bb5
  %28 = llvm.getelementptr inbounds %arg1[%10, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %29 = llvm.load %28 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %30 = llvm.zext %23 : i32 to i64
  llvm.br ^bb7(%1 : i64)
^bb7(%31: i64):  // 2 preds: ^bb6, ^bb7
  %32 = llvm.getelementptr inbounds %29[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %33 = llvm.load %32 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %34 = llvm.getelementptr inbounds %26[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %33, %34 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %35 = llvm.add %31, %7 overflow<nsw, nuw>  : i64
  %36 = llvm.icmp "eq" %35, %30 : i64
  llvm.cond_br %36, ^bb22, ^bb7(%35 : i64) {loop_annotation = #loop_annotation}
^bb8:  // pred: ^bb2
  %37 = llvm.sext %12 : i32 to i64
  %38 = llvm.getelementptr inbounds %arg3[%37] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %39 = llvm.getelementptr inbounds %arg1[%10] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %40 = llvm.getelementptr inbounds %arg3[%10] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.intr.experimental.noalias.scope.decl #alias_scope3
  llvm.intr.experimental.noalias.scope.decl #alias_scope4
  llvm.intr.experimental.noalias.scope.decl #alias_scope5
  %41 = llvm.load %38 {alias_scopes = [#alias_scope3], alignment = 8 : i64, noalias_scopes = [#alias_scope4, #alias_scope5], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %42 = llvm.getelementptr inbounds %arg1[%10, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %43 = llvm.load %42 {alias_scopes = [#alias_scope4], alignment = 4 : i64, noalias_scopes = [#alias_scope3, #alias_scope5], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %44 = llvm.load %40 {alias_scopes = [#alias_scope5], alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %45 = llvm.getelementptr inbounds %arg3[%10, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %46 = llvm.load %45 {alias_scopes = [#alias_scope5], alignment = 4 : i64, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %47 = llvm.mul %46, %44 overflow<nsw>  : i32
  %48 = llvm.mul %43, %41 overflow<nsw>  : i32
  %49 = llvm.icmp "eq" %47, %48 : i32
  llvm.cond_br %49, ^bb14, ^bb9
^bb9:  // pred: ^bb8
  %50 = llvm.getelementptr inbounds %arg3[%10, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %51 = llvm.load %50 {alias_scopes = [#alias_scope5], alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %52 = llvm.icmp "eq" %51, %5 : !llvm.ptr
  llvm.cond_br %52, ^bb11, ^bb10
^bb10:  // pred: ^bb9
  llvm.call @free(%51) {noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5]} : (!llvm.ptr) -> ()
  llvm.br ^bb11
^bb11:  // 2 preds: ^bb9, ^bb10
  %53 = llvm.icmp "sgt" %48, %0 : i32
  llvm.cond_br %53, ^bb12, ^bb13(%5 : !llvm.ptr)
^bb12:  // pred: ^bb11
  %54 = llvm.zext %48 : i32 to i64
  %55 = llvm.shl %54, %6 overflow<nsw, nuw>  : i64
  %56 = llvm.call @malloc(%55) {tag = "rta2"} : (i64) -> !llvm.ptr
  llvm.br ^bb13(%56 : !llvm.ptr)
^bb13(%57: !llvm.ptr):  // 2 preds: ^bb11, ^bb12
  llvm.store %57, %50 {alias_scopes = [#alias_scope5], alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb14
^bb14:  // 2 preds: ^bb8, ^bb13
  llvm.store %43, %45 {alias_scopes = [#alias_scope5], alignment = 4 : i64, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
  llvm.store %41, %40 {alias_scopes = [#alias_scope5], alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
  %58 = llvm.icmp "sgt" %41, %0 : i32
  llvm.cond_br %58, ^bb15, ^bb22
^bb15:  // pred: ^bb14
  %59 = llvm.icmp "sgt" %43, %0 : i32
  %60 = llvm.getelementptr inbounds %arg3[%37, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %61 = llvm.getelementptr inbounds %arg1[%10, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %62 = llvm.getelementptr inbounds %arg3[%10, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %63 = llvm.getelementptr inbounds %arg3[%37, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %64 = llvm.zext %41 : i32 to i64
  %65 = llvm.zext %43 : i32 to i64
  llvm.br ^bb16(%1 : i64)
^bb16(%66: i64):  // 2 preds: ^bb15, ^bb21
  llvm.cond_br %59, ^bb17, ^bb21
^bb17:  // pred: ^bb16
  %67 = llvm.load %60 {alias_scopes = [#alias_scope3], alignment = 8 : i64, noalias_scopes = [#alias_scope4, #alias_scope5], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %68 = llvm.getelementptr inbounds %67[%66] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %69 = llvm.load %61 {alias_scopes = [#alias_scope4], alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope5], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %70 = llvm.load %39 {alias_scopes = [#alias_scope4], alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope5], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %71 = llvm.load %62 {alias_scopes = [#alias_scope5], alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %72 = llvm.load %63 {alias_scopes = [#alias_scope3], alignment = 4 : i64, noalias_scopes = [#alias_scope4, #alias_scope5], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %73 = llvm.icmp "sgt" %72, %3 : i32
  %74 = llvm.sext %70 : i32 to i64
  %75 = llvm.getelementptr %71[%66] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %76 = llvm.zext %72 : i32 to i64
  llvm.br ^bb18(%1 : i64)
^bb18(%77: i64):  // 2 preds: ^bb17, ^bb20
  %78 = llvm.load %68 {alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %79 = llvm.mul %77, %74 overflow<nsw>  : i64
  %80 = llvm.getelementptr inbounds %69[%79] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %81 = llvm.load %80 {alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %82 = llvm.fmul %81, %78  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %83 = llvm.mul %77, %64 overflow<nsw, nuw>  : i64
  %84 = llvm.getelementptr %75[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %82, %84 {alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  llvm.cond_br %73, ^bb19(%7, %82 : i64, f64), ^bb20
^bb19(%85: i64, %86: f64):  // 2 preds: ^bb18, ^bb19
  %87 = llvm.mul %85, %64 overflow<nsw, nuw>  : i64
  %88 = llvm.getelementptr %68[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %89 = llvm.load %88 {alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %90 = llvm.getelementptr %80[%85] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %91 = llvm.load %90 {alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %92 = llvm.fmul %91, %89  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %93 = llvm.fadd %92, %86  {fastmathFlags = #llvm.fastmath<fast>} : f64
  llvm.store %93, %84 {alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %94 = llvm.add %85, %7 overflow<nsw, nuw>  : i64
  %95 = llvm.icmp "eq" %94, %76 : i64
  llvm.cond_br %95, ^bb20, ^bb19(%94, %93 : i64, f64) {loop_annotation = #loop_annotation}
^bb20:  // 2 preds: ^bb18, ^bb19
  %96 = llvm.add %77, %7 overflow<nsw, nuw>  : i64
  %97 = llvm.icmp "eq" %96, %65 : i64
  llvm.cond_br %97, ^bb21, ^bb18(%96 : i64) {loop_annotation = #loop_annotation}
^bb21:  // 2 preds: ^bb16, ^bb20
  %98 = llvm.add %66, %7 overflow<nsw, nuw>  : i64
  %99 = llvm.icmp "eq" %98, %64 : i64
  llvm.cond_br %99, ^bb22, ^bb16(%98 : i64) {loop_annotation = #loop_annotation}
^bb22:  // 4 preds: ^bb5, ^bb7, ^bb14, ^bb21
  %100 = llvm.add %10, %7 overflow<nsw, nuw>  : i64
  %101 = llvm.icmp "eq" %100, %9 : i64
  llvm.cond_br %101, ^bb23, ^bb2(%100 : i64) {loop_annotation = #loop_annotation}
^bb23:  // 2 preds: ^bb0, ^bb22
  llvm.return
}

// CHECK-LABEL: processing function @angle_axis_to_rotation_matrix
// CHECK: p2p summary:
// CHECK-NEXT:    distinct[0]<"arg-angle_axis_to_rotation_matrix-1"> -> [distinct[0]<"arg-angle_axis_to_rotation_matrix-1-deref">]
llvm.func local_unnamed_addr @angle_axis_to_rotation_matrix(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, memory = #llvm.memory_effects<other = write, argMem = readwrite, inaccessibleMem = none>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.mlir.constant(3 : i64) : i64
  %2 = llvm.mlir.constant(1.000000e-04 : f64) : f64
  %3 = llvm.mlir.constant(2 : i64) : i64
  %4 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %5 = llvm.mlir.constant(0 : i64) : i64
  %6 = llvm.mlir.constant(2 : i32) : i32
  %7 = llvm.mlir.constant(1 : i32) : i32
  %8 = llvm.mlir.constant(0 : i32) : i32
  %9 = llvm.mlir.constant(0.000000e+00 : f64) : f64
  %10 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %11 = llvm.fmul %10, %10  {fastmathFlags = #llvm.fastmath<fast>} : f64
  llvm.br ^bb1(%0, %11 : i64, f64)
^bb1(%12: i64, %13: f64):  // 2 preds: ^bb0, ^bb1
  %14 = llvm.getelementptr inbounds %arg0[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %16 = llvm.fmul %15, %15  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %17 = llvm.fadd %16, %13  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %18 = llvm.add %12, %0 overflow<nsw, nuw>  : i64
  %19 = llvm.icmp "eq" %18, %1 : i64
  llvm.cond_br %19, ^bb2, ^bb1(%18, %17 : i64, f64) {loop_annotation = #loop_annotation}
^bb2:  // pred: ^bb1
  %20 = llvm.intr.sqrt(%17)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
  %21 = llvm.fcmp "olt" %20, %2 {fastmathFlags = #llvm.fastmath<fast>} : f64
  llvm.cond_br %21, ^bb3, ^bb9
^bb3:  // pred: ^bb2
  %22 = llvm.getelementptr inbounds %arg1[%5, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %23 = llvm.load %22 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %24 = llvm.icmp "sgt" %23, %8 : i32
  llvm.cond_br %24, ^bb4, ^bb10
^bb4:  // pred: ^bb3
  %25 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %26 = llvm.icmp "sgt" %25, %8 : i32
  %27 = llvm.getelementptr inbounds %arg1[%5, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %28 = llvm.zext %23 : i32 to i64
  %29 = llvm.zext %25 : i32 to i64
  llvm.br ^bb5(%5 : i64)
^bb5(%30: i64):  // 2 preds: ^bb4, ^bb8
  llvm.cond_br %26, ^bb6, ^bb8
^bb6:  // pred: ^bb5
  %31 = llvm.trunc %30 : i64 to i32
  %32 = llvm.mul %25, %31  : i32
  %33 = llvm.zext %32 : i32 to i64
  %34 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %35 = llvm.getelementptr %34[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb7(%5 : i64)
^bb7(%36: i64):  // 2 preds: ^bb6, ^bb7
  %37 = llvm.icmp "eq" %30, %36 : i64
  %38 = llvm.select %37, %4, %9 : i1, f64
  %39 = llvm.getelementptr %35[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %38, %39 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %40 = llvm.add %36, %0 overflow<nsw, nuw>  : i64
  %41 = llvm.icmp "eq" %40, %29 : i64
  llvm.cond_br %41, ^bb8, ^bb7(%40 : i64) {loop_annotation = #loop_annotation}
^bb8:  // 2 preds: ^bb5, ^bb7
  %42 = llvm.add %30, %0 overflow<nsw, nuw>  : i64
  %43 = llvm.icmp "eq" %42, %28 : i64
  llvm.cond_br %43, ^bb10, ^bb5(%42 : i64) {loop_annotation = #loop_annotation}
^bb9:  // pred: ^bb2
  %44 = llvm.fdiv %10, %20  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %45 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %46 = llvm.load %45 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %47 = llvm.fdiv %46, %20  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %48 = llvm.getelementptr inbounds %arg0[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %49 = llvm.load %48 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %50 = llvm.fdiv %49, %20  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %51 = llvm.intr.sin(%20)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
  %52 = llvm.intr.cos(%20)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
  %53 = llvm.fmul %44, %44  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %54 = llvm.fsub %4, %53  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %55 = llvm.fmul %54, %52  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %56 = llvm.fadd %55, %53  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %57 = llvm.getelementptr inbounds %arg1[%5, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %58 = llvm.load %57 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  llvm.store %56, %58 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %59 = llvm.fsub %4, %52  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %60 = llvm.fmul %59, %44  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %61 = llvm.fmul %60, %47  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %62 = llvm.fmul %50, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %63 = llvm.fsub %61, %62  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %64 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %65 = llvm.sext %64 : i32 to i64
  %66 = llvm.getelementptr inbounds %58[%65] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %63, %66 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %67 = llvm.fmul %60, %50  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %68 = llvm.fmul %47, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %69 = llvm.fadd %67, %68  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %70 = llvm.shl %64, %7 overflow<nsw>  : i32
  %71 = llvm.sext %70 : i32 to i64
  %72 = llvm.getelementptr inbounds %58[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %69, %72 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %73 = llvm.fadd %61, %62  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %74 = llvm.getelementptr inbounds %58[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %73, %74 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %75 = llvm.fmul %47, %47  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %76 = llvm.fsub %4, %75  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %77 = llvm.fmul %76, %52  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %78 = llvm.fadd %77, %75  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %79 = llvm.getelementptr %66[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %78, %79 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %80 = llvm.fmul %47, %59  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %81 = llvm.fmul %80, %50  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %82 = llvm.fmul %44, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %83 = llvm.fsub %81, %82  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %84 = llvm.or %70, %7  : i32
  %85 = llvm.sext %84 : i32 to i64
  %86 = llvm.getelementptr inbounds %58[%85] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %83, %86 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %87 = llvm.fsub %67, %68  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %88 = llvm.getelementptr inbounds %58[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %87, %88 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %89 = llvm.fadd %81, %82  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %90 = llvm.getelementptr %66[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %89, %90 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %91 = llvm.fmul %50, %50  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %92 = llvm.fsub %4, %91  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %93 = llvm.fmul %92, %52  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %94 = llvm.fadd %93, %91  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %95 = llvm.getelementptr %72[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %94, %95 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  llvm.br ^bb10
^bb10:  // 3 preds: ^bb3, ^bb8, ^bb9
  llvm.return
}
