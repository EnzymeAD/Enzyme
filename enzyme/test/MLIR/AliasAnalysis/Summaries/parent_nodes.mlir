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
llvm.func local_unnamed_addr @calloc(i64 {llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, passthrough = ["nofree", "nounwind", "willreturn", ["allockind", "17"], ["allocsize", "1"], ["alloc-family", "malloc"]], sym_visibility = "private"}
llvm.func local_unnamed_addr @free(!llvm.ptr {llvm.allocptr, llvm.nocapture, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nounwind", "willreturn", ["allockind", "4"], ["alloc-family", "malloc"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], sym_visibility = "private", target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>}

llvm.func @euler_angles_to_rotation_matrix(!llvm.ptr, !llvm.ptr) attributes {
  p2psummary = [
    [distinct[30]<"arg-euler_angles_to_rotation_matrix-1">, [distinct[31]<"arg-euler_angles_to_rotation_matrix-1-deref">, distinct[32]<"fresh-euler_angles_malloc">]],
    [distinct[32]<"fresh-euler_angles_malloc">, []],
    [distinct[33]<"fresh-malloc_RX">, []],
    [distinct[34]<"fresh-malloc_RY">, []],
    [distinct[35]<"fresh-malloc_RZ">, []],
    [distinct[36]<"fresh-malloc_tmp">, []]
  ]
}

llvm.func @angle_axis_to_rotation_matrix(!llvm.ptr, !llvm.ptr) attributes {
  p2psummary = [[distinct[40]<"arg-angle_axis_to_rotation_matrix-1">, [distinct[41]<"arg-angle_axis_to_rotation_matrix-1-deref">]]]
}

llvm.func @relatives_to_absolutes(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {
  p2psummary = [
    [distinct[50]<"arg-relatives_to_absolutes-1">, [distinct[51]<"arg-relatives_to_absolutes-1-deref">]],
    [distinct[52]<"arg-relatives_to_absolutes-3">, [distinct[53]<"arg-relatives_to_absolutes-3-deref">, distinct[54]<"fresh-rta1">, distinct[55]<"fresh-rta2">]],
    [distinct[54]<"fresh-rta1">, []],
    [distinct[55]<"fresh-rta2">, []]
  ]
}

llvm.func @to_pose_params(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {
  p2psummary = [
    [distinct[60]<"arg-to_pose_params-3">, [distinct[61]<"arg-to_pose_params-3-deref">, distinct[62]<"fresh-pose_params_data">]],
    [distinct[62]<"fresh-pose_params_data">, []]
  ]
}

// CHECK-LABEL: processing function @get_posed_relatives
// CHECK: p2p summary:
// CHECK-NEXT:    distinct[0]<"arg-get_posed_relatives-1"> -> [distinct[0]<"arg-get_posed_relatives-1-deref">]
// CHECK-NEXT:    distinct[0]<"arg-get_posed_relatives-2"> -> [distinct[0]<"arg-get_posed_relatives-2-deref">]
// CHECK-NEXT:    distinct[0]<"arg-get_posed_relatives-3"> -> [distinct[0]<"arg-get_posed_relatives-3-deref">, distinct[1]<"fresh-malloc4">]
// CHECK-NEXT:    distinct[0]<"fresh-malloc1"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-malloc2"> -> [distinct[0]<"fresh-euler_angles_malloc">, distinct[1]<"fresh-malloc3">]
// CHECK-NEXT:    distinct[0]<"fresh-malloc3"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-malloc4"> -> []
llvm.func local_unnamed_addr @get_posed_relatives(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
  %0 = llvm.mlir.constant(128 : i64) : i64
  %1 = llvm.mlir.constant(16 : i64) : i64
  %2 = llvm.mlir.constant(3 : i32) : i32
  %3 = llvm.mlir.constant(0 : i64) : i64
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.mlir.constant(72 : i64) : i64
  %6 = llvm.mlir.constant(2 : i32) : i32
  %7 = llvm.mlir.constant(0 : i32) : i32
  %8 = llvm.mlir.constant(2 : i64) : i64
  %9 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %10 = llvm.mlir.constant(0.000000e+00 : f64) : f64
  %11 = llvm.mlir.constant(1 : i64) : i64
  %12 = llvm.mlir.constant(4 : i64) : i64
  %13 = llvm.mlir.constant(3 : i64) : i64
  %14 = llvm.mlir.constant(5 : i64) : i64
  %15 = llvm.mlir.zero {nullptr} : !llvm.ptr
  %16 = llvm.mlir.constant(4 : i32) : i32
  %17 = llvm.call @malloc(%0) {tag = "malloc1"} : (i64) -> !llvm.ptr
  %18 = llvm.call @malloc(%1) {tag = "malloc2"} : (i64) -> !llvm.ptr
  llvm.store %2, %18 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
  %19 = llvm.getelementptr inbounds %18[%3, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.store %2, %19 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
  %20 = llvm.call @malloc(%5) {tag = "malloc3"} : (i64) -> !llvm.ptr
  %21 = llvm.getelementptr inbounds %18[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.store %20, %21 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  %22 = llvm.icmp "sgt" %arg0, %7 : i32
  llvm.cond_br %22, ^bb1, ^bb25
^bb1:  // pred: ^bb0
  %23 = llvm.getelementptr inbounds %arg2[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %24 = llvm.load %23 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %25 = llvm.load %arg2 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %26 = llvm.sext %25 : i32 to i64
  %27 = llvm.zext %arg0 : i32 to i64
  llvm.br ^bb2(%3 : i64)
^bb2(%28: i64):  // 2 preds: ^bb1, ^bb24
  llvm.br ^bb3(%3 : i64)
^bb3(%29: i64):  // 2 preds: ^bb2, ^bb5
  %30 = llvm.shl %29, %8  : i64
  %31 = llvm.getelementptr %17[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb4(%3 : i64)
^bb4(%32: i64):  // 2 preds: ^bb3, ^bb4
  %33 = llvm.icmp "eq" %29, %32 : i64
  %34 = llvm.select %33, %9, %10 : i1, f64
  %35 = llvm.getelementptr %31[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %34, %35 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %36 = llvm.add %32, %11 overflow<nsw, nuw>  : i64
  %37 = llvm.icmp "eq" %36, %12 : i64
  llvm.cond_br %37, ^bb5, ^bb4(%36 : i64) {loop_annotation = #loop_annotation}
^bb5:  // pred: ^bb4
  %38 = llvm.add %29, %11 overflow<nsw, nuw>  : i64
  %39 = llvm.icmp "eq" %38, %12 : i64
  llvm.cond_br %39, ^bb6, ^bb3(%38 : i64) {loop_annotation = #loop_annotation}
^bb6:  // pred: ^bb5
  %40 = llvm.add %28, %13 overflow<nsw, nuw>  : i64
  %41 = llvm.mul %40, %26 overflow<nsw>  : i64
  %42 = llvm.getelementptr inbounds %24[%41] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.call @euler_angles_to_rotation_matrix(%42, %18) : (!llvm.ptr, !llvm.ptr) -> ()
  %43 = llvm.load %19 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %44 = llvm.icmp "sgt" %43, %7 : i32
  llvm.cond_br %44, ^bb7, ^bb11
^bb7:  // pred: ^bb6
  %45 = llvm.load %18 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %46 = llvm.icmp "sgt" %45, %7 : i32
  %47 = llvm.sext %45 : i32 to i64
  %48 = llvm.zext %43 : i32 to i64
  %49 = llvm.zext %45 : i32 to i64
  %50 = llvm.shl %49, %13 overflow<nsw, nuw>  : i64
  llvm.br ^bb8(%3 : i64)
^bb8(%51: i64):  // 2 preds: ^bb7, ^bb10
  llvm.cond_br %46, ^bb9, ^bb10
^bb9:  // pred: ^bb8
  %52 = llvm.shl %51, %14 overflow<nsw, nuw>  : i64
  %53 = llvm.getelementptr %17[%52] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %54 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %55 = llvm.mul %51, %47 overflow<nsw>  : i64
  %56 = llvm.getelementptr %54[%55] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  "llvm.intr.memcpy"(%53, %56, %50) <{isVolatile = false, tbaa = [#tbaa_tag1]}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.br ^bb10
^bb10:  // 2 preds: ^bb8, ^bb9
  %57 = llvm.add %51, %11 overflow<nsw, nuw>  : i64
  %58 = llvm.icmp "eq" %57, %48 : i64
  llvm.cond_br %58, ^bb11, ^bb8(%57 : i64) {loop_annotation = #loop_annotation}
^bb11:  // 2 preds: ^bb6, ^bb10
  %59 = llvm.getelementptr inbounds %arg1[%28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %60 = llvm.getelementptr inbounds %arg3[%28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.intr.experimental.noalias.scope.decl #alias_scope12
  llvm.intr.experimental.noalias.scope.decl #alias_scope13
  %61 = llvm.load %59 {alias_scopes = [#alias_scope12], alignment = 8 : i64, noalias_scopes = [#alias_scope14, #alias_scope13], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %62 = llvm.load %60 {alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %63 = llvm.getelementptr inbounds %arg3[%28, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %64 = llvm.load %63 {alias_scopes = [#alias_scope13], alignment = 4 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %65 = llvm.mul %64, %62 overflow<nsw>  : i32
  %66 = llvm.shl %61, %6 overflow<nsw>  : i32
  %67 = llvm.icmp "eq" %65, %66 : i32
  llvm.cond_br %67, ^bb17, ^bb12
^bb12:  // pred: ^bb11
  %68 = llvm.getelementptr inbounds %arg3[%28, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %69 = llvm.load %68 {alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %70 = llvm.icmp "eq" %69, %15 : !llvm.ptr
  llvm.cond_br %70, ^bb14, ^bb13
^bb13:  // pred: ^bb12
  llvm.call @free(%69) {noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13]} : (!llvm.ptr) -> ()
  llvm.br ^bb14
^bb14:  // 2 preds: ^bb12, ^bb13
  %71 = llvm.icmp "sgt" %61, %7 : i32
  llvm.cond_br %71, ^bb15, ^bb16(%15 : !llvm.ptr)
^bb15:  // pred: ^bb14
  %72 = llvm.zext %66 : i32 to i64
  %73 = llvm.shl %72, %13 overflow<nsw, nuw>  : i64
  %74 = llvm.call @malloc(%73) {tag = "malloc4"} : (i64) -> !llvm.ptr
  llvm.br ^bb16(%74 : !llvm.ptr)
^bb16(%75: !llvm.ptr):  // 2 preds: ^bb14, ^bb15
  llvm.store %75, %68 {debugme, alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb17
^bb17:  // 2 preds: ^bb11, ^bb16
  llvm.store %16, %63 {alias_scopes = [#alias_scope13], alignment = 4 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
  llvm.store %61, %60 {bookmark, alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
  %76 = llvm.icmp "sgt" %61, %7 : i32
  llvm.cond_br %76, ^bb18, ^bb24
^bb18:  // pred: ^bb17
  %77 = llvm.getelementptr inbounds %arg1[%28, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %78 = llvm.getelementptr inbounds %arg3[%28, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %79 = llvm.getelementptr inbounds %arg1[%28, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %80 = llvm.zext %61 : i32 to i64
  %81 = llvm.load %77 {alias_scopes = [#alias_scope12], alignment = 8 : i64, noalias_scopes = [#alias_scope14, #alias_scope13], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %82 = llvm.load %78 {alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %83 = llvm.load %79 {alias_scopes = [#alias_scope12], alignment = 4 : i64, noalias_scopes = [#alias_scope14, #alias_scope13], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %84 = llvm.icmp "sgt" %83, %4 : i32
  %85 = llvm.zext %83 : i32 to i64
  llvm.br ^bb19(%3 : i64)
^bb19(%86: i64):  // 2 preds: ^bb18, ^bb23
  %87 = llvm.getelementptr inbounds %81[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %88 = llvm.getelementptr %82[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb20(%3 : i64)
^bb20(%89: i64):  // 2 preds: ^bb19, ^bb22
  %90 = llvm.load %87 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %91 = llvm.shl %89, %8 overflow<nsw>  : i64
  %92 = llvm.getelementptr inbounds %17[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %93 = llvm.load %92 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %94 = llvm.fmul %93, %90  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %95 = llvm.mul %89, %80 overflow<nsw, nuw>  : i64
  %96 = llvm.getelementptr %88[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %94, %96 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  llvm.cond_br %84, ^bb21(%11, %94 : i64, f64), ^bb22
^bb21(%97: i64, %98: f64):  // 2 preds: ^bb20, ^bb21
  %99 = llvm.mul %97, %80 overflow<nsw, nuw>  : i64
  %100 = llvm.getelementptr %87[%99] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %101 = llvm.load %100 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %102 = llvm.getelementptr %92[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %103 = llvm.load %102 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %104 = llvm.fmul %103, %101  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %105 = llvm.fadd %104, %98  {fastmathFlags = #llvm.fastmath<fast>} : f64
  llvm.store %105, %96 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %106 = llvm.add %97, %11 overflow<nsw, nuw>  : i64
  %107 = llvm.icmp "eq" %106, %85 : i64
  llvm.cond_br %107, ^bb22, ^bb21(%106, %105 : i64, f64) {loop_annotation = #loop_annotation}
^bb22:  // 2 preds: ^bb20, ^bb21
  %108 = llvm.add %89, %11 overflow<nsw, nuw>  : i64
  %109 = llvm.icmp "eq" %108, %12 : i64
  llvm.cond_br %109, ^bb23, ^bb20(%108 : i64) {loop_annotation = #loop_annotation}
^bb23:  // pred: ^bb22
  %110 = llvm.add %86, %11 overflow<nsw, nuw>  : i64
  %111 = llvm.icmp "eq" %110, %80 : i64
  llvm.cond_br %111, ^bb24, ^bb19(%110 : i64) {loop_annotation = #loop_annotation}
^bb24:  // 2 preds: ^bb17, ^bb23
  %112 = llvm.add %28, %11 overflow<nsw, nuw>  : i64
  %113 = llvm.icmp "eq" %112, %27 : i64
  llvm.cond_br %113, ^bb25, ^bb2(%112 : i64) {loop_annotation = #loop_annotation}
^bb25:  // 2 preds: ^bb0, ^bb24
  %114 = llvm.icmp "eq" %17, %15 : !llvm.ptr
  llvm.cond_br %114, ^bb27, ^bb26
^bb26:  // pred: ^bb25
  llvm.call @free(%17) : (!llvm.ptr) -> ()
  llvm.br ^bb27
^bb27:  // 2 preds: ^bb25, ^bb26
  %115 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %116 = llvm.icmp "eq" %115, %15 : !llvm.ptr
  llvm.cond_br %116, ^bb29, ^bb28
^bb28:  // pred: ^bb27
  llvm.call @free(%115) : (!llvm.ptr) -> ()
  llvm.br ^bb29
^bb29:  // 2 preds: ^bb27, ^bb28
  llvm.call @free(%18) : (!llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: processing function @apply_global_transform
// CHECK: p2p summary:
// CHECK-NEXT:    distinct[0]<"arg-apply_global_transform-0"> -> [distinct[0]<"arg-apply_global_transform-0-deref">]
// CHECK-NEXT:    distinct[0]<"arg-apply_global_transform-1"> -> [distinct[0]<"arg-apply_global_transform-1-deref">]
// CHECK-NEXT:    distinct[0]<"fresh-agt_tmp"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-rmat"> -> [distinct[0]<"fresh-rmat_data">]
// CHECK-NEXT:    distinct[0]<"fresh-rmat_data"> -> []
llvm.func local_unnamed_addr @apply_global_transform(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.mlir.constant(3 : i32) : i32
  %2 = llvm.mlir.constant(0 : i64) : i64
  %3 = llvm.mlir.constant(1 : i32) : i32
  %4 = llvm.mlir.constant(72 : i64) : i64
  %5 = llvm.mlir.constant(2 : i32) : i32
  %6 = llvm.mlir.constant(3 : i64) : i64
  %7 = llvm.mlir.constant(1 : i64) : i64
  %8 = llvm.mlir.constant(0 : i32) : i32
  %9 = llvm.mlir.zero : !llvm.ptr
  %10 = llvm.call @malloc(%0) {tag = "rmat"} : (i64) -> !llvm.ptr
  llvm.store %1, %10 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
  %11 = llvm.getelementptr inbounds %10[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.store %1, %11 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
  %12 = llvm.call @malloc(%4) {tag = "rmat_data"} : (i64) -> !llvm.ptr
  %13 = llvm.getelementptr inbounds %10[%2, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.store %12, %13 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  %14 = llvm.getelementptr inbounds %arg0[%2, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  llvm.call @angle_axis_to_rotation_matrix(%15, %10) : (!llvm.ptr, !llvm.ptr) -> ()
  %16 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %17 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %18 = llvm.sext %17 : i32 to i64
  %19 = llvm.getelementptr %16[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb1(%2 : i64)
^bb1(%20: i64):  // 2 preds: ^bb0, ^bb3
  %21 = llvm.getelementptr %19[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %22 = llvm.mul %20, %6 overflow<nsw, nuw>  : i64
  %23 = llvm.getelementptr %12[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb2(%2 : i64)
^bb2(%24: i64):  // 2 preds: ^bb1, ^bb2
  %25 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %26 = llvm.getelementptr %23[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %27 = llvm.load %26 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %28 = llvm.fmul %27, %25  {fastmathFlags = #llvm.fastmath<fast>} : f64
  llvm.store %28, %26 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %29 = llvm.add %24, %7 overflow<nsw, nuw>  : i64
  %30 = llvm.icmp "eq" %29, %6 : i64
  llvm.cond_br %30, ^bb3, ^bb2(%29 : i64) {loop_annotation = #loop_annotation}
^bb3:  // pred: ^bb2
  %31 = llvm.add %20, %7 overflow<nsw, nuw>  : i64
  %32 = llvm.icmp "eq" %31, %6 : i64
  llvm.cond_br %32, ^bb4, ^bb1(%31 : i64) {loop_annotation = #loop_annotation}
^bb4:  // pred: ^bb3
  llvm.intr.experimental.noalias.scope.decl #alias_scope
  %33 = llvm.getelementptr inbounds %arg1[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %34 = llvm.load %33 {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %35 = llvm.icmp "sgt" %34, %8 : i32
  llvm.cond_br %35, ^bb5, ^bb6(%9 : !llvm.ptr)
^bb5:  // pred: ^bb4
  %36 = llvm.mul %34, %1 overflow<nsw>  : i32
  %37 = llvm.zext %36 : i32 to i64
  %38 = llvm.shl %37, %6 overflow<nsw, nuw>  : i64
  %39 = llvm.call @malloc(%38) {tag = "agt_tmp"} : (i64) -> !llvm.ptr
  llvm.br ^bb6(%39 : !llvm.ptr)
^bb6(%40: !llvm.ptr):  // 2 preds: ^bb4, ^bb5
  %41 = llvm.icmp "sgt" %34, %8 : i32
  %42 = llvm.getelementptr inbounds %arg1[%2, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %43 = llvm.zext %34 : i32 to i64
  llvm.br ^bb7(%2 : i64)
^bb7(%44: i64):  // 2 preds: ^bb6, ^bb12
  llvm.cond_br %41, ^bb8, ^bb12
^bb8:  // pred: ^bb7
  %45 = llvm.getelementptr inbounds %12[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %46 = llvm.load %42 {alias_scopes = [#alias_scope], alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %47 = llvm.load %arg1 {alias_scopes = [#alias_scope], alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %48 = llvm.sext %47 : i32 to i64
  %49 = llvm.getelementptr %40[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %50 = llvm.load %45 {alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  llvm.br ^bb9(%2 : i64)
^bb9(%51: i64):  // 2 preds: ^bb8, ^bb11
  %52 = llvm.mul %51, %48 overflow<nsw>  : i64
  %53 = llvm.getelementptr inbounds %46[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %54 = llvm.load %53 {alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %55 = llvm.fmul %54, %50  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %56 = llvm.mul %51, %6 overflow<nsw, nuw>  : i64
  %57 = llvm.getelementptr %49[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %55, %57 {alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  llvm.br ^bb10(%7, %55 : i64, f64)
^bb10(%58: i64, %59: f64):  // 2 preds: ^bb9, ^bb10
  %60 = llvm.mul %58, %6 overflow<nsw, nuw>  : i64
  %61 = llvm.getelementptr %45[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %62 = llvm.load %61 {alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %63 = llvm.getelementptr %53[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %64 = llvm.load %63 {alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %65 = llvm.fmul %64, %62  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %66 = llvm.fadd %65, %59  {fastmathFlags = #llvm.fastmath<fast>} : f64
  llvm.store %66, %57 {alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %67 = llvm.add %58, %7 overflow<nsw, nuw>  : i64
  %68 = llvm.icmp "eq" %67, %6 : i64
  llvm.cond_br %68, ^bb11, ^bb10(%67, %66 : i64, f64) {loop_annotation = #loop_annotation}
^bb11:  // pred: ^bb10
  %69 = llvm.add %51, %7 overflow<nsw, nuw>  : i64
  %70 = llvm.icmp "eq" %69, %43 : i64
  llvm.cond_br %70, ^bb12, ^bb9(%69 : i64) {loop_annotation = #loop_annotation}
^bb12:  // 2 preds: ^bb7, ^bb11
  %71 = llvm.add %44, %7 overflow<nsw, nuw>  : i64
  %72 = llvm.icmp "eq" %71, %6 : i64
  llvm.cond_br %72, ^bb13, ^bb7(%71 : i64) {loop_annotation = #loop_annotation}
^bb13:  // pred: ^bb12
  %73 = llvm.load %33 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %74 = llvm.icmp "sgt" %73, %8 : i32
  llvm.cond_br %74, ^bb14, ^bb19
^bb14:  // pred: ^bb13
  %75 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %76 = llvm.icmp "sgt" %75, %8 : i32
  %77 = llvm.sext %75 : i32 to i64
  %78 = llvm.zext %73 : i32 to i64
  %79 = llvm.shl %17, %3 overflow<nsw>  : i32
  %80 = llvm.sext %79 : i32 to i64
  %81 = llvm.zext %75 : i32 to i64
  %82 = llvm.getelementptr %16[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb15(%2 : i64)
^bb15(%83: i64):  // 2 preds: ^bb14, ^bb18
  llvm.cond_br %76, ^bb16, ^bb18
^bb16:  // pred: ^bb15
  %84 = llvm.mul %83, %6 overflow<nsw, nuw>  : i64
  %85 = llvm.load %42 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %86 = llvm.mul %83, %77 overflow<nsw>  : i64
  %87 = llvm.getelementptr %85[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb17(%2 : i64)
^bb17(%88: i64):  // 2 preds: ^bb16, ^bb17
  %89 = llvm.add %88, %84 overflow<nsw, nuw>  : i64
  %90 = llvm.getelementptr inbounds %40[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %91 = llvm.load %90 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %92 = llvm.getelementptr %82[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %93 = llvm.load %92 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %94 = llvm.fadd %93, %91  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %95 = llvm.getelementptr %87[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %94, %95 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %96 = llvm.add %88, %7 overflow<nsw, nuw>  : i64
  %97 = llvm.icmp "eq" %96, %81 : i64
  llvm.cond_br %97, ^bb18, ^bb17(%96 : i64) {loop_annotation = #loop_annotation}
^bb18:  // 2 preds: ^bb15, ^bb17
  %98 = llvm.add %83, %7 overflow<nsw, nuw>  : i64
  %99 = llvm.icmp "eq" %98, %78 : i64
  llvm.cond_br %99, ^bb19, ^bb15(%98 : i64) {loop_annotation = #loop_annotation}
^bb19:  // 2 preds: ^bb13, ^bb18
  %100 = llvm.icmp "eq" %12, %9 : !llvm.ptr
  llvm.cond_br %100, ^bb21, ^bb20
^bb20:  // pred: ^bb19
  llvm.call @free(%12) : (!llvm.ptr) -> ()
  llvm.br ^bb21
^bb21:  // 2 preds: ^bb19, ^bb20
  llvm.call @free(%10) : (!llvm.ptr) -> ()
  %101 = llvm.icmp "eq" %40, %9 : !llvm.ptr
  llvm.cond_br %101, ^bb23, ^bb22
^bb22:  // pred: ^bb21
  llvm.call @free(%40) : (!llvm.ptr) -> ()
  llvm.br ^bb23
^bb23:  // 2 preds: ^bb21, ^bb22
  llvm.return
}

// CHECK-LABEL: processing function @get_skinned_vertex_positions
// CHECK: p2p summary:
// CHECK-NEXT:    distinct[0]<"arg-get_skinned_vertex_positions-3"> -> [distinct[0]<"arg-get_skinned_vertex_positions-3-deref">]
// CHECK-NEXT:    distinct[0]<"arg-get_skinned_vertex_positions-4"> -> [distinct[0]<"arg-get_skinned_vertex_positions-4-deref">]
// CHECK-NEXT:    distinct[0]<"arg-get_skinned_vertex_positions-5"> -> [distinct[0]<"arg-get_skinned_vertex_positions-5-deref">]
// CHECK-NEXT:    distinct[0]<"arg-get_skinned_vertex_positions-8"> -> [distinct[0]<"arg-get_skinned_vertex_positions-8-deref">, distinct[1]<"fresh-positions_data">]
// CHECK-NEXT:    distinct[0]<"fresh-absolutes"> -> [distinct[0]<"fresh-rta1">, distinct[1]<"fresh-rta2">]
// CHECK-NEXT:    distinct[0]<"fresh-absolutes_empty"> -> [distinct[0]<"fresh-rta1">, distinct[1]<"fresh-rta2">]
// CHECK-NEXT:    distinct[0]<"fresh-curr_pos"> -> [distinct[0]<"fresh-curr_pos_data">, distinct[1]<"fresh-curr_pos_resize">]
// CHECK-NEXT:    distinct[0]<"fresh-curr_pos_data"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-curr_pos_resize"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-positions_data"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-relatives"> -> [distinct[0]<"fresh-malloc4">]
// CHECK-NEXT:    distinct[0]<"fresh-transforms"> -> [distinct[0]<"fresh-transforms_data">]
// CHECK-NEXT:    distinct[0]<"fresh-transforms_data"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-transforms_empty"> -> [distinct[0]<"fresh-transforms_data">]
llvm.func local_unnamed_addr @get_skinned_vertex_positions(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: i32 {llvm.noundef}, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}, %arg9: i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
  %0 = llvm.mlir.constant(4 : i64) : i64
  %1 = llvm.mlir.constant(0 : i32) : i32
  %2 = llvm.mlir.constant(0 : i8) : i8
  %3 = llvm.mlir.constant(0 : i64) : i64
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.mlir.constant(2 : i32) : i32
  %6 = llvm.mlir.zero : !llvm.ptr
  %7 = llvm.mlir.constant(3 : i64) : i64
  %8 = llvm.mlir.constant(1 : i64) : i64
  %9 = llvm.mlir.constant(3 : i32) : i32
  %10 = llvm.mlir.constant(16 : i64) : i64
  %11 = llvm.mlir.constant(4 : i32) : i32
  %12 = llvm.sext %arg0 : i32 to i64
  %13 = llvm.shl %12, %0 overflow<nsw>  : i64
  %14 = llvm.call @malloc(%13) {tag = "relatives"} : (i64) -> !llvm.ptr
  %15 = llvm.icmp "sgt" %arg0, %1 : i32
  llvm.cond_br %15, ^bb2, ^bb1
^bb1:  // pred: ^bb0
  %16 = llvm.call @malloc(%13) {tag = "absolutes_empty"} : (i64) -> !llvm.ptr
  %17 = llvm.call @malloc(%13) {tag = "transforms_empty"} : (i64) -> !llvm.ptr
  llvm.br ^bb3(%17, %16 : !llvm.ptr, !llvm.ptr)
^bb2:  // pred: ^bb0
  %18 = llvm.zext %arg0 : i32 to i64
  %19 = llvm.shl %18, %0 overflow<nsw, nuw>  : i64
  "llvm.intr.memset"(%14, %2, %19) <{isVolatile = false, tbaa = [#tbaa_tag]}> : (!llvm.ptr, i8, i64) -> ()
  %20 = llvm.call @malloc(%13) {tag = "absolutes"} : (i64) -> !llvm.ptr
  "llvm.intr.memset"(%20, %2, %19) <{isVolatile = false, tbaa = [#tbaa_tag]}> : (!llvm.ptr, i8, i64) -> ()
  %21 = llvm.call @malloc(%13) {tag = "transforms"} : (i64) -> !llvm.ptr
  "llvm.intr.memset"(%21, %2, %19) <{isVolatile = false, tbaa = [#tbaa_tag]}> : (!llvm.ptr, i8, i64) -> ()
  llvm.br ^bb3(%21, %20 : !llvm.ptr, !llvm.ptr)
^bb3(%22: !llvm.ptr, %23: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
  llvm.call @get_posed_relatives(%arg0, %arg1, %arg7, %14) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.call @relatives_to_absolutes(%arg0, %14, %arg2, %23) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.cond_br %15, ^bb4, ^bb20
^bb4:  // pred: ^bb3
  %24 = llvm.zext %arg0 : i32 to i64
  llvm.br ^bb5(%3 : i64)
^bb5(%25: i64):  // 2 preds: ^bb4, ^bb19
  %26 = llvm.getelementptr inbounds %23[%25] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %27 = llvm.getelementptr inbounds %arg3[%25] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %28 = llvm.getelementptr inbounds %22[%25] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.intr.experimental.noalias.scope.decl #alias_scope15
  llvm.intr.experimental.noalias.scope.decl #alias_scope16
  llvm.intr.experimental.noalias.scope.decl #alias_scope17
  %29 = llvm.load %26 {alias_scopes = [#alias_scope15], alignment = 8 : i64, noalias_scopes = [#alias_scope16, #alias_scope17], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %30 = llvm.getelementptr inbounds %arg3[%25, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %31 = llvm.load %30 {alias_scopes = [#alias_scope16], alignment = 4 : i64, noalias_scopes = [#alias_scope15, #alias_scope17], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %32 = llvm.load %28 {alias_scopes = [#alias_scope17], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %33 = llvm.getelementptr inbounds %22[%25, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %34 = llvm.load %33 {alias_scopes = [#alias_scope17], alignment = 4 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %35 = llvm.mul %34, %32 overflow<nsw>  : i32
  %36 = llvm.mul %31, %29 overflow<nsw>  : i32
  %37 = llvm.icmp "eq" %35, %36 : i32
  llvm.cond_br %37, ^bb11, ^bb6
^bb6:  // pred: ^bb5
  %38 = llvm.getelementptr inbounds %22[%25, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %39 = llvm.load %38 {alias_scopes = [#alias_scope17], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %40 = llvm.icmp "eq" %39, %6 : !llvm.ptr
  llvm.cond_br %40, ^bb8, ^bb7
^bb7:  // pred: ^bb6
  llvm.call @free(%39) {noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17]} : (!llvm.ptr) -> ()
  llvm.br ^bb8
^bb8:  // 2 preds: ^bb6, ^bb7
  %41 = llvm.icmp "sgt" %36, %1 : i32
  llvm.cond_br %41, ^bb9, ^bb10(%6 : !llvm.ptr)
^bb9:  // pred: ^bb8
  %42 = llvm.zext %36 : i32 to i64
  %43 = llvm.shl %42, %7 overflow<nsw, nuw>  : i64
  %44 = llvm.call @malloc(%43) {tag = "transforms_data"} : (i64) -> !llvm.ptr
  llvm.br ^bb10(%44 : !llvm.ptr)
^bb10(%45: !llvm.ptr):  // 2 preds: ^bb8, ^bb9
  llvm.store %45, %38 {alias_scopes = [#alias_scope17], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb11
^bb11:  // 2 preds: ^bb5, ^bb10
  llvm.store %31, %33 {alias_scopes = [#alias_scope17], alignment = 4 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
  llvm.store %29, %28 {alias_scopes = [#alias_scope17], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
  %46 = llvm.icmp "sgt" %29, %1 : i32
  llvm.cond_br %46, ^bb12, ^bb19
^bb12:  // pred: ^bb11
  %47 = llvm.icmp "sgt" %31, %1 : i32
  %48 = llvm.getelementptr inbounds %23[%25, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %49 = llvm.getelementptr inbounds %arg3[%25, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %50 = llvm.getelementptr inbounds %22[%25, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %51 = llvm.getelementptr inbounds %23[%25, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %52 = llvm.zext %29 : i32 to i64
  %53 = llvm.zext %31 : i32 to i64
  llvm.br ^bb13(%3 : i64)
^bb13(%54: i64):  // 2 preds: ^bb12, ^bb18
  llvm.cond_br %47, ^bb14, ^bb18
^bb14:  // pred: ^bb13
  %55 = llvm.load %48 {alias_scopes = [#alias_scope15], alignment = 8 : i64, noalias_scopes = [#alias_scope16, #alias_scope17], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %56 = llvm.getelementptr inbounds %55[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %57 = llvm.load %49 {alias_scopes = [#alias_scope16], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope17], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %58 = llvm.load %27 {alias_scopes = [#alias_scope16], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope17], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %59 = llvm.load %50 {alias_scopes = [#alias_scope17], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %60 = llvm.load %51 {alias_scopes = [#alias_scope15], alignment = 4 : i64, noalias_scopes = [#alias_scope16, #alias_scope17], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %61 = llvm.icmp "sgt" %60, %4 : i32
  %62 = llvm.sext %58 : i32 to i64
  %63 = llvm.getelementptr %59[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %64 = llvm.zext %60 : i32 to i64
  llvm.br ^bb15(%3 : i64)
^bb15(%65: i64):  // 2 preds: ^bb14, ^bb17
  %66 = llvm.load %56 {alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %67 = llvm.mul %65, %62 overflow<nsw>  : i64
  %68 = llvm.getelementptr inbounds %57[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %69 = llvm.load %68 {alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %70 = llvm.fmul %69, %66  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %71 = llvm.mul %65, %52 overflow<nsw, nuw>  : i64
  %72 = llvm.getelementptr %63[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %70, %72 {alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  llvm.cond_br %61, ^bb16(%8, %70 : i64, f64), ^bb17
^bb16(%73: i64, %74: f64):  // 2 preds: ^bb15, ^bb16
  %75 = llvm.mul %73, %52 overflow<nsw, nuw>  : i64
  %76 = llvm.getelementptr %56[%75] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %77 = llvm.load %76 {alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %78 = llvm.getelementptr %68[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %79 = llvm.load %78 {alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %80 = llvm.fmul %79, %77  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %81 = llvm.fadd %80, %74  {fastmathFlags = #llvm.fastmath<fast>} : f64
  llvm.store %81, %72 {alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %82 = llvm.add %73, %8 overflow<nsw, nuw>  : i64
  %83 = llvm.icmp "eq" %82, %64 : i64
  llvm.cond_br %83, ^bb17, ^bb16(%82, %81 : i64, f64) {loop_annotation = #loop_annotation}
^bb17:  // 2 preds: ^bb15, ^bb16
  %84 = llvm.add %65, %8 overflow<nsw, nuw>  : i64
  %85 = llvm.icmp "eq" %84, %53 : i64
  llvm.cond_br %85, ^bb18, ^bb15(%84 : i64) {loop_annotation = #loop_annotation}
^bb18:  // 2 preds: ^bb13, ^bb17
  %86 = llvm.add %54, %8 overflow<nsw, nuw>  : i64
  %87 = llvm.icmp "eq" %86, %52 : i64
  llvm.cond_br %87, ^bb19, ^bb13(%86 : i64) {loop_annotation = #loop_annotation}
^bb19:  // 2 preds: ^bb11, ^bb18
  %88 = llvm.add %25, %8 overflow<nsw, nuw>  : i64
  %89 = llvm.icmp "eq" %88, %24 : i64
  llvm.cond_br %89, ^bb20, ^bb5(%88 : i64) {loop_annotation = #loop_annotation}
^bb20:  // 2 preds: ^bb3, ^bb19
  %90 = llvm.getelementptr inbounds %arg4[%3, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %91 = llvm.load %90 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %92 = llvm.load %arg8 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %93 = llvm.getelementptr inbounds %arg8[%3, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %94 = llvm.load %93 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %95 = llvm.mul %94, %92 overflow<nsw>  : i32
  %96 = llvm.mul %91, %9 overflow<nsw>  : i32
  %97 = llvm.icmp "eq" %95, %96 : i32
  llvm.cond_br %97, ^bb26, ^bb21
^bb21:  // pred: ^bb20
  %98 = llvm.getelementptr inbounds %arg8[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %99 = llvm.load %98 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %100 = llvm.icmp "eq" %99, %6 : !llvm.ptr
  llvm.cond_br %100, ^bb23, ^bb22
^bb22:  // pred: ^bb21
  llvm.call @free(%99) : (!llvm.ptr) -> ()
  llvm.br ^bb23
^bb23:  // 2 preds: ^bb21, ^bb22
  %101 = llvm.icmp "sgt" %91, %1 : i32
  llvm.cond_br %101, ^bb24, ^bb25(%6 : !llvm.ptr)
^bb24:  // pred: ^bb23
  %102 = llvm.zext %96 : i32 to i64
  %103 = llvm.shl %102, %7 overflow<nsw, nuw>  : i64
  %104 = llvm.call @malloc(%103) {tag = "positions_data"} : (i64) -> !llvm.ptr
  llvm.br ^bb25(%104 : !llvm.ptr)
^bb25(%105: !llvm.ptr):  // 2 preds: ^bb23, ^bb24
  llvm.store %105, %98 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb26
^bb26:  // 2 preds: ^bb20, ^bb25
  llvm.store %91, %93 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
  llvm.store %9, %arg8 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
  %106 = llvm.icmp "sgt" %91, %1 : i32
  llvm.cond_br %106, ^bb27, ^bb28
^bb27:  // pred: ^bb26
  %107 = llvm.getelementptr inbounds %arg8[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %108 = llvm.load %107 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %109 = llvm.zext %96 : i32 to i64
  %110 = llvm.shl %109, %7 overflow<nsw, nuw>  : i64
  "llvm.intr.memset"(%108, %2, %110) <{isVolatile = false, tbaa = [#tbaa_tag1]}> : (!llvm.ptr, i8, i64) -> ()
  llvm.br ^bb28
^bb28:  // 2 preds: ^bb26, ^bb27
  %111 = llvm.call @malloc(%10) {tag = "curr_pos"} : (i64) -> !llvm.ptr
  llvm.store %11, %111 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
  %112 = llvm.getelementptr inbounds %111[%3, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.store %91, %112 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
  %113 = llvm.shl %91, %5 overflow<nsw>  : i32
  %114 = llvm.sext %113 : i32 to i64
  %115 = llvm.shl %114, %7 overflow<nsw>  : i64
  %116 = llvm.call @malloc(%115) {tag = "curr_pos_data"} : (i64) -> !llvm.ptr
  %117 = llvm.getelementptr inbounds %111[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.store %116, %117 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  llvm.cond_br %15, ^bb29, ^bb50
^bb29:  // pred: ^bb28
  %118 = llvm.getelementptr inbounds %arg4[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %119 = llvm.zext %91 : i32 to i64
  %120 = llvm.getelementptr inbounds %arg5[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %121 = llvm.getelementptr inbounds %arg8[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %122 = llvm.zext %arg0 : i32 to i64
  llvm.br ^bb30(%116, %116, %3, %11 : !llvm.ptr, !llvm.ptr, i64, i32)
^bb30(%123: !llvm.ptr, %124: !llvm.ptr, %125: i64, %126: i32):  // 2 preds: ^bb29, ^bb49
  %127 = llvm.getelementptr inbounds %22[%125] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.intr.experimental.noalias.scope.decl #alias_scope18
  llvm.intr.experimental.noalias.scope.decl #alias_scope19
  llvm.intr.experimental.noalias.scope.decl #alias_scope20
  %128 = llvm.load %127 {alias_scopes = [#alias_scope18], alignment = 8 : i64, noalias_scopes = [#alias_scope19, #alias_scope20], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %129 = llvm.mul %91, %126 overflow<nsw>  : i32
  %130 = llvm.mul %128, %91 overflow<nsw>  : i32
  %131 = llvm.icmp "eq" %129, %130 : i32
  llvm.cond_br %131, ^bb36(%123, %124 : !llvm.ptr, !llvm.ptr), ^bb31
^bb31:  // pred: ^bb30
  %132 = llvm.icmp "eq" %124, %6 : !llvm.ptr
  llvm.cond_br %132, ^bb33, ^bb32
^bb32:  // pred: ^bb31
  llvm.call @free(%124) {noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20]} : (!llvm.ptr) -> ()
  llvm.br ^bb33
^bb33:  // 2 preds: ^bb31, ^bb32
  %133 = llvm.icmp "sgt" %130, %1 : i32
  llvm.cond_br %133, ^bb34, ^bb35(%6 : !llvm.ptr)
^bb34:  // pred: ^bb33
  %134 = llvm.zext %130 : i32 to i64
  %135 = llvm.shl %134, %7 overflow<nsw, nuw>  : i64
  %136 = llvm.call @malloc(%135) {tag = "curr_pos_resize"} : (i64) -> !llvm.ptr
  llvm.br ^bb35(%136 : !llvm.ptr)
^bb35(%137: !llvm.ptr):  // 2 preds: ^bb33, ^bb34
  llvm.store %137, %117 {alias_scopes = [#alias_scope20], alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb36(%137, %137 : !llvm.ptr, !llvm.ptr)
^bb36(%138: !llvm.ptr, %139: !llvm.ptr):  // 2 preds: ^bb30, ^bb35
  %140 = llvm.icmp "sgt" %128, %1 : i32
  llvm.cond_br %140, ^bb37, ^bb44(%139 : !llvm.ptr)
^bb37:  // pred: ^bb36
  %141 = llvm.getelementptr inbounds %22[%125, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %142 = llvm.getelementptr inbounds %22[%125, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %143 = llvm.zext %128 : i32 to i64
  llvm.br ^bb38(%139, %3 : !llvm.ptr, i64)
^bb38(%144: !llvm.ptr, %145: i64):  // 2 preds: ^bb37, ^bb43
  llvm.cond_br %106, ^bb39, ^bb43(%144 : !llvm.ptr)
^bb39:  // pred: ^bb38
  %146 = llvm.load %141 {alias_scopes = [#alias_scope18], alignment = 8 : i64, noalias_scopes = [#alias_scope19, #alias_scope20], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %147 = llvm.getelementptr inbounds %146[%145] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %148 = llvm.load %118 {alias_scopes = [#alias_scope19], alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope20], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %149 = llvm.load %arg4 {alias_scopes = [#alias_scope19], alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope20], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %150 = llvm.load %142 {alias_scopes = [#alias_scope18], alignment = 4 : i64, noalias_scopes = [#alias_scope19, #alias_scope20], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %151 = llvm.icmp "sgt" %150, %4 : i32
  %152 = llvm.sext %149 : i32 to i64
  %153 = llvm.getelementptr %138[%145] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %154 = llvm.zext %150 : i32 to i64
  llvm.br ^bb40(%3 : i64)
^bb40(%155: i64):  // 2 preds: ^bb39, ^bb42
  %156 = llvm.load %147 {alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %157 = llvm.mul %155, %152 overflow<nsw>  : i64
  %158 = llvm.getelementptr inbounds %148[%157] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %159 = llvm.load %158 {alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %160 = llvm.fmul %159, %156  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %161 = llvm.mul %155, %143 overflow<nsw, nuw>  : i64
  %162 = llvm.getelementptr %153[%161] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %160, %162 {alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  llvm.cond_br %151, ^bb41(%8, %160 : i64, f64), ^bb42
^bb41(%163: i64, %164: f64):  // 2 preds: ^bb40, ^bb41
  %165 = llvm.mul %163, %143 overflow<nsw, nuw>  : i64
  %166 = llvm.getelementptr %147[%165] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %167 = llvm.load %166 {alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %168 = llvm.getelementptr %158[%163] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %169 = llvm.load %168 {alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %170 = llvm.fmul %169, %167  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %171 = llvm.fadd %170, %164  {fastmathFlags = #llvm.fastmath<fast>} : f64
  llvm.store %171, %162 {alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %172 = llvm.add %163, %8 overflow<nsw, nuw>  : i64
  %173 = llvm.icmp "eq" %172, %154 : i64
  llvm.cond_br %173, ^bb42, ^bb41(%172, %171 : i64, f64) {loop_annotation = #loop_annotation}
^bb42:  // 2 preds: ^bb40, ^bb41
  %174 = llvm.add %155, %8 overflow<nsw, nuw>  : i64
  %175 = llvm.icmp "eq" %174, %119 : i64
  llvm.cond_br %175, ^bb43(%138 : !llvm.ptr), ^bb40(%174 : i64) {loop_annotation = #loop_annotation}
^bb43(%176: !llvm.ptr):  // 2 preds: ^bb38, ^bb42
  %177 = llvm.add %145, %8 overflow<nsw, nuw>  : i64
  %178 = llvm.icmp "eq" %177, %143 : i64
  llvm.cond_br %178, ^bb44(%176 : !llvm.ptr), ^bb38(%176, %177 : !llvm.ptr, i64) {loop_annotation = #loop_annotation}
^bb44(%179: !llvm.ptr):  // 2 preds: ^bb36, ^bb43
  llvm.cond_br %106, ^bb45, ^bb49(%138, %179 : !llvm.ptr, !llvm.ptr)
^bb45:  // pred: ^bb44
  %180 = llvm.load %117 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %181 = llvm.load %120 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %182 = llvm.load %arg5 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %183 = llvm.load %121 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %184 = llvm.sext %128 : i32 to i64
  %185 = llvm.sext %182 : i32 to i64
  %186 = llvm.getelementptr %181[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb46(%3 : i64)
^bb46(%187: i64):  // 2 preds: ^bb45, ^bb48
  %188 = llvm.mul %187, %184 overflow<nsw>  : i64
  %189 = llvm.mul %187, %185 overflow<nsw>  : i64
  %190 = llvm.getelementptr %186[%189] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %191 = llvm.mul %187, %7 overflow<nsw, nuw>  : i64
  %192 = llvm.getelementptr %180[%188] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %193 = llvm.getelementptr %183[%191] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb47(%3 : i64)
^bb47(%194: i64):  // 2 preds: ^bb46, ^bb47
  %195 = llvm.getelementptr %192[%194] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %196 = llvm.load %195 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %197 = llvm.load %190 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %198 = llvm.fmul %197, %196  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %199 = llvm.getelementptr %193[%194] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %200 = llvm.load %199 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %201 = llvm.fadd %200, %198  {fastmathFlags = #llvm.fastmath<fast>} : f64
  llvm.store %201, %199 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %202 = llvm.add %194, %8 overflow<nsw, nuw>  : i64
  %203 = llvm.icmp "eq" %202, %7 : i64
  llvm.cond_br %203, ^bb48, ^bb47(%202 : i64) {loop_annotation = #loop_annotation}
^bb48:  // pred: ^bb47
  %204 = llvm.add %187, %8 overflow<nsw, nuw>  : i64
  %205 = llvm.icmp "eq" %204, %119 : i64
  llvm.cond_br %205, ^bb49(%180, %180 : !llvm.ptr, !llvm.ptr), ^bb46(%204 : i64) {loop_annotation = #loop_annotation}
^bb49(%206: !llvm.ptr, %207: !llvm.ptr):  // 2 preds: ^bb44, ^bb48
  %208 = llvm.add %125, %8 overflow<nsw, nuw>  : i64
  %209 = llvm.icmp "eq" %208, %122 : i64
  llvm.cond_br %209, ^bb50, ^bb30(%206, %207, %208, %128 : !llvm.ptr, !llvm.ptr, i64, i32) {loop_annotation = #loop_annotation}
^bb50:  // 2 preds: ^bb28, ^bb49
  %210 = llvm.icmp "ne" %arg6, %1 : i32
  %211 = llvm.and %210, %106  : i1
  llvm.cond_br %211, ^bb51, ^bb53
^bb51:  // pred: ^bb50
  %212 = llvm.getelementptr inbounds %arg8[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %213 = llvm.load %212 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %214 = llvm.zext %91 : i32 to i64
  llvm.br ^bb52(%3 : i64)
^bb52(%215: i64):  // 2 preds: ^bb51, ^bb52
  %216 = llvm.mul %215, %7 overflow<nsw, nuw>  : i64
  %217 = llvm.getelementptr inbounds %213[%216] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %218 = llvm.load %217 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %219 = llvm.fneg %218  {fastmathFlags = #llvm.fastmath<fast>} : f64
  llvm.store %219, %217 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %220 = llvm.add %215, %8 overflow<nsw, nuw>  : i64
  %221 = llvm.icmp "eq" %220, %214 : i64
  llvm.cond_br %221, ^bb53, ^bb52(%220 : i64) {loop_annotation = #loop_annotation}
^bb53:  // 2 preds: ^bb50, ^bb52
  %222 = llvm.icmp "eq" %arg9, %1 : i32
  llvm.cond_br %222, ^bb55, ^bb54
^bb54:  // pred: ^bb53
  llvm.call @apply_global_transform(%arg7, %arg8) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb55
^bb55:  // 2 preds: ^bb53, ^bb54
  %223 = llvm.load %117 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %224 = llvm.icmp "eq" %223, %6 : !llvm.ptr
  llvm.cond_br %224, ^bb57, ^bb56
^bb56:  // pred: ^bb55
  llvm.call @free(%223) : (!llvm.ptr) -> ()
  llvm.br ^bb57
^bb57:  // 2 preds: ^bb55, ^bb56
  llvm.call @free(%111) : (!llvm.ptr) -> ()
  llvm.cond_br %15, ^bb59, ^bb58
^bb58:  // pred: ^bb57
  llvm.call @free(%14) : (!llvm.ptr) -> ()
  llvm.call @free(%23) : (!llvm.ptr) -> ()
  llvm.br ^bb71
^bb59:  // pred: ^bb57
  %225 = llvm.zext %arg0 : i32 to i64
  llvm.br ^bb60(%3 : i64)
^bb60(%226: i64):  // 2 preds: ^bb59, ^bb62
  %227 = llvm.getelementptr inbounds %14[%226, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %228 = llvm.load %227 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %229 = llvm.icmp "eq" %228, %6 : !llvm.ptr
  llvm.cond_br %229, ^bb62, ^bb61
^bb61:  // pred: ^bb60
  llvm.call @free(%228) : (!llvm.ptr) -> ()
  llvm.br ^bb62
^bb62:  // 2 preds: ^bb60, ^bb61
  %230 = llvm.add %226, %8 overflow<nsw, nuw>  : i64
  %231 = llvm.icmp "eq" %230, %225 : i64
  llvm.cond_br %231, ^bb63, ^bb60(%230 : i64) {loop_annotation = #loop_annotation}
^bb63:  // pred: ^bb62
  llvm.call @free(%14) : (!llvm.ptr) -> ()
  llvm.br ^bb64(%3 : i64)
^bb64(%232: i64):  // 2 preds: ^bb63, ^bb66
  %233 = llvm.getelementptr inbounds %23[%232, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %234 = llvm.load %233 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %235 = llvm.icmp "eq" %234, %6 : !llvm.ptr
  llvm.cond_br %235, ^bb66, ^bb65
^bb65:  // pred: ^bb64
  llvm.call @free(%234) : (!llvm.ptr) -> ()
  llvm.br ^bb66
^bb66:  // 2 preds: ^bb64, ^bb65
  %236 = llvm.add %232, %8 overflow<nsw, nuw>  : i64
  %237 = llvm.icmp "eq" %236, %225 : i64
  llvm.cond_br %237, ^bb67, ^bb64(%236 : i64) {loop_annotation = #loop_annotation}
^bb67:  // pred: ^bb66
  llvm.call @free(%23) : (!llvm.ptr) -> ()
  llvm.br ^bb68(%3 : i64)
^bb68(%238: i64):  // 2 preds: ^bb67, ^bb70
  %239 = llvm.getelementptr inbounds %22[%238, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %240 = llvm.load %239 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %241 = llvm.icmp "eq" %240, %6 : !llvm.ptr
  llvm.cond_br %241, ^bb70, ^bb69
^bb69:  // pred: ^bb68
  llvm.call @free(%240) : (!llvm.ptr) -> ()
  llvm.br ^bb70
^bb70:  // 2 preds: ^bb68, ^bb69
  %242 = llvm.add %238, %8 overflow<nsw, nuw>  : i64
  %243 = llvm.icmp "eq" %242, %225 : i64
  llvm.cond_br %243, ^bb71, ^bb68(%242 : i64) {loop_annotation = #loop_annotation}
^bb71:  // 2 preds: ^bb58, ^bb70
  llvm.call @free(%22) : (!llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: processing function @hand_objective
// CHECK: p2p summary:
// CHECK-NEXT:    distinct[0]<"arg-hand_objective-12"> -> [distinct[0]<"arg-hand_objective-12-deref">]
// CHECK-NEXT:    distinct[0]<"fresh-pose_params"> -> [distinct[0]<"fresh-pose_params_data">]
// CHECK-NEXT:    distinct[0]<"fresh-vertex_positions"> -> [distinct[0]<"fresh-positions_data">]
llvm.func @hand_objective(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: i32 {llvm.noundef}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.readnone}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.readnone}, %arg9: i32 {llvm.noundef}, %arg10: i32 {llvm.noundef}, %arg11: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg12: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg13: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.mlir.constant(16 : i64) : i64
  %2 = llvm.mlir.poison : !llvm.ptr
  %3 = llvm.mlir.constant(1 : i32) : i32
  %4 = llvm.mlir.constant(0 : i32) : i32
  %5 = llvm.mlir.constant(0 : i64) : i64
  %6 = llvm.mlir.constant(2 : i32) : i32
  %7 = llvm.mlir.constant(3 : i64) : i64
  %8 = llvm.mlir.zero : !llvm.ptr
  %9 = llvm.call @calloc(%0, %1) {tag = "pose_params"} : (i64, i64) -> !llvm.ptr
  llvm.call @to_pose_params(%arg1, %arg0, %2, %9) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  %10 = llvm.call @calloc(%0, %1) {tag = "vertex_positions"} : (i64, i64) -> !llvm.ptr
  llvm.call @get_skinned_vertex_positions(%arg1, %arg4, %arg3, %arg5, %arg6, %arg7, %arg9, %9, %10, %3) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32) -> ()
  %11 = llvm.icmp "sgt" %arg10, %4 : i32
  llvm.cond_br %11, ^bb1, ^bb5
^bb1:  // pred: ^bb0
  %12 = llvm.getelementptr inbounds %arg12[%5, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %13 = llvm.load %12 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %14 = llvm.load %arg12 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %15 = llvm.getelementptr inbounds %10[%5, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %16 = llvm.load %15 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %17 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %18 = llvm.sext %14 : i32 to i64
  %19 = llvm.zext %arg10 : i32 to i64
  llvm.br ^bb2(%5 : i64)
^bb2(%20: i64):  // 2 preds: ^bb1, ^bb4
  %21 = llvm.mul %20, %18 overflow<nsw>  : i64
  %22 = llvm.getelementptr inbounds %arg11[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %23 = llvm.load %22 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
  %24 = llvm.mul %17, %23 overflow<nsw>  : i32
  %25 = llvm.mul %20, %7 overflow<nsw, nuw>  : i64
  %26 = llvm.sext %24 : i32 to i64
  %27 = llvm.getelementptr %13[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %28 = llvm.getelementptr %16[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %29 = llvm.getelementptr %arg13[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb3(%5 : i64)
^bb3(%30: i64):  // 2 preds: ^bb2, ^bb3
  %31 = llvm.getelementptr %27[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %32 = llvm.load %31 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %33 = llvm.getelementptr %28[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %34 = llvm.load %33 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %35 = llvm.fsub %32, %34  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %36 = llvm.getelementptr %29[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %35, %36 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %37 = llvm.add %30, %0 overflow<nsw, nuw>  : i64
  %38 = llvm.icmp "eq" %37, %7 : i64
  llvm.cond_br %38, ^bb4, ^bb3(%37 : i64) {loop_annotation = #loop_annotation}
^bb4:  // pred: ^bb3
  %39 = llvm.add %20, %0 overflow<nsw, nuw>  : i64
  %40 = llvm.icmp "eq" %39, %19 : i64
  llvm.cond_br %40, ^bb5, ^bb2(%39 : i64) {loop_annotation = #loop_annotation}
^bb5:  // 2 preds: ^bb0, ^bb4
  %41 = llvm.getelementptr inbounds %9[%5, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %42 = llvm.load %41 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %43 = llvm.icmp "eq" %42, %8 : !llvm.ptr
  llvm.cond_br %43, ^bb7, ^bb6
^bb6:  // pred: ^bb5
  llvm.call @free(%42) : (!llvm.ptr) -> ()
  llvm.br ^bb7
^bb7:  // 2 preds: ^bb5, ^bb6
  llvm.call @free(%9) : (!llvm.ptr) -> ()
  %44 = llvm.getelementptr inbounds %10[%5, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %45 = llvm.load %44 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %46 = llvm.icmp "eq" %45, %8 : !llvm.ptr
  llvm.cond_br %46, ^bb9, ^bb8
^bb8:  // pred: ^bb7
  llvm.call @free(%45) : (!llvm.ptr) -> ()
  llvm.br ^bb9
^bb9:  // 2 preds: ^bb7, ^bb8
  llvm.call @free(%10) : (!llvm.ptr) -> ()
  llvm.return
}
