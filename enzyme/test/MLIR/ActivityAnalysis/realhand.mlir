// RUN: %eopt --pass-pipeline="builtin.module(print-activity-analysis{dataflow=true annotate=true})" %s --split-input-file 2>&1 | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.mlir.global external local_unnamed_addr @enzyme_dup() {addr_space = 0 : i32, alignment = 4 : i64, sym_visibility = "private"} : i32
  llvm.mlir.global external local_unnamed_addr @enzyme_const() {addr_space = 0 : i32, alignment = 4 : i64, sym_visibility = "private"} : i32
  llvm.mlir.global external local_unnamed_addr @enzyme_dupnoneed() {addr_space = 0 : i32, alignment = 4 : i64, sym_visibility = "private"} : i32
  llvm.func local_unnamed_addr @get_new_matrix(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {approx_func_fp_math = true, memory = #llvm.memory_effects<other = write, argMem = none, inaccessibleMem = readwrite>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(16 : i64) : i64
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.mlir.constant(3 : i64) : i64
    %3 = llvm.mlir.constant(8 : i64) : i64
    %4 = llvm.call @malloc(%0) : (i64) -> !llvm.ptr
    llvm.store %arg0, %4 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %5 = llvm.getelementptr inbounds %4[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %arg1, %5 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    %6 = llvm.mul %arg1, %arg0 overflow<nsw> : i32
    %7 = llvm.sext %6 : i32 to i64
    %8 = llvm.shl %7, %2 overflow<nsw> : i64
    %9 = llvm.call @malloc(%8) : (i64) -> !llvm.ptr
    %10 = llvm.getelementptr inbounds %4[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %10 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    llvm.return %4 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @malloc(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {approx_func_fp_math = true, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["allockind", "9"], ["allocsize", "4294967295"], ["alloc-family", "malloc"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true}
  llvm.func local_unnamed_addr @get_new_empty_matrix() -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {approx_func_fp_math = true, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(16 : i64) : i64
    %2 = llvm.call @calloc(%0, %1) : (i64, i64) -> !llvm.ptr
    llvm.return %2 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @delete_matrix(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["mustprogress", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(8 : i64) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %3 = llvm.load %2 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %4 = llvm.icmp "eq" %3, %1 : !llvm.ptr
    llvm.cond_br %4, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @free(%3) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.call @free(%arg0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @free(!llvm.ptr {llvm.allocptr, llvm.nocapture, llvm.noundef}) attributes {approx_func_fp_math = true, memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["mustprogress", "nounwind", "willreturn", ["allockind", "4"], ["alloc-family", "malloc"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true}
  llvm.func local_unnamed_addr @get_matrix_array(%arg0: i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {approx_func_fp_math = true, memory = #llvm.memory_effects<other = write, argMem = none, inaccessibleMem = readwrite>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.sext %arg0 : i32 to i64
    %4 = llvm.shl %3, %0 overflow<nsw> : i64
    %5 = llvm.call @malloc(%4) : (i64) -> !llvm.ptr
    %6 = llvm.icmp "sgt" %arg0, %1 : i32
    llvm.cond_br %6, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %7 = llvm.zext %arg0 : i32 to i64
    %8 = llvm.shl %7, %0 overflow<nsw, nuw> : i64
    "llvm.intr.memset"(%5, %2, %8) <{isVolatile = false, tbaa = [#llvm.tbaa_tag<base_type = <id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, access_type = <id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, offset = 0>]}> : (!llvm.ptr, i8, i64) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return %5 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @delete_light_matrix_array(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: i32 {llvm.noundef}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.icmp "sgt" %arg1, %0 : i32
    llvm.cond_br %5, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    %6 = llvm.zext %arg1 : i32 to i64
    llvm.br ^bb2(%1 : i64)
  ^bb2(%7: i64):  // 2 preds: ^bb1, ^bb4
    %8 = llvm.getelementptr inbounds %arg0[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %9 = llvm.load %8 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %10 = llvm.icmp "eq" %9, %3 : !llvm.ptr
    llvm.cond_br %10, ^bb4, ^bb3
  ^bb3:  // pred: ^bb2
    llvm.call @free(%9) : (!llvm.ptr) -> ()
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %11 = llvm.add %7, %4 overflow<nsw, nuw> : i64
    %12 = llvm.icmp "eq" %11, %6 : i64
    llvm.cond_br %12, ^bb5, ^bb2(%11 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.call @free(%arg0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @resize(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["mustprogress", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(8 : i64) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(3 : i64) : i64
    %5 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %6 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %7 = llvm.load %6 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %8 = llvm.mul %7, %5 overflow<nsw> : i32
    %9 = llvm.mul %arg2, %arg1 overflow<nsw> : i32
    %10 = llvm.icmp "eq" %8, %9 : i32
    llvm.cond_br %10, ^bb6, ^bb1
  ^bb1:  // pred: ^bb0
    %11 = llvm.getelementptr inbounds %arg0[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %12 = llvm.load %11 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %13 = llvm.icmp "eq" %12, %2 : !llvm.ptr
    llvm.cond_br %13, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    llvm.call @free(%12) : (!llvm.ptr) -> ()
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %14 = llvm.icmp "sgt" %9, %3 : i32
    llvm.cond_br %14, ^bb4, ^bb5(%2 : !llvm.ptr)
  ^bb4:  // pred: ^bb3
    %15 = llvm.zext %9 : i32 to i64
    %16 = llvm.shl %15, %4 overflow<nsw, nuw> : i64
    %17 = llvm.call @malloc(%16) : (i64) -> !llvm.ptr
    llvm.br ^bb5(%17 : !llvm.ptr)
  ^bb5(%18: !llvm.ptr):  // 2 preds: ^bb3, ^bb4
    llvm.store %18, %11 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb0, ^bb5
    llvm.store %arg2, %6 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    llvm.store %arg1, %arg0 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @set_identity(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {approx_func_fp_math = true, memory = #llvm.memory_effects<other = write, argMem = readwrite, inaccessibleMem = none>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i64) : i64
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %5 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %8 = llvm.load %7 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %9 = llvm.icmp "sgt" %8, %1 : i32
    llvm.cond_br %9, ^bb1, ^bb6
  ^bb1:  // pred: ^bb0
    %10 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %11 = llvm.icmp "sgt" %10, %1 : i32
    %12 = llvm.getelementptr inbounds %arg0[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %13 = llvm.zext %10 : i32 to i64
    %14 = llvm.zext %8 : i32 to i64
    llvm.br ^bb2(%3 : i64)
  ^bb2(%15: i64):  // 2 preds: ^bb1, ^bb5
    llvm.cond_br %11, ^bb3, ^bb5
  ^bb3:  // pred: ^bb2
    %16 = llvm.mul %15, %13 overflow<nsw, nuw> : i64
    %17 = llvm.load %12 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.getelementptr inbounds %17[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb4(%3 : i64)
  ^bb4(%19: i64):  // 2 preds: ^bb3, ^bb4
    %20 = llvm.icmp "eq" %15, %19 : i64
    %21 = llvm.select %20, %4, %5 : i1, f64
    %22 = llvm.getelementptr inbounds %18[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %21, %22 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %23 = llvm.add %19, %6 overflow<nsw, nuw> : i64
    %24 = llvm.icmp "eq" %23, %13 : i64
    llvm.cond_br %24, ^bb5, ^bb4(%23 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb5:  // 2 preds: ^bb2, ^bb4
    %25 = llvm.add %15, %6 overflow<nsw, nuw> : i64
    %26 = llvm.icmp "eq" %25, %14 : i64
    llvm.cond_br %26, ^bb6, ^bb2(%25 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb6:  // 2 preds: ^bb0, ^bb5
    llvm.return
  }
  llvm.func local_unnamed_addr @fill(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: f64 {llvm.noundef}) attributes {approx_func_fp_math = true, memory = #llvm.memory_effects<other = write, argMem = readwrite, inaccessibleMem = none>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i64) : i64
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %6 = llvm.load %5 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %7 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %8 = llvm.mul %7, %6 overflow<nsw> : i32
    %9 = llvm.icmp "sgt" %8, %1 : i32
    llvm.cond_br %9, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %10 = llvm.getelementptr inbounds %arg0[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %11 = llvm.load %10 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %12 = llvm.zext %8 : i32 to i64
    llvm.br ^bb2(%3 : i64)
  ^bb2(%13: i64):  // 2 preds: ^bb1, ^bb2
    %14 = llvm.getelementptr inbounds %11[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %arg1, %14 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %15 = llvm.add %13, %4 overflow<nsw, nuw> : i64
    %16 = llvm.icmp "eq" %15, %12 : i64
    llvm.cond_br %16, ^bb3, ^bb2(%15 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb3:  // 2 preds: ^bb0, ^bb2
    llvm.return
  }
  llvm.func local_unnamed_addr @set_block(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {approx_func_fp_math = true, memory = #llvm.memory_effects<other = readwrite, argMem = readwrite, inaccessibleMem = none>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i64) : i64
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.getelementptr inbounds %arg3[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %6 = llvm.load %5 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %7 = llvm.icmp "sgt" %6, %1 : i32
    llvm.cond_br %7, ^bb1, ^bb6
  ^bb1:  // pred: ^bb0
    %8 = llvm.load %arg3 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %9 = llvm.icmp "sgt" %8, %1 : i32
    %10 = llvm.getelementptr inbounds %arg3[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %11 = llvm.getelementptr inbounds %arg0[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %12 = llvm.sext %arg1 : i32 to i64
    %13 = llvm.sext %8 : i32 to i64
    %14 = llvm.sext %arg2 : i32 to i64
    %15 = llvm.zext %6 : i32 to i64
    %16 = llvm.zext %8 : i32 to i64
    llvm.br ^bb2(%3 : i64)
  ^bb2(%17: i64):  // 2 preds: ^bb1, ^bb5
    llvm.cond_br %9, ^bb3, ^bb5
  ^bb3:  // pred: ^bb2
    %18 = llvm.load %10 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %19 = llvm.mul %17, %13 overflow<nsw, nuw> : i64
    %20 = llvm.load %11 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %21 = llvm.add %17, %14 overflow<nsw> : i64
    %22 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %23 = llvm.sext %22 : i32 to i64
    %24 = llvm.mul %21, %23 overflow<nsw> : i64
    %25 = llvm.getelementptr inbounds %18[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %26 = llvm.getelementptr %20[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %27 = llvm.getelementptr %26[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb4(%3 : i64)
  ^bb4(%28: i64):  // 2 preds: ^bb3, ^bb4
    %29 = llvm.getelementptr inbounds %25[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %30 = llvm.load %29 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %31 = llvm.getelementptr %27[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %30, %31 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %32 = llvm.add %28, %4 overflow<nsw, nuw> : i64
    %33 = llvm.icmp "eq" %32, %16 : i64
    llvm.cond_br %33, ^bb5, ^bb4(%32 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb5:  // 2 preds: ^bb2, ^bb4
    %34 = llvm.add %17, %4 overflow<nsw, nuw> : i64
    %35 = llvm.icmp "eq" %34, %15 : i64
    llvm.cond_br %35, ^bb6, ^bb2(%34 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb6:  // 2 preds: ^bb0, ^bb5
    llvm.return
  }
  llvm.func local_unnamed_addr @copy(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(8 : i64) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(4 : i64) : i64
    %3 = llvm.mlir.constant(3 : i64) : i64
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %8 = llvm.load %7 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %9 = llvm.icmp "eq" %8, %1 : !llvm.ptr
    llvm.cond_br %9, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @free(%8) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %10 = llvm.getelementptr inbounds %arg1[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %11 = llvm.load %10 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %12 = llvm.getelementptr inbounds %arg0[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %11, %12 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    %13 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.store %13, %arg0 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %14 = llvm.mul %13, %11 overflow<nsw> : i32
    %15 = llvm.sext %14 : i32 to i64
    %16 = llvm.shl %15, %3 overflow<nsw> : i64
    %17 = llvm.call @malloc(%16) : (i64) -> !llvm.ptr
    llvm.store %17, %7 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    %18 = llvm.icmp "sgt" %14, %4 : i32
    llvm.cond_br %18, ^bb3, ^bb5
  ^bb3:  // pred: ^bb2
    %19 = llvm.getelementptr inbounds %arg1[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %20 = llvm.load %19 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %21 = llvm.zext %14 : i32 to i64
    llvm.br ^bb4(%5 : i64)
  ^bb4(%22: i64):  // 2 preds: ^bb3, ^bb4
    %23 = llvm.getelementptr inbounds %20[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %24 = llvm.load %23 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %25 = llvm.getelementptr inbounds %17[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %24, %25 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %26 = llvm.add %22, %6 overflow<nsw, nuw> : i64
    %27 = llvm.icmp "eq" %26, %21 : i64
    llvm.cond_br %27, ^bb5, ^bb4(%26 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb5:  // 2 preds: ^bb2, ^bb4
    llvm.return
  }
  llvm.func local_unnamed_addr @square_sum(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> f64 attributes {approx_func_fp_math = true, memory = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %3 = llvm.fmul %2, %2  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %4 = llvm.icmp "sgt" %arg0, %0 : i32
    llvm.cond_br %4, ^bb1, ^bb3(%3 : f64)
  ^bb1:  // pred: ^bb0
    %5 = llvm.zext %arg0 : i32 to i64
    llvm.br ^bb2(%1, %3 : i64, f64)
  ^bb2(%6: i64, %7: f64):  // 2 preds: ^bb1, ^bb2
    %8 = llvm.getelementptr inbounds %arg1[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %9 = llvm.load %8 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %10 = llvm.fmul %9, %9  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %11 = llvm.fadd %10, %7  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %12 = llvm.add %6, %1 overflow<nsw, nuw> : i64
    %13 = llvm.icmp "eq" %12, %5 : i64
    llvm.cond_br %13, ^bb3(%11 : f64), ^bb2(%12, %11 : i64, f64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb3(%14: f64):  // 2 preds: ^bb0, ^bb2
    llvm.return %14 : f64
  }
  llvm.func local_unnamed_addr @angle_axis_to_rotation_matrix(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {approx_func_fp_math = true, memory = #llvm.memory_effects<other = write, argMem = readwrite, inaccessibleMem = none>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nofree", "norecurse", "nosync", "nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(3 : i64) : i64
    %2 = llvm.mlir.constant(1.000000e-04 : f64) : f64
    %3 = llvm.mlir.constant(8 : i64) : i64
    %4 = llvm.mlir.constant(16 : i64) : i64
    %5 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(4 : i64) : i64
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %11 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %12 = llvm.fmul %11, %11  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.br ^bb1(%0, %12 : i64, f64)
  ^bb1(%13: i64, %14: f64):  // 2 preds: ^bb0, ^bb1
    %15 = llvm.getelementptr inbounds %arg0[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %16 = llvm.load %15 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %17 = llvm.fmul %16, %16  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %18 = llvm.fadd %17, %14  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %19 = llvm.add %13, %0 overflow<nsw, nuw> : i64
    %20 = llvm.icmp "eq" %19, %1 : i64
    llvm.cond_br %20, ^bb2, ^bb1(%19, %18 : i64, f64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb2:  // pred: ^bb1
    %21 = llvm.intr.sqrt(%18)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    %22 = llvm.fcmp "olt" %21, %2 {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.cond_br %22, ^bb3, ^bb9
  ^bb3:  // pred: ^bb2
    %23 = llvm.getelementptr inbounds %arg1[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %24 = llvm.load %23 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %25 = llvm.icmp "sgt" %24, %8 : i32
    llvm.cond_br %25, ^bb4, ^bb10
  ^bb4:  // pred: ^bb3
    %26 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %27 = llvm.icmp "sgt" %26, %8 : i32
    %28 = llvm.getelementptr inbounds %arg1[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %29 = llvm.zext %26 : i32 to i64
    %30 = llvm.zext %24 : i32 to i64
    llvm.br ^bb5(%9 : i64)
  ^bb5(%31: i64):  // 2 preds: ^bb4, ^bb8
    llvm.cond_br %27, ^bb6, ^bb8
  ^bb6:  // pred: ^bb5
    %32 = llvm.mul %31, %29 overflow<nsw, nuw> : i64
    %33 = llvm.load %28 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %34 = llvm.getelementptr inbounds %33[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb7(%9 : i64)
  ^bb7(%35: i64):  // 2 preds: ^bb6, ^bb7
    %36 = llvm.icmp "eq" %31, %35 : i64
    %37 = llvm.select %36, %5, %10 : i1, f64
    %38 = llvm.getelementptr inbounds %34[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %37, %38 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %39 = llvm.add %35, %0 overflow<nsw, nuw> : i64
    %40 = llvm.icmp "eq" %39, %29 : i64
    llvm.cond_br %40, ^bb8, ^bb7(%39 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb8:  // 2 preds: ^bb5, ^bb7
    %41 = llvm.add %31, %0 overflow<nsw, nuw> : i64
    %42 = llvm.icmp "eq" %41, %30 : i64
    llvm.cond_br %42, ^bb10, ^bb5(%41 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb9:  // pred: ^bb2
    %43 = llvm.fdiv %11, %21  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %44 = llvm.getelementptr inbounds %arg0[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %45 = llvm.load %44 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %46 = llvm.fdiv %45, %21  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %47 = llvm.getelementptr inbounds %arg0[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %48 = llvm.load %47 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %49 = llvm.fdiv %48, %21  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %50 = llvm.intr.sin(%21)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    %51 = llvm.intr.cos(%21)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    %52 = llvm.fmul %43, %43  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %53 = llvm.fsub %5, %52  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %54 = llvm.fmul %53, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %55 = llvm.fadd %54, %52  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %56 = llvm.getelementptr inbounds %arg1[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %57 = llvm.load %56 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    llvm.store %55, %57 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %58 = llvm.fsub %5, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %59 = llvm.fmul %58, %43  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %60 = llvm.fmul %59, %46  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %61 = llvm.fmul %49, %50  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %62 = llvm.fsub %60, %61  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %63 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %64 = llvm.sext %63 : i32 to i64
    %65 = llvm.getelementptr inbounds %57[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %62, %65 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %66 = llvm.fmul %59, %49  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %67 = llvm.fmul %46, %50  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %68 = llvm.fadd %66, %67  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %69 = llvm.shl %63, %6 overflow<nsw> : i32
    %70 = llvm.sext %69 : i32 to i64
    %71 = llvm.getelementptr inbounds %57[%70] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %68, %71 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %72 = llvm.fadd %60, %61  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %73 = llvm.getelementptr inbounds %57[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %72, %73 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %74 = llvm.fmul %46, %46  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %75 = llvm.fsub %5, %74  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %76 = llvm.fmul %75, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %77 = llvm.fadd %76, %74  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %78 = llvm.getelementptr %65[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %77, %78 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %79 = llvm.fmul %46, %58  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %80 = llvm.fmul %79, %49  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %81 = llvm.fmul %43, %50  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %82 = llvm.fsub %80, %81  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %83 = llvm.or %69, %6  : i32
    %84 = llvm.sext %83 : i32 to i64
    %85 = llvm.getelementptr inbounds %57[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %82, %85 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %86 = llvm.fsub %66, %67  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %87 = llvm.getelementptr inbounds %57[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %86, %87 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %88 = llvm.fadd %80, %81  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %89 = llvm.getelementptr %65[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %88, %89 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %90 = llvm.fmul %49, %49  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %91 = llvm.fsub %5, %90  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %92 = llvm.fmul %91, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %93 = llvm.fadd %92, %90  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %94 = llvm.getelementptr %71[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %93, %94 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.br ^bb10
  ^bb10:  // 3 preds: ^bb3, ^bb8, ^bb9
    llvm.return
  }
  llvm.func local_unnamed_addr @apply_global_transform(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(16 : i64) : i64
    %1 = llvm.mlir.constant(3 : i32) : i32
    %2 = llvm.mlir.constant(4 : i64) : i64
    %3 = llvm.mlir.constant(72 : i64) : i64
    %4 = llvm.mlir.constant(8 : i64) : i64
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.constant(24 : i64) : i64
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.mlir.constant(3 : i64) : i64
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.zero : !llvm.ptr
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.call @malloc(%0) : (i64) -> !llvm.ptr
    llvm.store %1, %12 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %13 = llvm.getelementptr inbounds %12[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %1, %13 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    %14 = llvm.call @malloc(%3) : (i64) -> !llvm.ptr
    %15 = llvm.getelementptr inbounds %12[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %14, %15 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    %16 = llvm.getelementptr inbounds %arg0[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %17 = llvm.load %16 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    llvm.call @angle_axis_to_rotation_matrix(%17, %12) : (!llvm.ptr, !llvm.ptr) -> ()
    %18 = llvm.load %16 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %19 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %20 = llvm.sext %19 : i32 to i64
    %21 = llvm.getelementptr %18[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb1(%5 : i64)
  ^bb1(%22: i64):  // 2 preds: ^bb0, ^bb3
    %23 = llvm.getelementptr %21[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %24 = llvm.mul %22, %6 : i64
    %25 = llvm.getelementptr %14[%24] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb2(%5 : i64)
  ^bb2(%26: i64):  // 2 preds: ^bb1, ^bb2
    %27 = llvm.load %23 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %28 = llvm.getelementptr %25[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %29 = llvm.load %28 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %30 = llvm.fmul %29, %27  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %30, %28 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %31 = llvm.add %26, %7 overflow<nsw, nuw> : i64
    %32 = llvm.icmp "eq" %31, %8 : i64
    llvm.cond_br %32, ^bb3, ^bb2(%31 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb3:  // pred: ^bb2
    %33 = llvm.add %22, %7 overflow<nsw, nuw> : i64
    %34 = llvm.icmp "eq" %33, %8 : i64
    llvm.cond_br %34, ^bb4, ^bb1(%33 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb4:  // pred: ^bb3
    llvm.intr.experimental.noalias.scope.decl <id = distinct[0]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 1">
    %35 = llvm.getelementptr inbounds %arg1[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %36 = llvm.load %35 {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 1">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[2]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[3]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %37 = llvm.icmp "sgt" %36, %9 : i32
    llvm.cond_br %37, ^bb5, ^bb6(%10 : !llvm.ptr)
  ^bb5:  // pred: ^bb4
    %38 = llvm.mul %36, %1 overflow<nsw, nuw> : i32
    %39 = llvm.zext %38 : i32 to i64
    %40 = llvm.shl %39, %8 overflow<nsw, nuw> : i64
    %41 = llvm.call @malloc(%40) : (i64) -> !llvm.ptr
    llvm.br ^bb6(%41 : !llvm.ptr)
  ^bb6(%42: !llvm.ptr):  // 2 preds: ^bb4, ^bb5
    %43 = llvm.icmp "sgt" %36, %9 : i32
    %44 = llvm.getelementptr inbounds %arg1[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %45 = llvm.zext %36 : i32 to i64
    llvm.br ^bb7(%5 : i64)
  ^bb7(%46: i64):  // 2 preds: ^bb6, ^bb12
    llvm.cond_br %43, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    %47 = llvm.getelementptr inbounds %14[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %48 = llvm.load %44 {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 1">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[2]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[3]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %49 = llvm.load %arg1 {alias_scopes = [#llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 1">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[2]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[3]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %50 = llvm.sext %49 : i32 to i64
    %51 = llvm.getelementptr inbounds %42[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %52 = llvm.load %47 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[2]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[3]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.br ^bb9(%5 : i64)
  ^bb9(%53: i64):  // 2 preds: ^bb8, ^bb11
    %54 = llvm.mul %53, %50 overflow<nsw> : i64
    %55 = llvm.getelementptr inbounds %48[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %56 = llvm.load %55 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[2]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[3]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %57 = llvm.fmul %56, %52  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %58 = llvm.mul %53, %6 overflow<nsw, nuw> : i64
    %59 = llvm.getelementptr inbounds %51[%58] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %57, %59 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[2]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[3]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.br ^bb10(%7, %57 : i64, f64)
  ^bb10(%60: i64, %61: f64):  // 2 preds: ^bb9, ^bb10
    %62 = llvm.mul %60, %6 overflow<nsw, nuw> : i64
    %63 = llvm.getelementptr inbounds %47[%62] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %64 = llvm.load %63 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[2]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[3]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %65 = llvm.getelementptr %55[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %66 = llvm.load %65 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[2]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[3]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %67 = llvm.fmul %66, %64  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %68 = llvm.fadd %67, %61  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %68, %59 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[2]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[0]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[3]<>, domain = <id = distinct[1]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %69 = llvm.add %60, %7 overflow<nsw, nuw> : i64
    %70 = llvm.icmp "eq" %69, %8 : i64
    llvm.cond_br %70, ^bb11, ^bb10(%69, %68 : i64, f64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb11:  // pred: ^bb10
    %71 = llvm.add %53, %7 overflow<nsw, nuw> : i64
    %72 = llvm.icmp "eq" %71, %45 : i64
    llvm.cond_br %72, ^bb12, ^bb9(%71 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb12:  // 2 preds: ^bb7, ^bb11
    %73 = llvm.add %46, %7 overflow<nsw, nuw> : i64
    %74 = llvm.icmp "eq" %73, %8 : i64
    llvm.cond_br %74, ^bb13, ^bb7(%73 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb13:  // pred: ^bb12
    %75 = llvm.load %35 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %76 = llvm.icmp "sgt" %75, %9 : i32
    llvm.cond_br %76, ^bb14, ^bb19
  ^bb14:  // pred: ^bb13
    %77 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %78 = llvm.icmp "sgt" %77, %9 : i32
    %79 = llvm.sext %77 : i32 to i64
    %80 = llvm.zext %75 : i32 to i64
    %81 = llvm.shl %19, %11 overflow<nsw> : i32
    %82 = llvm.sext %81 : i32 to i64
    %83 = llvm.zext %77 : i32 to i64
    %84 = llvm.getelementptr %18[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb15(%5 : i64)
  ^bb15(%85: i64):  // 2 preds: ^bb14, ^bb18
    llvm.cond_br %78, ^bb16, ^bb18
  ^bb16:  // pred: ^bb15
    %86 = llvm.mul %85, %8 overflow<nsw, nuw> : i64
    %87 = llvm.load %44 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %88 = llvm.mul %85, %79 overflow<nsw, nuw> : i64
    %89 = llvm.getelementptr inbounds %87[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb17(%5 : i64)
  ^bb17(%90: i64):  // 2 preds: ^bb16, ^bb17
    %91 = llvm.add %90, %86 overflow<nsw, nuw> : i64
    %92 = llvm.getelementptr inbounds %42[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %93 = llvm.load %92 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %94 = llvm.getelementptr %84[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %95 = llvm.load %94 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %96 = llvm.fadd %95, %93  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %97 = llvm.getelementptr inbounds %89[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %96, %97 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %98 = llvm.add %90, %7 overflow<nsw, nuw> : i64
    %99 = llvm.icmp "eq" %98, %83 : i64
    llvm.cond_br %99, ^bb18, ^bb17(%98 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb18:  // 2 preds: ^bb15, ^bb17
    %100 = llvm.add %85, %7 overflow<nsw, nuw> : i64
    %101 = llvm.icmp "eq" %100, %80 : i64
    llvm.cond_br %101, ^bb19, ^bb15(%100 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb19:  // 2 preds: ^bb13, ^bb18
    %102 = llvm.icmp "eq" %14, %10 : !llvm.ptr
    llvm.cond_br %102, ^bb21, ^bb20
  ^bb20:  // pred: ^bb19
    llvm.call @free(%14) : (!llvm.ptr) -> ()
    llvm.br ^bb21
  ^bb21:  // 2 preds: ^bb19, ^bb20
    llvm.call @free(%12) : (!llvm.ptr) -> ()
    %103 = llvm.icmp "eq" %42, %10 : !llvm.ptr
    llvm.cond_br %103, ^bb23, ^bb22
  ^bb22:  // pred: ^bb21
    llvm.call @free(%42) : (!llvm.ptr) -> ()
    llvm.br ^bb23
  ^bb23:  // 2 preds: ^bb21, ^bb22
    llvm.return
  }
  llvm.func local_unnamed_addr @relatives_to_absolutes(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(-1 : i32) : i32
    %3 = llvm.mlir.constant(4 : i64) : i64
    %4 = llvm.mlir.constant(8 : i64) : i64
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.constant(3 : i64) : i64
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.icmp "sgt" %arg0, %0 : i32
    llvm.cond_br %9, ^bb1, ^bb23
  ^bb1:  // pred: ^bb0
    %10 = llvm.zext %arg0 : i32 to i64
    llvm.br ^bb2(%1 : i64)
  ^bb2(%11: i64):  // 2 preds: ^bb1, ^bb22
    %12 = llvm.getelementptr inbounds %arg2[%11] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %13 = llvm.load %12 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %14 = llvm.icmp "eq" %13, %2 : i32
    llvm.cond_br %14, ^bb3, ^bb8
  ^bb3:  // pred: ^bb2
    %15 = llvm.getelementptr inbounds %arg3[%11] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %16 = llvm.getelementptr inbounds %arg1[%11] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %17 = llvm.getelementptr inbounds %15[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %18 = llvm.load %17 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %19 = llvm.icmp "eq" %18, %5 : !llvm.ptr
    llvm.cond_br %19, ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    llvm.call @free(%18) : (!llvm.ptr) -> ()
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %20 = llvm.getelementptr inbounds %16[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %21 = llvm.load %20 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %22 = llvm.getelementptr inbounds %15[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %21, %22 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    %23 = llvm.load %16 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.store %23, %15 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %24 = llvm.mul %23, %21 overflow<nsw> : i32
    %25 = llvm.sext %24 : i32 to i64
    %26 = llvm.shl %25, %6 overflow<nsw> : i64
    %27 = llvm.call @malloc(%26) : (i64) -> !llvm.ptr
    llvm.store %27, %17 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    %28 = llvm.icmp "sgt" %24, %0 : i32
    llvm.cond_br %28, ^bb6, ^bb22
  ^bb6:  // pred: ^bb5
    %29 = llvm.getelementptr inbounds %16[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %30 = llvm.load %29 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %31 = llvm.zext %24 : i32 to i64
    llvm.br ^bb7(%1 : i64)
  ^bb7(%32: i64):  // 2 preds: ^bb6, ^bb7
    %33 = llvm.getelementptr inbounds %30[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %34 = llvm.load %33 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %35 = llvm.getelementptr inbounds %27[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %34, %35 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %36 = llvm.add %32, %8 overflow<nsw, nuw> : i64
    %37 = llvm.icmp "eq" %36, %31 : i64
    llvm.cond_br %37, ^bb22, ^bb7(%36 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb8:  // pred: ^bb2
    %38 = llvm.sext %13 : i32 to i64
    %39 = llvm.getelementptr inbounds %arg3[%38] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %40 = llvm.getelementptr inbounds %arg1[%11] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %41 = llvm.getelementptr inbounds %arg3[%11] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.intr.experimental.noalias.scope.decl <id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">
    llvm.intr.experimental.noalias.scope.decl <id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">
    llvm.intr.experimental.noalias.scope.decl <id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">
    %42 = llvm.load %39 {alias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %43 = llvm.getelementptr inbounds %40[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %44 = llvm.load %43 {alias_scopes = [#llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %45 = llvm.load %41 {alias_scopes = [#llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %46 = llvm.getelementptr inbounds %41[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %47 = llvm.load %46 {alias_scopes = [#llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %48 = llvm.mul %47, %45 overflow<nsw> : i32
    %49 = llvm.mul %44, %42 overflow<nsw> : i32
    %50 = llvm.icmp "eq" %48, %49 : i32
    llvm.cond_br %50, ^bb14, ^bb9
  ^bb9:  // pred: ^bb8
    %51 = llvm.getelementptr inbounds %41[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %52 = llvm.load %51 {alias_scopes = [#llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %53 = llvm.icmp "eq" %52, %5 : !llvm.ptr
    llvm.cond_br %53, ^bb11, ^bb10
  ^bb10:  // pred: ^bb9
    llvm.call @free(%52) {noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">]} : (!llvm.ptr) -> ()
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb9, ^bb10
    %54 = llvm.icmp "sgt" %49, %0 : i32
    llvm.cond_br %54, ^bb12, ^bb13(%5 : !llvm.ptr)
  ^bb12:  // pred: ^bb11
    %55 = llvm.zext %49 : i32 to i64
    %56 = llvm.shl %55, %6 overflow<nsw, nuw> : i64
    %57 = llvm.call @malloc(%56) : (i64) -> !llvm.ptr
    llvm.br ^bb13(%57 : !llvm.ptr)
  ^bb13(%58: !llvm.ptr):  // 2 preds: ^bb11, ^bb12
    llvm.store %58, %51 {alias_scopes = [#llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb14
  ^bb14:  // 2 preds: ^bb8, ^bb13
    llvm.store %44, %46 {alias_scopes = [#llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    llvm.store %42, %41 {alias_scopes = [#llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %59 = llvm.icmp "sgt" %42, %0 : i32
    llvm.cond_br %59, ^bb15, ^bb22
  ^bb15:  // pred: ^bb14
    %60 = llvm.icmp "sgt" %44, %0 : i32
    %61 = llvm.getelementptr inbounds %39[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %62 = llvm.getelementptr inbounds %40[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %63 = llvm.getelementptr inbounds %41[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %64 = llvm.getelementptr inbounds %39[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %65 = llvm.zext %42 : i32 to i64
    %66 = llvm.zext %44 : i32 to i64
    llvm.br ^bb16(%1 : i64)
  ^bb16(%67: i64):  // 2 preds: ^bb15, ^bb21
    llvm.cond_br %60, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    %68 = llvm.load %61 {alias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %69 = llvm.getelementptr inbounds %68[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %70 = llvm.load %62 {alias_scopes = [#llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %71 = llvm.load %40 {alias_scopes = [#llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %72 = llvm.load %63 {alias_scopes = [#llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %73 = llvm.load %64 {alias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %74 = llvm.icmp "sgt" %73, %7 : i32
    %75 = llvm.sext %71 : i32 to i64
    %76 = llvm.getelementptr inbounds %72[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %77 = llvm.zext %73 : i32 to i64
    llvm.br ^bb18(%1 : i64)
  ^bb18(%78: i64):  // 2 preds: ^bb17, ^bb20
    %79 = llvm.load %69 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %80 = llvm.mul %78, %75 overflow<nsw> : i64
    %81 = llvm.getelementptr inbounds %70[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %82 = llvm.load %81 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %83 = llvm.fmul %82, %79  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %84 = llvm.mul %78, %65 overflow<nsw, nuw> : i64
    %85 = llvm.getelementptr inbounds %76[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %83, %85 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.cond_br %74, ^bb19(%8, %83 : i64, f64), ^bb20
  ^bb19(%86: i64, %87: f64):  // 2 preds: ^bb18, ^bb19
    %88 = llvm.mul %86, %65 overflow<nsw, nuw> : i64
    %89 = llvm.getelementptr inbounds %69[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %90 = llvm.load %89 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %91 = llvm.getelementptr %81[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %92 = llvm.load %91 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %93 = llvm.fmul %92, %90  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %94 = llvm.fadd %93, %87  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %94, %85 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[4]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[6]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[7]<>, domain = <id = distinct[5]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %95 = llvm.add %86, %8 overflow<nsw, nuw> : i64
    %96 = llvm.icmp "eq" %95, %77 : i64
    llvm.cond_br %96, ^bb20, ^bb19(%95, %94 : i64, f64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb20:  // 2 preds: ^bb18, ^bb19
    %97 = llvm.add %78, %8 overflow<nsw, nuw> : i64
    %98 = llvm.icmp "eq" %97, %66 : i64
    llvm.cond_br %98, ^bb21, ^bb18(%97 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb21:  // 2 preds: ^bb16, ^bb20
    %99 = llvm.add %67, %8 overflow<nsw, nuw> : i64
    %100 = llvm.icmp "eq" %99, %65 : i64
    llvm.cond_br %100, ^bb22, ^bb16(%99 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb22:  // 4 preds: ^bb5, ^bb7, ^bb14, ^bb21
    %101 = llvm.add %11, %8 overflow<nsw, nuw> : i64
    %102 = llvm.icmp "eq" %101, %10 : i64
    llvm.cond_br %102, ^bb23, ^bb2(%101 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb23:  // 2 preds: ^bb0, ^bb22
    llvm.return
  }
  llvm.func local_unnamed_addr @euler_angles_to_rotation_matrix(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(16 : i64) : i64
    %1 = llvm.mlir.constant(8 : i64) : i64
    %2 = llvm.mlir.constant(72 : i64) : i64
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(24 : i64) : i64
    %5 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %6 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.mlir.constant(3 : i64) : i64
    %9 = llvm.mlir.constant(32 : i64) : i64
    %10 = llvm.mlir.constant(40 : i64) : i64
    %11 = llvm.mlir.constant(56 : i64) : i64
    %12 = llvm.mlir.constant(64 : i64) : i64
    %13 = llvm.mlir.constant(48 : i64) : i64
    %14 = llvm.mlir.constant(4 : i64) : i64
    %15 = llvm.mlir.constant(9 : i32) : i32
    %16 = llvm.mlir.zero : !llvm.ptr
    %17 = llvm.mlir.constant(3 : i32) : i32
    %18 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %19 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %20 = llvm.load %19 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %21 = llvm.getelementptr inbounds %arg0[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %22 = llvm.load %21 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %23 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
    %24 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
    %25 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
    llvm.br ^bb1(%3 : i64)
  ^bb1(%26: i64):  // 2 preds: ^bb0, ^bb3
    %27 = llvm.mul %26, %4 overflow<nsw, nuw> : i64
    %28 = llvm.getelementptr inbounds %23[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb2(%3 : i64)
  ^bb2(%29: i64):  // 2 preds: ^bb1, ^bb2
    %30 = llvm.icmp "eq" %26, %29 : i64
    %31 = llvm.select %30, %5, %6 : i1, f64
    %32 = llvm.getelementptr inbounds %28[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %31, %32 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %33 = llvm.add %29, %7 overflow<nsw, nuw> : i64
    %34 = llvm.icmp "eq" %33, %8 : i64
    llvm.cond_br %34, ^bb3, ^bb2(%33 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb3:  // pred: ^bb2
    %35 = llvm.add %26, %7 overflow<nsw, nuw> : i64
    %36 = llvm.icmp "eq" %35, %8 : i64
    llvm.cond_br %36, ^bb4, ^bb1(%35 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb4:  // pred: ^bb3
    %37 = llvm.intr.cos(%18)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    %38 = llvm.getelementptr %23[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %37, %38 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %39 = llvm.intr.sin(%18)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    %40 = llvm.getelementptr %23[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %39, %40 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %41 = llvm.fneg %39  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %42 = llvm.getelementptr inbounds %23[%11] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %41, %42 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %43 = llvm.getelementptr %23[%12] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %37, %43 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.br ^bb5(%3 : i64)
  ^bb5(%44: i64):  // 2 preds: ^bb4, ^bb7
    %45 = llvm.mul %44, %4 overflow<nsw, nuw> : i64
    %46 = llvm.getelementptr inbounds %24[%45] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb6(%3 : i64)
  ^bb6(%47: i64):  // 2 preds: ^bb5, ^bb6
    %48 = llvm.icmp "eq" %44, %47 : i64
    %49 = llvm.select %48, %5, %6 : i1, f64
    %50 = llvm.getelementptr inbounds %46[%47] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %49, %50 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %51 = llvm.add %47, %7 overflow<nsw, nuw> : i64
    %52 = llvm.icmp "eq" %51, %8 : i64
    llvm.cond_br %52, ^bb7, ^bb6(%51 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb7:  // pred: ^bb6
    %53 = llvm.add %44, %7 overflow<nsw, nuw> : i64
    %54 = llvm.icmp "eq" %53, %8 : i64
    llvm.cond_br %54, ^bb8, ^bb5(%53 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb8:  // pred: ^bb7
    %55 = llvm.intr.cos(%20)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    llvm.store %55, %24 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %56 = llvm.intr.sin(%20)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    %57 = llvm.getelementptr inbounds %24[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %56, %57 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %58 = llvm.fneg %56  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %59 = llvm.getelementptr inbounds %24[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %58, %59 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %60 = llvm.getelementptr %24[%12] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %55, %60 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.br ^bb9(%3 : i64)
  ^bb9(%61: i64):  // 2 preds: ^bb8, ^bb11
    %62 = llvm.mul %61, %4 overflow<nsw, nuw> : i64
    %63 = llvm.getelementptr inbounds %25[%62] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb10(%3 : i64)
  ^bb10(%64: i64):  // 2 preds: ^bb9, ^bb10
    %65 = llvm.icmp "eq" %61, %64 : i64
    %66 = llvm.select %65, %5, %6 : i1, f64
    %67 = llvm.getelementptr inbounds %63[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %66, %67 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %68 = llvm.add %64, %7 overflow<nsw, nuw> : i64
    %69 = llvm.icmp "eq" %68, %8 : i64
    llvm.cond_br %69, ^bb11, ^bb10(%68 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb11:  // pred: ^bb10
    %70 = llvm.add %61, %7 overflow<nsw, nuw> : i64
    %71 = llvm.icmp "eq" %70, %8 : i64
    llvm.cond_br %71, ^bb12, ^bb9(%70 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb12:  // pred: ^bb11
    %72 = llvm.intr.cos(%22)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    llvm.store %72, %25 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %73 = llvm.intr.sin(%22)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    %74 = llvm.getelementptr inbounds %25[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %73, %74 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %75 = llvm.fneg %73  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %76 = llvm.getelementptr inbounds %25[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %75, %76 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %77 = llvm.getelementptr %25[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %72, %77 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %78 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
    llvm.br ^bb13(%3 : i64)
  ^bb13(%79: i64):  // 2 preds: ^bb12, ^bb17
    %80 = llvm.getelementptr inbounds %25[%79] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %81 = llvm.getelementptr inbounds %78[%79] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %82 = llvm.load %80 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[8]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[10]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[11]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.br ^bb14(%3 : i64)
  ^bb14(%83: i64):  // 2 preds: ^bb13, ^bb16
    %84 = llvm.mul %83, %8 overflow<nsw, nuw> : i64
    %85 = llvm.getelementptr inbounds %24[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %86 = llvm.load %85 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[8]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[10]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[11]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %87 = llvm.fmul %86, %82  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %88 = llvm.getelementptr inbounds %81[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb15(%7, %87 : i64, f64)
  ^bb15(%89: i64, %90: f64):  // 2 preds: ^bb14, ^bb15
    %91 = llvm.mul %89, %4 overflow<nsw, nuw> : i64
    %92 = llvm.getelementptr inbounds %80[%91] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %93 = llvm.load %92 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[8]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[10]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[11]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %94 = llvm.getelementptr %85[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %95 = llvm.load %94 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[8]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[10]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[11]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %96 = llvm.fmul %95, %93  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %97 = llvm.fadd %96, %90  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %98 = llvm.add %89, %7 overflow<nsw, nuw> : i64
    %99 = llvm.icmp "eq" %98, %8 : i64
    llvm.cond_br %99, ^bb16, ^bb15(%98, %97 : i64, f64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb16:  // pred: ^bb15
    llvm.store %97, %88 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[8]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[10]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[11]<>, domain = <id = distinct[9]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %100 = llvm.add %83, %7 overflow<nsw, nuw> : i64
    %101 = llvm.icmp "eq" %100, %8 : i64
    llvm.cond_br %101, ^bb17, ^bb14(%100 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb17:  // pred: ^bb16
    %102 = llvm.add %79, %7 overflow<nsw, nuw> : i64
    %103 = llvm.icmp "eq" %102, %8 : i64
    llvm.cond_br %103, ^bb18, ^bb13(%102 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb18:  // pred: ^bb17
    llvm.intr.experimental.noalias.scope.decl <id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">
    %104 = llvm.load %arg1 {alias_scopes = [#llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %105 = llvm.getelementptr inbounds %arg1[%14] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %106 = llvm.load %105 {alias_scopes = [#llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %107 = llvm.mul %106, %104 overflow<nsw> : i32
    %108 = llvm.icmp "eq" %107, %15 : i32
    %109 = llvm.getelementptr inbounds %arg1[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %110 = llvm.load %109 {alias_scopes = [#llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    llvm.cond_br %108, ^bb22(%110 : !llvm.ptr), ^bb19
  ^bb19:  // pred: ^bb18
    %111 = llvm.icmp "eq" %110, %16 : !llvm.ptr
    llvm.cond_br %111, ^bb21, ^bb20
  ^bb20:  // pred: ^bb19
    llvm.call @free(%110) {noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">]} : (!llvm.ptr) -> ()
    llvm.br ^bb21
  ^bb21:  // 2 preds: ^bb19, ^bb20
    %112 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
    llvm.store %112, %109 {alias_scopes = [#llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb22(%112 : !llvm.ptr)
  ^bb22(%113: !llvm.ptr):  // 2 preds: ^bb18, ^bb21
    llvm.store %17, %105 {alias_scopes = [#llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    llvm.store %17, %arg1 {alias_scopes = [#llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    llvm.br ^bb23(%3 : i64)
  ^bb23(%114: i64):  // 2 preds: ^bb22, ^bb27
    %115 = llvm.getelementptr inbounds %78[%114] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %116 = llvm.getelementptr inbounds %113[%114] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %117 = llvm.load %115 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    llvm.br ^bb24(%3 : i64)
  ^bb24(%118: i64):  // 2 preds: ^bb23, ^bb26
    %119 = llvm.mul %118, %8 overflow<nsw, nuw> : i64
    %120 = llvm.getelementptr inbounds %23[%119] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %121 = llvm.load %120 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %122 = llvm.fmul %121, %117  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %123 = llvm.getelementptr inbounds %116[%119] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %122, %123 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.br ^bb25(%7, %122 : i64, f64)
  ^bb25(%124: i64, %125: f64):  // 2 preds: ^bb24, ^bb25
    %126 = llvm.mul %124, %4 overflow<nsw, nuw> : i64
    %127 = llvm.getelementptr inbounds %115[%126] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %128 = llvm.load %127 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %129 = llvm.getelementptr %120[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %130 = llvm.load %129 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %131 = llvm.fmul %130, %128  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %132 = llvm.fadd %131, %125  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %133 = llvm.add %124, %7 overflow<nsw, nuw> : i64
    %134 = llvm.icmp "eq" %133, %8 : i64
    llvm.cond_br %134, ^bb26, ^bb25(%133, %132 : i64, f64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb26:  // pred: ^bb25
    llvm.store %132, %123 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[14]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[15]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[12]<>, domain = <id = distinct[13]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %135 = llvm.add %118, %7 overflow<nsw, nuw> : i64
    %136 = llvm.icmp "eq" %135, %8 : i64
    llvm.cond_br %136, ^bb27, ^bb24(%135 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb27:  // pred: ^bb26
    %137 = llvm.add %114, %7 overflow<nsw, nuw> : i64
    %138 = llvm.icmp "eq" %137, %8 : i64
    llvm.cond_br %138, ^bb28, ^bb23(%137 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb28:  // pred: ^bb27
    llvm.call @free(%23) : (!llvm.ptr) -> ()
    llvm.call @free(%24) : (!llvm.ptr) -> ()
    llvm.call @free(%25) : (!llvm.ptr) -> ()
    llvm.call @free(%78) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @get_posed_relatives(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(128 : i64) : i64
    %1 = llvm.mlir.constant(16 : i64) : i64
    %2 = llvm.mlir.constant(3 : i32) : i32
    %3 = llvm.mlir.constant(4 : i64) : i64
    %4 = llvm.mlir.constant(72 : i64) : i64
    %5 = llvm.mlir.constant(8 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(0 : i64) : i64
    %8 = llvm.mlir.constant(5 : i64) : i64
    %9 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %10 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %11 = llvm.mlir.constant(1 : i64) : i64
    %12 = llvm.mlir.constant(3 : i64) : i64
    %13 = llvm.mlir.constant(2 : i32) : i32
    %14 = llvm.mlir.zero : !llvm.ptr
    %15 = llvm.mlir.constant(4 : i32) : i32
    %16 = llvm.mlir.constant(1 : i32) : i32
    %17 = llvm.call @malloc(%0) : (i64) -> !llvm.ptr
    %18 = llvm.call @malloc(%1) : (i64) -> !llvm.ptr
    llvm.store %2, %18 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %19 = llvm.getelementptr inbounds %18[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %2, %19 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    %20 = llvm.call @malloc(%4) : (i64) -> !llvm.ptr
    %21 = llvm.getelementptr inbounds %18[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %20, %21 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    %22 = llvm.icmp "sgt" %arg0, %6 : i32
    llvm.cond_br %22, ^bb1, ^bb25
  ^bb1:  // pred: ^bb0
    %23 = llvm.getelementptr inbounds %arg2[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %24 = llvm.load %23 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %25 = llvm.load %arg2 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %26 = llvm.sext %25 : i32 to i64
    %27 = llvm.zext %arg0 : i32 to i64
    llvm.br ^bb2(%7 : i64)
  ^bb2(%28: i64):  // 2 preds: ^bb1, ^bb24
    llvm.br ^bb3(%7 : i64)
  ^bb3(%29: i64):  // 2 preds: ^bb2, ^bb5
    %30 = llvm.shl %29, %8 overflow<nsw> : i64
    %31 = llvm.getelementptr inbounds %17[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb4(%7 : i64)
  ^bb4(%32: i64):  // 2 preds: ^bb3, ^bb4
    %33 = llvm.icmp "eq" %29, %32 : i64
    %34 = llvm.select %33, %9, %10 : i1, f64
    %35 = llvm.getelementptr inbounds %31[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %34, %35 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %36 = llvm.add %32, %11 overflow<nsw, nuw> : i64
    %37 = llvm.icmp "eq" %36, %3 : i64
    llvm.cond_br %37, ^bb5, ^bb4(%36 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb5:  // pred: ^bb4
    %38 = llvm.add %29, %11 overflow<nsw, nuw> : i64
    %39 = llvm.icmp "eq" %38, %3 : i64
    llvm.cond_br %39, ^bb6, ^bb3(%38 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb6:  // pred: ^bb5
    %40 = llvm.add %28, %12 overflow<nsw, nuw> : i64
    %41 = llvm.mul %40, %26 overflow<nsw> : i64
    %42 = llvm.getelementptr inbounds %24[%41] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.call @euler_angles_to_rotation_matrix(%42, %18) : (!llvm.ptr, !llvm.ptr) -> ()
    %43 = llvm.load %19 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %44 = llvm.icmp "sgt" %43, %6 : i32
    llvm.cond_br %44, ^bb7, ^bb11
  ^bb7:  // pred: ^bb6
    %45 = llvm.load %18 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %46 = llvm.icmp "sgt" %45, %6 : i32
    %47 = llvm.sext %45 : i32 to i64
    %48 = llvm.zext %43 : i32 to i64
    %49 = llvm.zext %45 : i32 to i64
    %50 = llvm.shl %49, %12 overflow<nsw, nuw> : i64
    llvm.br ^bb8(%7 : i64)
  ^bb8(%51: i64):  // 2 preds: ^bb7, ^bb10
    llvm.cond_br %46, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %52 = llvm.shl %51, %8 overflow<nsw, nuw> : i64
    %53 = llvm.getelementptr %17[%52] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %54 = llvm.load %21 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %55 = llvm.mul %51, %47 overflow<nsw, nuw> : i64
    %56 = llvm.getelementptr inbounds %54[%55] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%53, %56, %50) <{isVolatile = false, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    %57 = llvm.add %51, %11 overflow<nsw, nuw> : i64
    %58 = llvm.icmp "eq" %57, %48 : i64
    llvm.cond_br %58, ^bb11, ^bb8(%57 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb11:  // 2 preds: ^bb6, ^bb10
    %59 = llvm.getelementptr inbounds %arg1[%28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %60 = llvm.getelementptr inbounds %arg3[%28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.intr.experimental.noalias.scope.decl <id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">
    llvm.intr.experimental.noalias.scope.decl <id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">
    %61 = llvm.load %59 {alias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %62 = llvm.load %60 {alias_scopes = [#llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %63 = llvm.getelementptr inbounds %60[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %64 = llvm.load %63 {alias_scopes = [#llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %65 = llvm.mul %64, %62 overflow<nsw> : i32
    %66 = llvm.shl %61, %13 overflow<nsw> : i32
    %67 = llvm.icmp "eq" %65, %66 : i32
    llvm.cond_br %67, ^bb17, ^bb12
  ^bb12:  // pred: ^bb11
    %68 = llvm.getelementptr inbounds %60[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %69 = llvm.load %68 {alias_scopes = [#llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %70 = llvm.icmp "eq" %69, %14 : !llvm.ptr
    llvm.cond_br %70, ^bb14, ^bb13
  ^bb13:  // pred: ^bb12
    llvm.call @free(%69) {noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">]} : (!llvm.ptr) -> ()
    llvm.br ^bb14
  ^bb14:  // 2 preds: ^bb12, ^bb13
    %71 = llvm.icmp "sgt" %61, %6 : i32
    llvm.cond_br %71, ^bb15, ^bb16(%14 : !llvm.ptr)
  ^bb15:  // pred: ^bb14
    %72 = llvm.zext %66 : i32 to i64
    %73 = llvm.shl %72, %12 overflow<nsw, nuw> : i64
    %74 = llvm.call @malloc(%73) : (i64) -> !llvm.ptr
    llvm.br ^bb16(%74 : !llvm.ptr)
  ^bb16(%75: !llvm.ptr):  // 2 preds: ^bb14, ^bb15
    llvm.store %75, %68 {alias_scopes = [#llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb17
  ^bb17:  // 2 preds: ^bb11, ^bb16
    llvm.store %15, %63 {alias_scopes = [#llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    llvm.store %61, %60 {alias_scopes = [#llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %76 = llvm.icmp "sgt" %61, %6 : i32
    llvm.cond_br %76, ^bb18, ^bb24
  ^bb18:  // pred: ^bb17
    %77 = llvm.getelementptr inbounds %59[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %78 = llvm.getelementptr inbounds %60[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %79 = llvm.getelementptr inbounds %59[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %80 = llvm.zext %61 : i32 to i64
    %81 = llvm.load %77 {alias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %82 = llvm.load %78 {alias_scopes = [#llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %83 = llvm.load %79 {alias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %84 = llvm.icmp "sgt" %83, %16 : i32
    %85 = llvm.zext %83 : i32 to i64
    llvm.br ^bb19(%7 : i64)
  ^bb19(%86: i64):  // 2 preds: ^bb18, ^bb23
    %87 = llvm.getelementptr inbounds %81[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %88 = llvm.getelementptr inbounds %82[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb20(%7 : i64)
  ^bb20(%89: i64):  // 2 preds: ^bb19, ^bb22
    %90 = llvm.load %87 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %91 = llvm.shl %89, %8 overflow<nsw> : i64
    %92 = llvm.getelementptr inbounds %17[%91] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %93 = llvm.load %92 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %94 = llvm.fmul %93, %90  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %95 = llvm.mul %89, %80 overflow<nsw, nuw> : i64
    %96 = llvm.getelementptr inbounds %88[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %94, %96 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.cond_br %84, ^bb21(%11, %94 : i64, f64), ^bb22
  ^bb21(%97: i64, %98: f64):  // 2 preds: ^bb20, ^bb21
    %99 = llvm.mul %97, %80 overflow<nsw, nuw> : i64
    %100 = llvm.getelementptr inbounds %87[%99] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %101 = llvm.load %100 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %102 = llvm.getelementptr %92[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %103 = llvm.load %102 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %104 = llvm.fmul %103, %101  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %105 = llvm.fadd %104, %98  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %105, %96 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[16]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[19]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[18]<>, domain = <id = distinct[17]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %106 = llvm.add %97, %11 overflow<nsw, nuw> : i64
    %107 = llvm.icmp "eq" %106, %85 : i64
    llvm.cond_br %107, ^bb22, ^bb21(%106, %105 : i64, f64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb22:  // 2 preds: ^bb20, ^bb21
    %108 = llvm.add %89, %11 overflow<nsw, nuw> : i64
    %109 = llvm.icmp "eq" %108, %3 : i64
    llvm.cond_br %109, ^bb23, ^bb20(%108 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb23:  // pred: ^bb22
    %110 = llvm.add %86, %11 overflow<nsw, nuw> : i64
    %111 = llvm.icmp "eq" %110, %80 : i64
    llvm.cond_br %111, ^bb24, ^bb19(%110 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb24:  // 2 preds: ^bb17, ^bb23
    %112 = llvm.add %28, %11 overflow<nsw, nuw> : i64
    %113 = llvm.icmp "eq" %112, %27 : i64
    llvm.cond_br %113, ^bb25, ^bb2(%112 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb25:  // 2 preds: ^bb0, ^bb24
    %114 = llvm.icmp "eq" %17, %14 : !llvm.ptr
    llvm.cond_br %114, ^bb27, ^bb26
  ^bb26:  // pred: ^bb25
    llvm.call @free(%17) : (!llvm.ptr) -> ()
    llvm.br ^bb27
  ^bb27:  // 2 preds: ^bb25, ^bb26
    %115 = llvm.load %21 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %116 = llvm.icmp "eq" %115, %14 : !llvm.ptr
    llvm.cond_br %116, ^bb29, ^bb28
  ^bb28:  // pred: ^bb27
    llvm.call @free(%115) : (!llvm.ptr) -> ()
    llvm.br ^bb29
  ^bb29:  // 2 preds: ^bb27, ^bb28
    llvm.call @free(%18) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @get_skinned_vertex_positions(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: i32 {llvm.noundef}, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}, %arg9: i32 {llvm.noundef}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(8 : i64) : i64
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.constant(3 : i64) : i64
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(3 : i32) : i32
    %10 = llvm.mlir.constant(16 : i64) : i64
    %11 = llvm.mlir.constant(4 : i32) : i32
    %12 = llvm.mlir.constant(2 : i32) : i32
    %13 = llvm.mlir.constant(24 : i64) : i64
    %14 = llvm.sext %arg0 : i32 to i64
    %15 = llvm.shl %14, %0 overflow<nsw> : i64
    %16 = llvm.call @malloc(%15) : (i64) -> !llvm.ptr
    %17 = llvm.icmp "sgt" %arg0, %1 : i32
    llvm.cond_br %17, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %18 = llvm.call @malloc(%15) : (i64) -> !llvm.ptr
    %19 = llvm.call @malloc(%15) : (i64) -> !llvm.ptr
    llvm.br ^bb3(%19, %18 : !llvm.ptr, !llvm.ptr)
  ^bb2:  // pred: ^bb0
    %20 = llvm.zext %arg0 : i32 to i64
    %21 = llvm.shl %20, %0 overflow<nsw, nuw> : i64
    "llvm.intr.memset"(%16, %2, %21) <{isVolatile = false, tbaa = [#llvm.tbaa_tag<base_type = <id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, access_type = <id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, offset = 0>]}> : (!llvm.ptr, i8, i64) -> ()
    %22 = llvm.call @malloc(%15) : (i64) -> !llvm.ptr
    "llvm.intr.memset"(%22, %2, %21) <{isVolatile = false, tbaa = [#llvm.tbaa_tag<base_type = <id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, access_type = <id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, offset = 0>]}> : (!llvm.ptr, i8, i64) -> ()
    %23 = llvm.call @malloc(%15) : (i64) -> !llvm.ptr
    "llvm.intr.memset"(%23, %2, %21) <{isVolatile = false, tbaa = [#llvm.tbaa_tag<base_type = <id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, access_type = <id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, offset = 0>]}> : (!llvm.ptr, i8, i64) -> ()
    llvm.br ^bb3(%23, %22 : !llvm.ptr, !llvm.ptr)
  ^bb3(%24: !llvm.ptr, %25: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
    llvm.call @get_posed_relatives(%arg0, %arg1, %arg7, %16) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @relatives_to_absolutes(%arg0, %16, %arg2, %25) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.cond_br %17, ^bb4, ^bb20
  ^bb4:  // pred: ^bb3
    %26 = llvm.zext %arg0 : i32 to i64
    llvm.br ^bb5(%3 : i64)
  ^bb5(%27: i64):  // 2 preds: ^bb4, ^bb19
    %28 = llvm.getelementptr inbounds %25[%27] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %29 = llvm.getelementptr inbounds %arg3[%27] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %30 = llvm.getelementptr inbounds %24[%27] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.intr.experimental.noalias.scope.decl <id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">
    llvm.intr.experimental.noalias.scope.decl <id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">
    llvm.intr.experimental.noalias.scope.decl <id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">
    %31 = llvm.load %28 {alias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %32 = llvm.getelementptr inbounds %29[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %33 = llvm.load %32 {alias_scopes = [#llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %34 = llvm.load %30 {alias_scopes = [#llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %35 = llvm.getelementptr inbounds %30[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %36 = llvm.load %35 {alias_scopes = [#llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %37 = llvm.mul %36, %34 overflow<nsw> : i32
    %38 = llvm.mul %33, %31 overflow<nsw> : i32
    %39 = llvm.icmp "eq" %37, %38 : i32
    llvm.cond_br %39, ^bb11, ^bb6
  ^bb6:  // pred: ^bb5
    %40 = llvm.getelementptr inbounds %30[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %41 = llvm.load %40 {alias_scopes = [#llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %42 = llvm.icmp "eq" %41, %5 : !llvm.ptr
    llvm.cond_br %42, ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    llvm.call @free(%41) {noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">]} : (!llvm.ptr) -> ()
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %43 = llvm.icmp "sgt" %38, %1 : i32
    llvm.cond_br %43, ^bb9, ^bb10(%5 : !llvm.ptr)
  ^bb9:  // pred: ^bb8
    %44 = llvm.zext %38 : i32 to i64
    %45 = llvm.shl %44, %6 overflow<nsw, nuw> : i64
    %46 = llvm.call @malloc(%45) : (i64) -> !llvm.ptr
    llvm.br ^bb10(%46 : !llvm.ptr)
  ^bb10(%47: !llvm.ptr):  // 2 preds: ^bb8, ^bb9
    llvm.store %47, %40 {alias_scopes = [#llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb5, ^bb10
    llvm.store %33, %35 {alias_scopes = [#llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    llvm.store %31, %30 {alias_scopes = [#llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %48 = llvm.icmp "sgt" %31, %1 : i32
    llvm.cond_br %48, ^bb12, ^bb19
  ^bb12:  // pred: ^bb11
    %49 = llvm.icmp "sgt" %33, %1 : i32
    %50 = llvm.getelementptr inbounds %28[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %51 = llvm.getelementptr inbounds %29[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %52 = llvm.getelementptr inbounds %30[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %53 = llvm.getelementptr inbounds %28[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %54 = llvm.zext %31 : i32 to i64
    %55 = llvm.zext %33 : i32 to i64
    llvm.br ^bb13(%3 : i64)
  ^bb13(%56: i64):  // 2 preds: ^bb12, ^bb18
    llvm.cond_br %49, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    %57 = llvm.load %50 {alias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %58 = llvm.getelementptr inbounds %57[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %59 = llvm.load %51 {alias_scopes = [#llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %60 = llvm.load %29 {alias_scopes = [#llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %61 = llvm.load %52 {alias_scopes = [#llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %62 = llvm.load %53 {alias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %63 = llvm.icmp "sgt" %62, %7 : i32
    %64 = llvm.sext %60 : i32 to i64
    %65 = llvm.getelementptr inbounds %61[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %66 = llvm.zext %62 : i32 to i64
    llvm.br ^bb15(%3 : i64)
  ^bb15(%67: i64):  // 2 preds: ^bb14, ^bb17
    %68 = llvm.load %58 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %69 = llvm.mul %67, %64 overflow<nsw> : i64
    %70 = llvm.getelementptr inbounds %59[%69] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %71 = llvm.load %70 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %72 = llvm.fmul %71, %68  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %73 = llvm.mul %67, %54 overflow<nsw, nuw> : i64
    %74 = llvm.getelementptr inbounds %65[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %72, %74 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.cond_br %63, ^bb16(%8, %72 : i64, f64), ^bb17
  ^bb16(%75: i64, %76: f64):  // 2 preds: ^bb15, ^bb16
    %77 = llvm.mul %75, %54 overflow<nsw, nuw> : i64
    %78 = llvm.getelementptr inbounds %58[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %79 = llvm.load %78 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %80 = llvm.getelementptr %70[%75] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %81 = llvm.load %80 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %82 = llvm.fmul %81, %79  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %83 = llvm.fadd %82, %76  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %83, %74 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[20]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[22]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[23]<>, domain = <id = distinct[21]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %84 = llvm.add %75, %8 overflow<nsw, nuw> : i64
    %85 = llvm.icmp "eq" %84, %66 : i64
    llvm.cond_br %85, ^bb17, ^bb16(%84, %83 : i64, f64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb17:  // 2 preds: ^bb15, ^bb16
    %86 = llvm.add %67, %8 overflow<nsw, nuw> : i64
    %87 = llvm.icmp "eq" %86, %55 : i64
    llvm.cond_br %87, ^bb18, ^bb15(%86 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb18:  // 2 preds: ^bb13, ^bb17
    %88 = llvm.add %56, %8 overflow<nsw, nuw> : i64
    %89 = llvm.icmp "eq" %88, %54 : i64
    llvm.cond_br %89, ^bb19, ^bb13(%88 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb19:  // 2 preds: ^bb11, ^bb18
    %90 = llvm.add %27, %8 overflow<nsw, nuw> : i64
    %91 = llvm.icmp "eq" %90, %26 : i64
    llvm.cond_br %91, ^bb20, ^bb5(%90 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb20:  // 2 preds: ^bb3, ^bb19
    %92 = llvm.getelementptr inbounds %arg4[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %93 = llvm.load %92 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %94 = llvm.load %arg8 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %95 = llvm.getelementptr inbounds %arg8[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %96 = llvm.load %95 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %97 = llvm.mul %96, %94 overflow<nsw> : i32
    %98 = llvm.mul %93, %9 overflow<nsw> : i32
    %99 = llvm.icmp "eq" %97, %98 : i32
    llvm.cond_br %99, ^bb26, ^bb21
  ^bb21:  // pred: ^bb20
    %100 = llvm.getelementptr inbounds %arg8[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %101 = llvm.load %100 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %102 = llvm.icmp "eq" %101, %5 : !llvm.ptr
    llvm.cond_br %102, ^bb23, ^bb22
  ^bb22:  // pred: ^bb21
    llvm.call @free(%101) : (!llvm.ptr) -> ()
    llvm.br ^bb23
  ^bb23:  // 2 preds: ^bb21, ^bb22
    %103 = llvm.icmp "sgt" %93, %1 : i32
    llvm.cond_br %103, ^bb24, ^bb25(%5 : !llvm.ptr)
  ^bb24:  // pred: ^bb23
    %104 = llvm.zext %98 : i32 to i64
    %105 = llvm.shl %104, %6 overflow<nsw, nuw> : i64
    %106 = llvm.call @malloc(%105) : (i64) -> !llvm.ptr
    llvm.br ^bb25(%106 : !llvm.ptr)
  ^bb25(%107: !llvm.ptr):  // 2 preds: ^bb23, ^bb24
    llvm.store %107, %100 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb26
  ^bb26:  // 2 preds: ^bb20, ^bb25
    llvm.store %93, %95 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    llvm.store %9, %arg8 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %108 = llvm.icmp "sgt" %93, %1 : i32
    llvm.cond_br %108, ^bb27, ^bb28
  ^bb27:  // pred: ^bb26
    %109 = llvm.getelementptr inbounds %arg8[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %110 = llvm.load %109 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %111 = llvm.zext %98 : i32 to i64
    %112 = llvm.shl %111, %6 overflow<nsw, nuw> : i64
    "llvm.intr.memset"(%110, %2, %112) <{isVolatile = false, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]}> : (!llvm.ptr, i8, i64) -> ()
    llvm.br ^bb28
  ^bb28:  // 2 preds: ^bb26, ^bb27
    %113 = llvm.call @malloc(%10) : (i64) -> !llvm.ptr
    llvm.store %11, %113 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %114 = llvm.getelementptr inbounds %113[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %93, %114 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    %115 = llvm.shl %93, %12 overflow<nsw> : i32
    %116 = llvm.sext %115 : i32 to i64
    %117 = llvm.shl %116, %6 overflow<nsw> : i64
    %118 = llvm.call @malloc(%117) : (i64) -> !llvm.ptr
    %119 = llvm.getelementptr inbounds %113[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %118, %119 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    llvm.cond_br %17, ^bb29, ^bb50
  ^bb29:  // pred: ^bb28
    %120 = llvm.getelementptr inbounds %arg4[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %121 = llvm.zext %93 : i32 to i64
    %122 = llvm.getelementptr inbounds %arg5[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %123 = llvm.getelementptr inbounds %arg8[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %124 = llvm.zext %arg0 : i32 to i64
    llvm.br ^bb30(%118, %118, %3, %11 : !llvm.ptr, !llvm.ptr, i64, i32)
  ^bb30(%125: !llvm.ptr, %126: !llvm.ptr, %127: i64, %128: i32):  // 2 preds: ^bb29, ^bb49
    %129 = llvm.getelementptr inbounds %24[%127] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.intr.experimental.noalias.scope.decl <id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">
    llvm.intr.experimental.noalias.scope.decl <id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">
    llvm.intr.experimental.noalias.scope.decl <id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">
    %130 = llvm.load %129 {alias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %131 = llvm.mul %93, %128 overflow<nsw> : i32
    %132 = llvm.mul %130, %93 overflow<nsw> : i32
    %133 = llvm.icmp "eq" %131, %132 : i32
    llvm.cond_br %133, ^bb36(%125, %126 : !llvm.ptr, !llvm.ptr), ^bb31
  ^bb31:  // pred: ^bb30
    %134 = llvm.icmp "eq" %126, %5 : !llvm.ptr
    llvm.cond_br %134, ^bb33, ^bb32
  ^bb32:  // pred: ^bb31
    llvm.call @free(%126) {noalias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">]} : (!llvm.ptr) -> ()
    llvm.br ^bb33
  ^bb33:  // 2 preds: ^bb31, ^bb32
    %135 = llvm.icmp "sgt" %132, %1 : i32
    llvm.cond_br %135, ^bb34, ^bb35(%5 : !llvm.ptr)
  ^bb34:  // pred: ^bb33
    %136 = llvm.zext %132 : i32 to i64
    %137 = llvm.shl %136, %6 overflow<nsw, nuw> : i64
    %138 = llvm.call @malloc(%137) : (i64) -> !llvm.ptr
    llvm.br ^bb35(%138 : !llvm.ptr)
  ^bb35(%139: !llvm.ptr):  // 2 preds: ^bb33, ^bb34
    llvm.store %139, %119 {alias_scopes = [#llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb36(%139, %139 : !llvm.ptr, !llvm.ptr)
  ^bb36(%140: !llvm.ptr, %141: !llvm.ptr):  // 2 preds: ^bb30, ^bb35
    %142 = llvm.icmp "sgt" %130, %1 : i32
    llvm.cond_br %142, ^bb37, ^bb44(%141 : !llvm.ptr)
  ^bb37:  // pred: ^bb36
    %143 = llvm.getelementptr inbounds %129[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %144 = llvm.getelementptr inbounds %129[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %145 = llvm.zext %130 : i32 to i64
    llvm.br ^bb38(%141, %3 : !llvm.ptr, i64)
  ^bb38(%146: !llvm.ptr, %147: i64):  // 2 preds: ^bb37, ^bb43
    llvm.cond_br %108, ^bb39, ^bb43(%146 : !llvm.ptr)
  ^bb39:  // pred: ^bb38
    %148 = llvm.load %143 {alias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %149 = llvm.getelementptr inbounds %148[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %150 = llvm.load %120 {alias_scopes = [#llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %151 = llvm.load %arg4 {alias_scopes = [#llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">], alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %152 = llvm.load %144 {alias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">], alignment = 4 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %153 = llvm.icmp "sgt" %152, %7 : i32
    %154 = llvm.sext %151 : i32 to i64
    %155 = llvm.getelementptr inbounds %140[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %156 = llvm.zext %152 : i32 to i64
    llvm.br ^bb40(%3 : i64)
  ^bb40(%157: i64):  // 2 preds: ^bb39, ^bb42
    %158 = llvm.load %149 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %159 = llvm.mul %157, %154 overflow<nsw> : i64
    %160 = llvm.getelementptr inbounds %150[%159] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %161 = llvm.load %160 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %162 = llvm.fmul %161, %158  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %163 = llvm.mul %157, %145 overflow<nsw, nuw> : i64
    %164 = llvm.getelementptr inbounds %155[%163] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %162, %164 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    llvm.cond_br %153, ^bb41(%8, %162 : i64, f64), ^bb42
  ^bb41(%165: i64, %166: f64):  // 2 preds: ^bb40, ^bb41
    %167 = llvm.mul %165, %145 overflow<nsw, nuw> : i64
    %168 = llvm.getelementptr inbounds %149[%167] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %169 = llvm.load %168 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %170 = llvm.getelementptr %160[%165] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %171 = llvm.load %170 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %172 = llvm.fmul %171, %169  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %173 = llvm.fadd %172, %166  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %173, %164 {alignment = 8 : i64, noalias_scopes = [#llvm.alias_scope<id = distinct[24]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 0">, #llvm.alias_scope<id = distinct[26]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 1">, #llvm.alias_scope<id = distinct[27]<>, domain = <id = distinct[25]<>, description = "mat_mult">, description = "mat_mult: argument 2">], tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %174 = llvm.add %165, %8 overflow<nsw, nuw> : i64
    %175 = llvm.icmp "eq" %174, %156 : i64
    llvm.cond_br %175, ^bb42, ^bb41(%174, %173 : i64, f64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb42:  // 2 preds: ^bb40, ^bb41
    %176 = llvm.add %157, %8 overflow<nsw, nuw> : i64
    %177 = llvm.icmp "eq" %176, %121 : i64
    llvm.cond_br %177, ^bb43(%140 : !llvm.ptr), ^bb40(%176 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb43(%178: !llvm.ptr):  // 2 preds: ^bb38, ^bb42
    %179 = llvm.add %147, %8 overflow<nsw, nuw> : i64
    %180 = llvm.icmp "eq" %179, %145 : i64
    llvm.cond_br %180, ^bb44(%178 : !llvm.ptr), ^bb38(%178, %179 : !llvm.ptr, i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb44(%181: !llvm.ptr):  // 2 preds: ^bb36, ^bb43
    llvm.cond_br %108, ^bb45, ^bb49(%140, %181 : !llvm.ptr, !llvm.ptr)
  ^bb45:  // pred: ^bb44
    %182 = llvm.load %119 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %183 = llvm.load %122 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %184 = llvm.load %arg5 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %185 = llvm.load %123 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %186 = llvm.sext %130 : i32 to i64
    %187 = llvm.sext %184 : i32 to i64
    %188 = llvm.getelementptr %183[%127] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb46(%3 : i64)
  ^bb46(%189: i64):  // 2 preds: ^bb45, ^bb48
    %190 = llvm.mul %189, %186 overflow<nsw> : i64
    %191 = llvm.mul %189, %187 overflow<nsw> : i64
    %192 = llvm.getelementptr %188[%191] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %193 = llvm.getelementptr %182[%190] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %194 = llvm.mul %189, %13 : i64
    %195 = llvm.getelementptr %185[%194] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb47(%3 : i64)
  ^bb47(%196: i64):  // 2 preds: ^bb46, ^bb47
    %197 = llvm.getelementptr %193[%196] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %198 = llvm.load %197 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %199 = llvm.load %192 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %200 = llvm.fmul %199, %198  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %201 = llvm.getelementptr %195[%196] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %202 = llvm.load %201 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %203 = llvm.fadd %202, %200  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %203, %201 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %204 = llvm.add %196, %8 overflow<nsw, nuw> : i64
    %205 = llvm.icmp "eq" %204, %6 : i64
    llvm.cond_br %205, ^bb48, ^bb47(%204 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb48:  // pred: ^bb47
    %206 = llvm.add %189, %8 overflow<nsw, nuw> : i64
    %207 = llvm.icmp "eq" %206, %121 : i64
    llvm.cond_br %207, ^bb49(%182, %182 : !llvm.ptr, !llvm.ptr), ^bb46(%206 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb49(%208: !llvm.ptr, %209: !llvm.ptr):  // 2 preds: ^bb44, ^bb48
    %210 = llvm.add %127, %8 overflow<nsw, nuw> : i64
    %211 = llvm.icmp "eq" %210, %124 : i64
    llvm.cond_br %211, ^bb50, ^bb30(%208, %209, %210, %130 : !llvm.ptr, !llvm.ptr, i64, i32) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb50:  // 2 preds: ^bb28, ^bb49
    %212 = llvm.icmp "ne" %arg6, %1 : i32
    %213 = llvm.and %212, %108  : i1
    llvm.cond_br %213, ^bb51, ^bb53
  ^bb51:  // pred: ^bb50
    %214 = llvm.getelementptr inbounds %arg8[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %215 = llvm.load %214 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %216 = llvm.zext %93 : i32 to i64
    llvm.br ^bb52(%3 : i64)
  ^bb52(%217: i64):  // 2 preds: ^bb51, ^bb52
    %218 = llvm.mul %217, %13 : i64
    %219 = llvm.getelementptr inbounds %215[%218] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %220 = llvm.load %219 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %221 = llvm.fneg %220  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %221, %219 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %222 = llvm.add %217, %8 overflow<nsw, nuw> : i64
    %223 = llvm.icmp "eq" %222, %216 : i64
    llvm.cond_br %223, ^bb53, ^bb52(%222 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb53:  // 2 preds: ^bb50, ^bb52
    %224 = llvm.icmp "eq" %arg9, %1 : i32
    llvm.cond_br %224, ^bb55, ^bb54
  ^bb54:  // pred: ^bb53
    llvm.call @apply_global_transform(%arg7, %arg8) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb55
  ^bb55:  // 2 preds: ^bb53, ^bb54
    %225 = llvm.load %119 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %226 = llvm.icmp "eq" %225, %5 : !llvm.ptr
    llvm.cond_br %226, ^bb57, ^bb56
  ^bb56:  // pred: ^bb55
    llvm.call @free(%225) : (!llvm.ptr) -> ()
    llvm.br ^bb57
  ^bb57:  // 2 preds: ^bb55, ^bb56
    llvm.call @free(%113) : (!llvm.ptr) -> ()
    llvm.cond_br %17, ^bb59, ^bb58
  ^bb58:  // pred: ^bb57
    llvm.call @free(%16) : (!llvm.ptr) -> ()
    llvm.call @free(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb71
  ^bb59:  // pred: ^bb57
    %227 = llvm.zext %arg0 : i32 to i64
    llvm.br ^bb60(%3 : i64)
  ^bb60(%228: i64):  // 2 preds: ^bb59, ^bb62
    %229 = llvm.getelementptr inbounds %16[%228, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %230 = llvm.load %229 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %231 = llvm.icmp "eq" %230, %5 : !llvm.ptr
    llvm.cond_br %231, ^bb62, ^bb61
  ^bb61:  // pred: ^bb60
    llvm.call @free(%230) : (!llvm.ptr) -> ()
    llvm.br ^bb62
  ^bb62:  // 2 preds: ^bb60, ^bb61
    %232 = llvm.add %228, %8 overflow<nsw, nuw> : i64
    %233 = llvm.icmp "eq" %232, %227 : i64
    llvm.cond_br %233, ^bb63, ^bb60(%232 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb63:  // pred: ^bb62
    llvm.call @free(%16) : (!llvm.ptr) -> ()
    llvm.br ^bb64(%3 : i64)
  ^bb64(%234: i64):  // 2 preds: ^bb63, ^bb66
    %235 = llvm.getelementptr inbounds %25[%234, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %236 = llvm.load %235 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %237 = llvm.icmp "eq" %236, %5 : !llvm.ptr
    llvm.cond_br %237, ^bb66, ^bb65
  ^bb65:  // pred: ^bb64
    llvm.call @free(%236) : (!llvm.ptr) -> ()
    llvm.br ^bb66
  ^bb66:  // 2 preds: ^bb64, ^bb65
    %238 = llvm.add %234, %8 overflow<nsw, nuw> : i64
    %239 = llvm.icmp "eq" %238, %227 : i64
    llvm.cond_br %239, ^bb67, ^bb64(%238 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb67:  // pred: ^bb66
    llvm.call @free(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb68(%3 : i64)
  ^bb68(%240: i64):  // 2 preds: ^bb67, ^bb70
    %241 = llvm.getelementptr inbounds %24[%240, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %242 = llvm.load %241 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %243 = llvm.icmp "eq" %242, %5 : !llvm.ptr
    llvm.cond_br %243, ^bb70, ^bb69
  ^bb69:  // pred: ^bb68
    llvm.call @free(%242) : (!llvm.ptr) -> ()
    llvm.br ^bb70
  ^bb70:  // 2 preds: ^bb68, ^bb69
    %244 = llvm.add %240, %8 overflow<nsw, nuw> : i64
    %245 = llvm.icmp "eq" %244, %227 : i64
    llvm.cond_br %245, ^bb71, ^bb68(%244 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb71:  // 2 preds: ^bb58, ^bb70
    llvm.call @free(%24) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @to_pose_params(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.readnone}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(3 : i32) : i32
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.mlir.constant(8 : i64) : i64
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.constant(-3 : i32) : i32
    %5 = llvm.mlir.constant(3 : i64) : i64
    %6 = llvm.mlir.constant(0 : i8) : i8
    %7 = llvm.mlir.constant(24 : i64) : i64
    %8 = llvm.mlir.constant(48 : i64) : i64
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %11 = llvm.mlir.constant(1 : i64) : i64
    %12 = llvm.mlir.constant(0 : i32) : i32
    %13 = llvm.mlir.constant(5 : i32) : i32
    %14 = llvm.mlir.constant(6 : i32) : i32
    %15 = llvm.mlir.constant(2 : i32) : i32
    %16 = llvm.mlir.constant(1 : i32) : i32
    %17 = llvm.add %arg0, %0 overflow<nsw> : i32
    %18 = llvm.load %arg3 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %19 = llvm.getelementptr inbounds %arg3[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %20 = llvm.load %19 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : !llvm.ptr -> i32
    %21 = llvm.mul %20, %18 overflow<nsw> : i32
    %22 = llvm.mul %17, %0 overflow<nsw> : i32
    %23 = llvm.icmp "eq" %21, %22 : i32
    llvm.cond_br %23, ^bb6, ^bb1
  ^bb1:  // pred: ^bb0
    %24 = llvm.getelementptr inbounds %arg3[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %25 = llvm.load %24 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %26 = llvm.icmp "eq" %25, %3 : !llvm.ptr
    llvm.cond_br %26, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    llvm.call @free(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %27 = llvm.icmp "sgt" %arg0, %4 : i32
    llvm.cond_br %27, ^bb4, ^bb5(%3 : !llvm.ptr)
  ^bb4:  // pred: ^bb3
    %28 = llvm.zext %22 : i32 to i64
    %29 = llvm.shl %28, %5 overflow<nsw, nuw> : i64
    %30 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    llvm.br ^bb5(%30 : !llvm.ptr)
  ^bb5(%31: !llvm.ptr):  // 2 preds: ^bb3, ^bb4
    llvm.store %31, %24 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb0, ^bb5
    llvm.store %17, %19 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 4>]} : i32, !llvm.ptr
    llvm.store %0, %arg3 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
    %32 = llvm.icmp "sgt" %arg0, %4 : i32
    %33 = llvm.getelementptr inbounds %arg3[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %34 = llvm.load %33 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    llvm.cond_br %32, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %35 = llvm.zext %22 : i32 to i64
    %36 = llvm.shl %35, %5 overflow<nsw, nuw> : i64
    "llvm.intr.memset"(%34, %6, %36) <{isVolatile = false, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]}> : (!llvm.ptr, i8, i64) -> ()
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %37 = llvm.getelementptr inbounds %34[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %38 = llvm.getelementptr inbounds %34[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    "llvm.intr.memcpy"(%34, %arg1, %7) <{isVolatile = false, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %39 = llvm.getelementptr inbounds %arg1[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb9(%9 : i64)
  ^bb9(%40: i64):  // 2 preds: ^bb8, ^bb9
    %41 = llvm.getelementptr inbounds %37[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %10, %41 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %42 = llvm.getelementptr inbounds %39[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %43 = llvm.load %42 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %44 = llvm.getelementptr inbounds %38[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %43, %44 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %45 = llvm.add %40, %11 overflow<nsw, nuw> : i64
    %46 = llvm.icmp "eq" %45, %5 : i64
    llvm.cond_br %46, ^bb10(%12, %13, %14 : i32, i32, i32), ^bb9(%45 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb10(%47: i32, %48: i32, %49: i32):  // 2 preds: ^bb9, ^bb14
    %50 = llvm.sext %48 : i32 to i64
    %51 = llvm.add %48, %0 : i32
    llvm.br ^bb11(%50, %15, %49 : i64, i32, i32)
  ^bb11(%52: i64, %53: i32, %54: i32):  // 2 preds: ^bb10, ^bb13
    %55 = llvm.sext %54 : i32 to i64
    %56 = llvm.getelementptr inbounds %arg1[%55] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %57 = llvm.load %56 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %58 = llvm.mul %52, %7 : i64
    %59 = llvm.getelementptr inbounds %34[%58] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %57, %59 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %60 = llvm.add %54, %16 overflow<nsw> : i32
    %61 = llvm.icmp "eq" %53, %15 : i32
    llvm.cond_br %61, ^bb12, ^bb13(%60 : i32)
  ^bb12:  // pred: ^bb11
    %62 = llvm.sext %60 : i32 to i64
    %63 = llvm.getelementptr inbounds %arg1[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %64 = llvm.load %63 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %65 = llvm.getelementptr %59[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %64, %65 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %66 = llvm.add %54, %15 overflow<nsw> : i32
    llvm.br ^bb13(%66 : i32)
  ^bb13(%67: i32):  // 2 preds: ^bb11, ^bb12
    %68 = llvm.add %52, %11 overflow<nsw> : i64
    %69 = llvm.add %53, %16 overflow<nsw, nuw> : i32
    %70 = llvm.trunc %68 : i64 to i32
    %71 = llvm.icmp "eq" %51, %70 : i32
    llvm.cond_br %71, ^bb14, ^bb11(%68, %69, %67 : i64, i32, i32) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb14:  // pred: ^bb13
    %72 = llvm.trunc %52 : i64 to i32
    %73 = llvm.add %72, %15 overflow<nsw> : i32
    %74 = llvm.add %47, %16 overflow<nsw, nuw> : i32
    %75 = llvm.icmp "eq" %74, %13 : i32
    llvm.cond_br %75, ^bb15, ^bb10(%74, %73, %67 : i32, i32, i32) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb15:  // pred: ^bb14
    llvm.return
  }
  llvm.func @hand_objective(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: i32 {llvm.noundef}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.readnone}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.readnone}, %arg9: i32 {llvm.noundef}, %arg10: i32 {llvm.noundef}, %arg11: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg12: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg13: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(1 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
    %1 = llvm.mlir.constant(16 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
    %2 = llvm.mlir.poison {enzyme.ici = true} : !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(8 : i64) : i64
    %6 = llvm.mlir.constant(0 : i64) : i64
    %7 = llvm.mlir.constant(24 : i64) : i64
    %8 = llvm.mlir.constant(3 : i64) : i64
    %9 = llvm.mlir.zero : !llvm.ptr
    %10 = llvm.call @calloc(%0, %1) : (i64, i64) -> !llvm.ptr
    llvm.call @to_pose_params(%arg1, %arg0, %2, %10) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %11 = llvm.call @calloc(%0, %1) : (i64, i64) -> !llvm.ptr
    llvm.call @get_skinned_vertex_positions(%arg1, %arg4, %arg3, %arg5, %arg6, %arg7, %arg9, %10, %11, %3) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32) -> ()
    %12 = llvm.icmp "sgt" %arg10, %4 : i32
    llvm.cond_br %12, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    %13 = llvm.getelementptr inbounds %arg12[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %14 = llvm.load %13 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %15 = llvm.load %arg12 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %16 = llvm.getelementptr inbounds %11[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %17 = llvm.load %16 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.load %11 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %19 = llvm.sext %15 : i32 to i64
    %20 = llvm.zext %arg10 : i32 to i64
    llvm.br ^bb2(%6 : i64)
  ^bb2(%21: i64):  // 2 preds: ^bb1, ^bb4
    %22 = llvm.mul %21, %19 overflow<nsw> : i64
    %23 = llvm.getelementptr inbounds %arg11[%21] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %24 = llvm.load %23 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %25 = llvm.mul %18, %24 overflow<nsw> : i32
    %26 = llvm.sext %25 : i32 to i64
    %27 = llvm.getelementptr %14[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %28 = llvm.getelementptr %17[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %29 = llvm.mul %21, %7 : i64
    %30 = llvm.getelementptr %arg13[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb3(%6 : i64)
  ^bb3(%31: i64):  // 2 preds: ^bb2, ^bb3
    %32 = llvm.getelementptr %27[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %33 = llvm.load %32 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %34 = llvm.getelementptr %28[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %35 = llvm.load %34 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %36 = llvm.fsub %33, %35  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %37 = llvm.getelementptr %30[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %36, %37 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %38 = llvm.add %31, %0 overflow<nsw, nuw> : i64
    %39 = llvm.icmp "eq" %38, %8 : i64
    llvm.cond_br %39, ^bb4, ^bb3(%38 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb4:  // pred: ^bb3
    %40 = llvm.add %21, %0 overflow<nsw, nuw> : i64
    %41 = llvm.icmp "eq" %40, %20 : i64
    llvm.cond_br %41, ^bb5, ^bb2(%40 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb5:  // 2 preds: ^bb0, ^bb4
    %42 = llvm.getelementptr inbounds %10[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %43 = llvm.load %42 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %44 = llvm.icmp "eq" %43, %9 : !llvm.ptr
    llvm.cond_br %44, ^bb7, ^bb6
  ^bb6:  // pred: ^bb5
    llvm.call @free(%43) : (!llvm.ptr) -> ()
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    llvm.call @free(%10) : (!llvm.ptr) -> ()
    %45 = llvm.getelementptr inbounds %11[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %46 = llvm.load %45 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %47 = llvm.icmp "eq" %46, %9 : !llvm.ptr
    llvm.cond_br %47, ^bb9, ^bb8
  ^bb8:  // pred: ^bb7
    llvm.call @free(%46) : (!llvm.ptr) -> ()
    llvm.br ^bb9
  ^bb9:  // 2 preds: ^bb7, ^bb8
    llvm.call @free(%11) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @hand_objective_complicated(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: i32 {llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg7: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg9: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg10: i32 {llvm.noundef}, %arg11: i32 {llvm.noundef}, %arg12: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg13: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg14: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(16 : i64) : i64
    %2 = llvm.mlir.poison : !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(8 : i64) : i64
    %6 = llvm.mlir.constant(0 : i64) : i64
    %7 = llvm.mlir.constant(4 : i64) : i64
    %8 = llvm.mlir.constant(24 : i64) : i64
    %9 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %10 = llvm.mlir.constant(3 : i64) : i64
    %11 = llvm.call @calloc(%0, %1) : (i64, i64) -> !llvm.ptr
    llvm.call @to_pose_params(%arg2, %arg0, %2, %11) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %12 = llvm.call @calloc(%0, %1) : (i64, i64) -> !llvm.ptr
    llvm.call @get_skinned_vertex_positions(%arg2, %arg5, %arg4, %arg6, %arg7, %arg8, %arg10, %11, %12, %3) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32) -> ()
    %13 = llvm.icmp "sgt" %arg11, %4 : i32
    llvm.cond_br %13, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    %14 = llvm.getelementptr inbounds %12[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %16 = llvm.load %12 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %17 = llvm.getelementptr inbounds %arg13[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %18 = llvm.load %17 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> !llvm.ptr
    %19 = llvm.load %arg13 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "Matrix", members = {<#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 4>, <#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %20 = llvm.sext %19 : i32 to i64
    %21 = llvm.zext %arg11 : i32 to i64
    llvm.br ^bb2(%6 : i64)
  ^bb2(%22: i64):  // 2 preds: ^bb1, ^bb4
    %23 = llvm.getelementptr inbounds %arg12[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %24 = llvm.load %23 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %25 = llvm.sext %24 : i32 to i64
    %26 = llvm.getelementptr inbounds %arg9[%25] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Triangle", (array<3 x i32>)>
    %27 = llvm.shl %22, %7 overflow<nsw> : i64
    %28 = llvm.getelementptr inbounds %arg1[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %29 = llvm.load %26 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %30 = llvm.mul %16, %29 overflow<nsw> : i32
    %31 = llvm.getelementptr inbounds %28[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %32 = llvm.getelementptr inbounds %26[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %33 = llvm.load %32 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %34 = llvm.mul %33, %16 overflow<nsw> : i32
    %35 = llvm.getelementptr inbounds %26[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %36 = llvm.load %35 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %37 = llvm.mul %36, %16 overflow<nsw> : i32
    %38 = llvm.mul %22, %20 overflow<nsw> : i64
    %39 = llvm.sext %30 : i32 to i64
    %40 = llvm.sext %34 : i32 to i64
    %41 = llvm.sext %37 : i32 to i64
    %42 = llvm.getelementptr %15[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %43 = llvm.getelementptr %15[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %44 = llvm.getelementptr %15[%41] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %45 = llvm.getelementptr %18[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %46 = llvm.mul %22, %8 : i64
    %47 = llvm.getelementptr %arg14[%46] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb3(%6 : i64)
  ^bb3(%48: i64):  // 2 preds: ^bb2, ^bb3
    %49 = llvm.load %28 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %50 = llvm.getelementptr %42[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %51 = llvm.load %50 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %52 = llvm.load %31 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %53 = llvm.getelementptr %43[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %54 = llvm.load %53 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %55 = llvm.fadd %49, %52  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %56 = llvm.fsub %9, %55  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %57 = llvm.getelementptr %44[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %58 = llvm.load %57 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %59 = llvm.getelementptr %45[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %60 = llvm.load %59 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> f64
    %61 = llvm.fmul %51, %49  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %62 = llvm.fmul %54, %52  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %63 = llvm.fmul %58, %56  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %64 = llvm.fadd %62, %61  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %65 = llvm.fadd %64, %63  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %66 = llvm.fsub %60, %65  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %67 = llvm.getelementptr %47[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %66, %67 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : f64, !llvm.ptr
    %68 = llvm.add %48, %0 overflow<nsw, nuw> : i64
    %69 = llvm.icmp "eq" %68, %10 : i64
    llvm.cond_br %69, ^bb4, ^bb3(%68 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb4:  // pred: ^bb3
    %70 = llvm.add %22, %0 overflow<nsw, nuw> : i64
    %71 = llvm.icmp "eq" %70, %21 : i64
    llvm.cond_br %71, ^bb5, ^bb2(%70 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, mustProgress = true>}
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.return
  }
  llvm.func local_unnamed_addr @dhand_objective(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: !llvm.ptr {llvm.noundef}, %arg4: !llvm.ptr {llvm.noundef}, %arg5: !llvm.ptr {llvm.noundef}, %arg6: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg7: !llvm.ptr {llvm.noundef}, %arg8: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg9: !llvm.ptr {llvm.noundef}, %arg10: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg11: !llvm.ptr {llvm.noundef}, %arg12: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg13: !llvm.ptr {llvm.noundef}, %arg14: i32 {llvm.noundef}, %arg15: i32 {llvm.noundef}, %arg16: !llvm.ptr {llvm.noundef}, %arg17: !llvm.ptr {llvm.noundef}, %arg18: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg19: !llvm.ptr {llvm.noundef}, %arg20: !llvm.ptr {llvm.noundef}) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true} {
    %0 = llvm.mlir.addressof @enzyme_dup : !llvm.ptr
    %1 = llvm.mlir.addressof @enzyme_const : !llvm.ptr
    %2 = llvm.mlir.addressof @enzyme_dupnoneed : !llvm.ptr
    %3 = llvm.mlir.addressof @hand_objective : !llvm.ptr
    %4 = llvm.load %0 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %5 = llvm.load %1 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    %6 = llvm.load %2 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C/C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
    llvm.call @__enzyme_autodiff(%3, %4, %arg0, %arg1, %5, %arg2, %5, %arg3, %5, %arg4, %5, %arg5, %5, %arg7, %5, %arg9, %5, %arg11, %5, %arg13, %5, %arg14, %5, %arg15, %5, %arg16, %5, %arg17, %6, %arg19, %arg20) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @__enzyme_autodiff(!llvm.ptr {llvm.noundef}, ...) attributes {approx_func_fp_math = true, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, unsafe_fp_math = true}
  llvm.func local_unnamed_addr @calloc(i64 {llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, passthrough = ["nofree", "nounwind", "willreturn", ["allockind", "17"], ["allocsize", "1"], ["alloc-family", "malloc"]], sym_visibility = "private"}
}

