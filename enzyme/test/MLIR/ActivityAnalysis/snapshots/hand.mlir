// RUN: %eopt --print-activity-analysis='annotate' %s | FileCheck %s

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
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.mlir.global external local_unnamed_addr @enzyme_dup() {addr_space = 0 : i32, alignment = 4 : i64, sym_visibility = "private"} : i32
  llvm.mlir.global external local_unnamed_addr @enzyme_const() {addr_space = 0 : i32, alignment = 4 : i64, sym_visibility = "private"} : i32
  llvm.mlir.global external local_unnamed_addr @enzyme_dupnoneed() {addr_space = 0 : i32, alignment = 4 : i64, sym_visibility = "private"} : i32
  llvm.func local_unnamed_addr @get_new_matrix(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias}) attributes {memory = #llvm.memory_effects<other = write, argMem = none, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nofree", "nounwind", "ssp", "willreturn", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(16 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(3 : i64) : i64
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.call @malloc(%0) : (i64) -> !llvm.ptr
    llvm.store %arg0, %5 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
    %6 = llvm.getelementptr inbounds %5[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.store %arg1, %6 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
    %7 = llvm.mul %arg1, %arg0  : i32
    %8 = llvm.sext %7 : i32 to i64
    %9 = llvm.shl %8, %3  : i64
    %10 = llvm.call @malloc(%9) : (i64) -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %5[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.store %10, %11 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    llvm.return %5 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @malloc(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["allockind", "9"], ["allocsize", "4294967295"], ["alloc-family", "malloc"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"}
  llvm.func local_unnamed_addr @get_new_empty_matrix() -> (!llvm.ptr {llvm.noalias}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nofree", "nounwind", "ssp", "willreturn", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(16 : i64) : i64
    %2 = llvm.call @calloc(%0, %1) : (i64, i64) -> !llvm.ptr
    llvm.return %2 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @delete_matrix(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {passthrough = ["mustprogress", "nounwind", "ssp", "willreturn", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %4 = llvm.load %3 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %5 = llvm.icmp "eq" %4, %2 : !llvm.ptr
    llvm.cond_br %5, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @free(%4) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.call @free(%arg0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @free(!llvm.ptr {llvm.allocptr, llvm.nocapture, llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nounwind", "willreturn", ["allockind", "4"], ["alloc-family", "malloc"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"}
  llvm.func local_unnamed_addr @get_matrix_array(%arg0: i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias}) attributes {memory = #llvm.memory_effects<other = write, argMem = none, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nofree", "nounwind", "ssp", "willreturn", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(false) : i1
    %4 = llvm.sext %arg0 : i32 to i64
    %5 = llvm.shl %4, %0  : i64
    %6 = llvm.call @malloc(%5) : (i64) -> !llvm.ptr
    %7 = llvm.icmp "sgt" %arg0, %1 : i32
    llvm.cond_br %7, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %8 = llvm.zext %arg0 : i32 to i64
    %9 = llvm.shl %8, %0  : i64
    "llvm.intr.memset"(%6, %2, %9) <{isVolatile = false, tbaa = [#tbaa_tag]}> : (!llvm.ptr, i8, i64) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return %6 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @delete_light_matrix_array(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: i32 {llvm.noundef}) attributes {passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
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
    %9 = llvm.load %8 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %10 = llvm.icmp "eq" %9, %3 : !llvm.ptr
    llvm.cond_br %10, ^bb4, ^bb3
  ^bb3:  // pred: ^bb2
    llvm.call @free(%9) : (!llvm.ptr) -> ()
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %11 = llvm.add %7, %4  : i64
    %12 = llvm.icmp "eq" %11, %6 : i64
    llvm.cond_br %12, ^bb5, ^bb2(%11 : i64) {loop_annotation = #loop_annotation}
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.call @free(%arg0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @resize(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "nounwind", "ssp", "willreturn", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(3 : i64) : i64
    %6 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %7 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %8 = llvm.load %7 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %9 = llvm.mul %8, %6  : i32
    %10 = llvm.mul %arg2, %arg1  : i32
    %11 = llvm.icmp "eq" %9, %10 : i32
    llvm.cond_br %11, ^bb6, ^bb1
  ^bb1:  // pred: ^bb0
    %12 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %13 = llvm.load %12 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %14 = llvm.icmp "eq" %13, %3 : !llvm.ptr
    llvm.cond_br %14, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    llvm.call @free(%13) : (!llvm.ptr) -> ()
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %15 = llvm.icmp "sgt" %10, %4 : i32
    llvm.cond_br %15, ^bb4, ^bb5(%3 : !llvm.ptr)
  ^bb4:  // pred: ^bb3
    %16 = llvm.zext %10 : i32 to i64
    %17 = llvm.shl %16, %5  : i64
    %18 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
    llvm.br ^bb5(%18 : !llvm.ptr)
  ^bb5(%19: !llvm.ptr):  // 2 preds: ^bb3, ^bb4
    llvm.store %19, %12 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb0, ^bb5
    llvm.store %arg2, %7 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
    llvm.store %arg1, %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @set_identity(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory = #llvm.memory_effects<other = write, argMem = readwrite, inaccessibleMem = none>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %5 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %8 = llvm.load %7 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %9 = llvm.icmp "sgt" %8, %2 : i32
    llvm.cond_br %9, ^bb1, ^bb6
  ^bb1:  // pred: ^bb0
    %10 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %11 = llvm.icmp "sgt" %10, %2 : i32
    %12 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %13 = llvm.sext %10 : i32 to i64
    %14 = llvm.zext %8 : i32 to i64
    %15 = llvm.zext %10 : i32 to i64
    llvm.br ^bb2(%0 : i64)
  ^bb2(%16: i64):  // 2 preds: ^bb1, ^bb5
    llvm.cond_br %11, ^bb3, ^bb5
  ^bb3:  // pred: ^bb2
    %17 = llvm.mul %16, %13  : i64
    %18 = llvm.load %12 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %19 = llvm.getelementptr %18[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb4(%0 : i64)
  ^bb4(%20: i64):  // 2 preds: ^bb3, ^bb4
    %21 = llvm.icmp "eq" %16, %20 : i64
    %22 = llvm.select %21, %4, %5 : i1, f64
    %23 = llvm.getelementptr %19[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %22, %23 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %24 = llvm.add %20, %6  : i64
    %25 = llvm.icmp "eq" %24, %15 : i64
    llvm.cond_br %25, ^bb5, ^bb4(%24 : i64) {loop_annotation = #loop_annotation}
  ^bb5:  // 2 preds: ^bb2, ^bb4
    %26 = llvm.add %16, %6  : i64
    %27 = llvm.icmp "eq" %26, %14 : i64
    llvm.cond_br %27, ^bb6, ^bb2(%26 : i64) {loop_annotation = #loop_annotation}
  ^bb6:  // 2 preds: ^bb0, ^bb5
    llvm.return
  }
  llvm.func local_unnamed_addr @fill(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = write, argMem = readwrite, inaccessibleMem = none>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %6 = llvm.load %5 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %7 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %8 = llvm.mul %7, %6  : i32
    %9 = llvm.icmp "sgt" %8, %2 : i32
    llvm.cond_br %9, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %10 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %11 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %12 = llvm.zext %8 : i32 to i64
    llvm.br ^bb2(%0 : i64)
  ^bb2(%13: i64):  // 2 preds: ^bb1, ^bb2
    %14 = llvm.getelementptr inbounds %11[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %arg1, %14 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %15 = llvm.add %13, %4  : i64
    %16 = llvm.icmp "eq" %15, %12 : i64
    llvm.cond_br %16, ^bb3, ^bb2(%15 : i64) {loop_annotation = #loop_annotation}
  ^bb3:  // 2 preds: ^bb0, ^bb2
    llvm.return
  }
  llvm.func local_unnamed_addr @set_block(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory = #llvm.memory_effects<other = readwrite, argMem = readwrite, inaccessibleMem = none>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.getelementptr inbounds %arg3[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %6 = llvm.load %5 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %7 = llvm.icmp "sgt" %6, %2 : i32
    llvm.cond_br %7, ^bb1, ^bb6
  ^bb1:  // pred: ^bb0
    %8 = llvm.load %arg3 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %9 = llvm.icmp "sgt" %8, %2 : i32
    %10 = llvm.getelementptr inbounds %arg3[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %11 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %12 = llvm.sext %arg1 : i32 to i64
    %13 = llvm.sext %8 : i32 to i64
    %14 = llvm.sext %arg2 : i32 to i64
    %15 = llvm.zext %6 : i32 to i64
    %16 = llvm.zext %8 : i32 to i64
    llvm.br ^bb2(%0 : i64)
  ^bb2(%17: i64):  // 2 preds: ^bb1, ^bb5
    llvm.cond_br %9, ^bb3, ^bb5
  ^bb3:  // pred: ^bb2
    %18 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %19 = llvm.mul %17, %13  : i64
    %20 = llvm.load %11 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %21 = llvm.add %17, %14  : i64
    %22 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %23 = llvm.sext %22 : i32 to i64
    %24 = llvm.mul %21, %23  : i64
    %25 = llvm.getelementptr %18[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %26 = llvm.getelementptr %20[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %27 = llvm.getelementptr %26[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb4(%0 : i64)
  ^bb4(%28: i64):  // 2 preds: ^bb3, ^bb4
    %29 = llvm.getelementptr %25[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %30 = llvm.load %29 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %31 = llvm.getelementptr %27[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %30, %31 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %32 = llvm.add %28, %4  : i64
    %33 = llvm.icmp "eq" %32, %16 : i64
    llvm.cond_br %33, ^bb5, ^bb4(%32 : i64) {loop_annotation = #loop_annotation}
  ^bb5:  // 2 preds: ^bb2, ^bb4
    %34 = llvm.add %17, %4  : i64
    %35 = llvm.icmp "eq" %34, %15 : i64
    llvm.cond_br %35, ^bb6, ^bb2(%34 : i64) {loop_annotation = #loop_annotation}
  ^bb6:  // 2 preds: ^bb0, ^bb5
    llvm.return
  }
  llvm.func local_unnamed_addr @copy(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(3 : i64) : i64
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %8 = llvm.load %7 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %9 = llvm.icmp "eq" %8, %2 : !llvm.ptr
    llvm.cond_br %9, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @free(%8) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %10 = llvm.getelementptr inbounds %arg1[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %11 = llvm.load %10 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %12 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.store %11, %12 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
    %13 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    llvm.store %13, %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
    %14 = llvm.mul %13, %11  : i32
    %15 = llvm.sext %14 : i32 to i64
    %16 = llvm.shl %15, %4  : i64
    %17 = llvm.call @malloc(%16) : (i64) -> !llvm.ptr
    llvm.store %17, %7 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %18 = llvm.icmp "sgt" %14, %5 : i32
    llvm.cond_br %18, ^bb3, ^bb5
  ^bb3:  // pred: ^bb2
    %19 = llvm.getelementptr inbounds %arg1[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %20 = llvm.load %19 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %21 = llvm.zext %14 : i32 to i64
    llvm.br ^bb4(%0 : i64)
  ^bb4(%22: i64):  // 2 preds: ^bb3, ^bb4
    %23 = llvm.getelementptr inbounds %20[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %24 = llvm.load %23 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %25 = llvm.getelementptr inbounds %17[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %24, %25 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %26 = llvm.add %22, %6  : i64
    %27 = llvm.icmp "eq" %26, %21 : i64
    llvm.cond_br %27, ^bb5, ^bb4(%26 : i64) {loop_annotation = #loop_annotation}
  ^bb5:  // 2 preds: ^bb2, ^bb4
    llvm.return
  }
  llvm.func local_unnamed_addr @square_sum(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> f64 attributes {memory = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %3 = llvm.fmul %2, %2  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %4 = llvm.icmp "sgt" %arg0, %0 : i32
    llvm.cond_br %4, ^bb1, ^bb3(%3 : f64)
  ^bb1:  // pred: ^bb0
    %5 = llvm.zext %arg0 : i32 to i64
    llvm.br ^bb2(%1, %3 : i64, f64)
  ^bb2(%6: i64, %7: f64):  // 2 preds: ^bb1, ^bb2
    %8 = llvm.getelementptr inbounds %arg1[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %9 = llvm.load %8 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %10 = llvm.fmul %9, %9  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %11 = llvm.fadd %10, %7  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %12 = llvm.add %6, %1  : i64
    %13 = llvm.icmp "eq" %12, %5 : i64
    llvm.cond_br %13, ^bb3(%11 : f64), ^bb2(%12, %11 : i64, f64) {loop_annotation = #loop_annotation}
  ^bb3(%14: f64):  // 2 preds: ^bb0, ^bb2
    llvm.return %14 : f64
  }
  llvm.func local_unnamed_addr @angle_axis_to_rotation_matrix(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory = #llvm.memory_effects<other = write, argMem = readwrite, inaccessibleMem = none>, passthrough = ["nofree", "nosync", "nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
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
    %18 = llvm.add %12, %0  : i64
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
    %28 = llvm.sext %25 : i32 to i64
    %29 = llvm.zext %23 : i32 to i64
    %30 = llvm.zext %25 : i32 to i64
    llvm.br ^bb5(%5 : i64)
  ^bb5(%31: i64):  // 2 preds: ^bb4, ^bb8
    llvm.cond_br %26, ^bb6, ^bb8
  ^bb6:  // pred: ^bb5
    %32 = llvm.mul %31, %28  : i64
    %33 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %34 = llvm.getelementptr %33[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb7(%5 : i64)
  ^bb7(%35: i64):  // 2 preds: ^bb6, ^bb7
    %36 = llvm.icmp "eq" %31, %35 : i64
    %37 = llvm.select %36, %4, %9 : i1, f64
    %38 = llvm.getelementptr %34[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %37, %38 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %39 = llvm.add %35, %0  : i64
    %40 = llvm.icmp "eq" %39, %30 : i64
    llvm.cond_br %40, ^bb8, ^bb7(%39 : i64) {loop_annotation = #loop_annotation}
  ^bb8:  // 2 preds: ^bb5, ^bb7
    %41 = llvm.add %31, %0  : i64
    %42 = llvm.icmp "eq" %41, %29 : i64
    llvm.cond_br %42, ^bb10, ^bb5(%41 : i64) {loop_annotation = #loop_annotation}
  ^bb9:  // pred: ^bb2
    %43 = llvm.fdiv %10, %20  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %44 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %45 = llvm.load %44 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %46 = llvm.fdiv %45, %20  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %47 = llvm.getelementptr inbounds %arg0[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %48 = llvm.load %47 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %49 = llvm.fdiv %48, %20  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %50 = llvm.intr.sin(%20)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    %51 = llvm.intr.cos(%20)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    %52 = llvm.fmul %43, %43  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %53 = llvm.fsub %4, %52  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %54 = llvm.fmul %53, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %55 = llvm.fadd %54, %52  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %56 = llvm.getelementptr inbounds %arg1[%5, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %57 = llvm.load %56 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    llvm.store %55, %57 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %58 = llvm.fsub %4, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %59 = llvm.fmul %58, %43  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %60 = llvm.fmul %59, %46  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %61 = llvm.fmul %49, %50  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %62 = llvm.fsub %60, %61  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %63 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %64 = llvm.sext %63 : i32 to i64
    %65 = llvm.getelementptr inbounds %57[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %62, %65 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %66 = llvm.fmul %59, %49  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %67 = llvm.fmul %46, %50  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %68 = llvm.fadd %66, %67  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %69 = llvm.shl %63, %7  : i32
    %70 = llvm.sext %69 : i32 to i64
    %71 = llvm.getelementptr inbounds %57[%70] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %68, %71 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %72 = llvm.fadd %60, %61  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %73 = llvm.getelementptr inbounds %57[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %72, %73 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %74 = llvm.fmul %46, %46  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %75 = llvm.fsub %4, %74  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %76 = llvm.fmul %75, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %77 = llvm.fadd %76, %74  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %78 = llvm.add %63, %7  : i32
    %79 = llvm.sext %78 : i32 to i64
    %80 = llvm.getelementptr inbounds %57[%79] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %77, %80 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %81 = llvm.fmul %46, %58  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %82 = llvm.fmul %81, %49  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %83 = llvm.fmul %43, %50  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %84 = llvm.fsub %82, %83  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %85 = llvm.or %69, %7  : i32
    %86 = llvm.sext %85 : i32 to i64
    %87 = llvm.getelementptr inbounds %57[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %84, %87 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %88 = llvm.fsub %66, %67  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %89 = llvm.getelementptr inbounds %57[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %88, %89 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %90 = llvm.fadd %82, %83  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %91 = llvm.add %63, %6  : i32
    %92 = llvm.sext %91 : i32 to i64
    %93 = llvm.getelementptr inbounds %57[%92] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %90, %93 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %94 = llvm.fmul %49, %49  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %95 = llvm.fsub %4, %94  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %96 = llvm.fmul %95, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %97 = llvm.fadd %96, %94  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %98 = llvm.add %69, %6  : i32
    %99 = llvm.sext %98 : i32 to i64
    %100 = llvm.getelementptr inbounds %57[%99] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %97, %100 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    llvm.br ^bb10
  ^bb10:  // 3 preds: ^bb3, ^bb8, ^bb9
    llvm.return
  }
  llvm.func local_unnamed_addr @apply_global_transform(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
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
    %10 = llvm.mlir.constant(4294967295 : i64) : i64
    %11 = llvm.call @malloc(%0) : (i64) -> !llvm.ptr
    llvm.store %1, %11 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
    %12 = llvm.getelementptr inbounds %11[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.store %1, %12 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
    %13 = llvm.call @malloc(%4) : (i64) -> !llvm.ptr
    %14 = llvm.getelementptr inbounds %11[%2, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.store %13, %14 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %15 = llvm.getelementptr inbounds %arg0[%2, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %16 = llvm.load %15 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    llvm.call @angle_axis_to_rotation_matrix(%16, %11) : (!llvm.ptr, !llvm.ptr) -> ()
    %17 = llvm.load %15 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %19 = llvm.sext %18 : i32 to i64
    %20 = llvm.getelementptr %17[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb1(%2 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb3
    %22 = llvm.getelementptr %20[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %23 = llvm.mul %21, %6  : i64
    %24 = llvm.getelementptr %13[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb2(%2 : i64)
  ^bb2(%25: i64):  // 2 preds: ^bb1, ^bb2
    %26 = llvm.load %22 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %27 = llvm.getelementptr %24[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %28 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %29 = llvm.fmul %28, %26  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %29, %27 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %30 = llvm.add %25, %7  : i64
    %31 = llvm.icmp "eq" %30, %6 : i64
    llvm.cond_br %31, ^bb3, ^bb2(%30 : i64) {loop_annotation = #loop_annotation}
  ^bb3:  // pred: ^bb2
    %32 = llvm.add %21, %7  : i64
    %33 = llvm.icmp "eq" %32, %6 : i64
    llvm.cond_br %33, ^bb4, ^bb1(%32 : i64) {loop_annotation = #loop_annotation}
  ^bb4:  // pred: ^bb3
    llvm.intr.experimental.noalias.scope.decl #alias_scope
    %34 = llvm.getelementptr inbounds %arg1[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %35 = llvm.load %34 {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %36 = llvm.icmp "sgt" %35, %8 : i32
    llvm.cond_br %36, ^bb5, ^bb6(%9 : !llvm.ptr)
  ^bb5:  // pred: ^bb4
    %37 = llvm.mul %35, %1  : i32
    %38 = llvm.zext %37 : i32 to i64
    %39 = llvm.shl %38, %6  : i64
    %40 = llvm.call @malloc(%39) : (i64) -> !llvm.ptr
    llvm.br ^bb6(%40 : !llvm.ptr)
  ^bb6(%41: !llvm.ptr):  // 2 preds: ^bb4, ^bb5
    %42 = llvm.icmp "sgt" %35, %8 : i32
    %43 = llvm.getelementptr inbounds %arg1[%2, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %44 = llvm.zext %35 : i32 to i64
    llvm.br ^bb7(%2 : i64)
  ^bb7(%45: i64):  // 2 preds: ^bb6, ^bb12
    llvm.cond_br %42, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    %46 = llvm.getelementptr inbounds %13[%45] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %47 = llvm.load %43 {alias_scopes = [#alias_scope], alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %48 = llvm.load %arg1 {alias_scopes = [#alias_scope], alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %49 = llvm.sext %48 : i32 to i64
    %50 = llvm.getelementptr %41[%45] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %51 = llvm.load %46 {alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    llvm.br ^bb9(%2 : i64)
  ^bb9(%52: i64):  // 2 preds: ^bb8, ^bb11
    %53 = llvm.mul %52, %49  : i64
    %54 = llvm.getelementptr inbounds %47[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %55 = llvm.load %54 {alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %56 = llvm.fmul %55, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %57 = llvm.mul %52, %6  : i64
    %58 = llvm.getelementptr %50[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %56, %58 {alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    llvm.br ^bb10(%7, %56 : i64, f64)
  ^bb10(%59: i64, %60: f64):  // 2 preds: ^bb9, ^bb10
    %61 = llvm.mul %59, %6  : i64
    %62 = llvm.getelementptr %46[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %63 = llvm.load %62 {alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %64 = llvm.getelementptr %54[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %65 = llvm.load %64 {alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %66 = llvm.fmul %65, %63  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %67 = llvm.fadd %66, %60  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %67, %58 {alignment = 8 : i64, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %68 = llvm.add %59, %7  : i64
    %69 = llvm.icmp "eq" %68, %6 : i64
    llvm.cond_br %69, ^bb11, ^bb10(%68, %67 : i64, f64) {loop_annotation = #loop_annotation}
  ^bb11:  // pred: ^bb10
    %70 = llvm.add %52, %7  : i64
    %71 = llvm.icmp "eq" %70, %44 : i64
    llvm.cond_br %71, ^bb12, ^bb9(%70 : i64) {loop_annotation = #loop_annotation}
  ^bb12:  // 2 preds: ^bb7, ^bb11
    %72 = llvm.add %45, %7  : i64
    %73 = llvm.icmp "eq" %72, %6 : i64
    llvm.cond_br %73, ^bb13, ^bb7(%72 : i64) {loop_annotation = #loop_annotation}
  ^bb13:  // pred: ^bb12
    %74 = llvm.load %34 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %75 = llvm.icmp "sgt" %74, %8 : i32
    llvm.cond_br %75, ^bb14, ^bb19
  ^bb14:  // pred: ^bb13
    %76 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %77 = llvm.icmp "sgt" %76, %8 : i32
    %78 = llvm.sext %76 : i32 to i64
    %79 = llvm.zext %74 : i32 to i64
    %80 = llvm.shl %18, %3  : i32
    %81 = llvm.sext %80 : i32 to i64
    %82 = llvm.zext %76 : i32 to i64
    %83 = llvm.getelementptr %17[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb15(%2 : i64)
  ^bb15(%84: i64):  // 2 preds: ^bb14, ^bb18
    llvm.cond_br %77, ^bb16, ^bb18
  ^bb16:  // pred: ^bb15
    %85 = llvm.mul %84, %6  : i64
    %86 = llvm.load %43 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %87 = llvm.mul %84, %78  : i64
    %88 = llvm.getelementptr %86[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb17(%2 : i64)
  ^bb17(%89: i64):  // 2 preds: ^bb16, ^bb17
    %90 = llvm.add %89, %85  : i64
    %91 = llvm.and %90, %10  : i64
    %92 = llvm.getelementptr inbounds %41[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %93 = llvm.load %92 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %94 = llvm.getelementptr %83[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %95 = llvm.load %94 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %96 = llvm.fadd %95, %93  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %97 = llvm.getelementptr %88[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %96, %97 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %98 = llvm.add %89, %7  : i64
    %99 = llvm.icmp "eq" %98, %82 : i64
    llvm.cond_br %99, ^bb18, ^bb17(%98 : i64) {loop_annotation = #loop_annotation}
  ^bb18:  // 2 preds: ^bb15, ^bb17
    %100 = llvm.add %84, %7  : i64
    %101 = llvm.icmp "eq" %100, %79 : i64
    llvm.cond_br %101, ^bb19, ^bb15(%100 : i64) {loop_annotation = #loop_annotation}
  ^bb19:  // 2 preds: ^bb13, ^bb18
    %102 = llvm.icmp "eq" %13, %9 : !llvm.ptr
    llvm.cond_br %102, ^bb21, ^bb20
  ^bb20:  // pred: ^bb19
    llvm.call @free(%13) : (!llvm.ptr) -> ()
    llvm.br ^bb21
  ^bb21:  // 2 preds: ^bb19, ^bb20
    llvm.call @free(%11) : (!llvm.ptr) -> ()
    %103 = llvm.icmp "eq" %41, %9 : !llvm.ptr
    llvm.cond_br %103, ^bb23, ^bb22
  ^bb22:  // pred: ^bb21
    llvm.call @free(%41) : (!llvm.ptr) -> ()
    llvm.br ^bb23
  ^bb23:  // 2 preds: ^bb21, ^bb22
    llvm.return
  }
  llvm.func local_unnamed_addr @relatives_to_absolutes(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
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
    %23 = llvm.mul %22, %20  : i32
    %24 = llvm.sext %23 : i32 to i64
    %25 = llvm.shl %24, %6  : i64
    %26 = llvm.call @malloc(%25) : (i64) -> !llvm.ptr
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
    %35 = llvm.add %31, %7  : i64
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
    %47 = llvm.mul %46, %44  : i32
    %48 = llvm.mul %43, %41  : i32
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
    %55 = llvm.shl %54, %6  : i64
    %56 = llvm.call @malloc(%55) : (i64) -> !llvm.ptr
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
    %79 = llvm.mul %77, %74  : i64
    %80 = llvm.getelementptr inbounds %69[%79] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %81 = llvm.load %80 {alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %82 = llvm.fmul %81, %78  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %83 = llvm.mul %77, %64  : i64
    %84 = llvm.getelementptr %75[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %82, %84 {alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    llvm.cond_br %73, ^bb19(%7, %82 : i64, f64), ^bb20
  ^bb19(%85: i64, %86: f64):  // 2 preds: ^bb18, ^bb19
    %87 = llvm.mul %85, %64  : i64
    %88 = llvm.getelementptr %68[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %89 = llvm.load %88 {alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %90 = llvm.getelementptr %80[%85] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %91 = llvm.load %90 {alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %92 = llvm.fmul %91, %89  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %93 = llvm.fadd %92, %86  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %93, %84 {alignment = 8 : i64, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %94 = llvm.add %85, %7  : i64
    %95 = llvm.icmp "eq" %94, %76 : i64
    llvm.cond_br %95, ^bb20, ^bb19(%94, %93 : i64, f64) {loop_annotation = #loop_annotation}
  ^bb20:  // 2 preds: ^bb18, ^bb19
    %96 = llvm.add %77, %7  : i64
    %97 = llvm.icmp "eq" %96, %65 : i64
    llvm.cond_br %97, ^bb21, ^bb18(%96 : i64) {loop_annotation = #loop_annotation}
  ^bb21:  // 2 preds: ^bb16, ^bb20
    %98 = llvm.add %66, %7  : i64
    %99 = llvm.icmp "eq" %98, %64 : i64
    llvm.cond_br %99, ^bb22, ^bb16(%98 : i64) {loop_annotation = #loop_annotation}
  ^bb22:  // 4 preds: ^bb5, ^bb7, ^bb14, ^bb21
    %100 = llvm.add %10, %7  : i64
    %101 = llvm.icmp "eq" %100, %9 : i64
    llvm.cond_br %101, ^bb23, ^bb2(%100 : i64) {loop_annotation = #loop_annotation}
  ^bb23:  // 2 preds: ^bb0, ^bb22
    llvm.return
  }
  llvm.func local_unnamed_addr @euler_angles_to_rotation_matrix(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}) attributes {passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
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
    %22 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
    %23 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
    %24 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
    llvm.br ^bb1(%3 : i64)
  ^bb1(%25: i64):  // 2 preds: ^bb0, ^bb3
    %26 = llvm.mul %25, %4  : i64
    %27 = llvm.getelementptr %22[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb2(%3 : i64)
  ^bb2(%28: i64):  // 2 preds: ^bb1, ^bb2
    %29 = llvm.icmp "eq" %25, %28 : i64
    %30 = llvm.select %29, %5, %6 : i1, f64
    %31 = llvm.getelementptr %27[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %30, %31 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %32 = llvm.add %28, %1  : i64
    %33 = llvm.icmp "eq" %32, %4 : i64
    llvm.cond_br %33, ^bb3, ^bb2(%32 : i64) {loop_annotation = #loop_annotation}
  ^bb3:  // pred: ^bb2
    %34 = llvm.add %25, %1  : i64
    %35 = llvm.icmp "eq" %34, %4 : i64
    llvm.cond_br %35, ^bb4, ^bb1(%34 : i64) {loop_annotation = #loop_annotation}
  ^bb4:  // pred: ^bb3
    %36 = llvm.intr.cos(%17)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    %37 = llvm.getelementptr inbounds %22[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %36, %37 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %38 = llvm.intr.sin(%17)  {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
    %39 = llvm.getelementptr inbounds %22[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %38, %39 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %40 = llvm.fneg %38  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %41 = llvm.getelementptr inbounds %22[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %40, %41 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %42 = llvm.getelementptr inbounds %22[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %36, %42 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    llvm.br ^bb5(%3 : i64)
  ^bb5(%43: i64):  // 2 preds: ^bb4, ^bb7
    %44 = llvm.mul %43, %4  : i64
    %45 = llvm.getelementptr %23[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb6(%3 : i64)
  ^bb6(%46: i64):  // 2 preds: ^bb5, ^bb6
    %47 = llvm.icmp "eq" %43, %46 : i64
    %48 = llvm.select %47, %5, %6 : i1, f64
    %49 = llvm.getelementptr %45[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %48, %49 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %50 = llvm.add %46, %1  : i64
    %51 = llvm.icmp "eq" %50, %4 : i64
    llvm.cond_br %51, ^bb7, ^bb6(%50 : i64) {loop_annotation = #loop_annotation}
  ^bb7:  // pred: ^bb6
    %52 = llvm.add %43, %1  : i64
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
    %59 = llvm.getelementptr inbounds %23[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %54, %59 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    llvm.br ^bb9(%3 : i64)
  ^bb9(%60: i64):  // 2 preds: ^bb8, ^bb11
    %61 = llvm.mul %60, %4  : i64
    %62 = llvm.getelementptr %24[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb10(%3 : i64)
  ^bb10(%63: i64):  // 2 preds: ^bb9, ^bb10
    %64 = llvm.icmp "eq" %60, %63 : i64
    %65 = llvm.select %64, %5, %6 : i1, f64
    %66 = llvm.getelementptr %62[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %65, %66 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %67 = llvm.add %63, %1  : i64
    %68 = llvm.icmp "eq" %67, %4 : i64
    llvm.cond_br %68, ^bb11, ^bb10(%67 : i64) {loop_annotation = #loop_annotation}
  ^bb11:  // pred: ^bb10
    %69 = llvm.add %60, %1  : i64
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
    %76 = llvm.getelementptr inbounds %24[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %71, %76 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %77 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
    llvm.br ^bb13(%3 : i64)
  ^bb13(%78: i64):  // 2 preds: ^bb12, ^bb17
    %79 = llvm.getelementptr inbounds %24[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %80 = llvm.getelementptr %77[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %81 = llvm.load %79 {alignment = 8 : i64, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    llvm.br ^bb14(%3 : i64)
  ^bb14(%82: i64):  // 2 preds: ^bb13, ^bb16
    %83 = llvm.mul %82, %4  : i64
    %84 = llvm.getelementptr inbounds %23[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %85 = llvm.load %84 {alignment = 8 : i64, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %86 = llvm.fmul %85, %81  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %87 = llvm.getelementptr %80[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb15(%1, %86 : i64, f64)
  ^bb15(%88: i64, %89: f64):  // 2 preds: ^bb14, ^bb15
    %90 = llvm.mul %88, %4  : i64
    %91 = llvm.getelementptr %79[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %92 = llvm.load %91 {alignment = 8 : i64, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %93 = llvm.getelementptr %84[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %94 = llvm.load %93 {alignment = 8 : i64, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %95 = llvm.fmul %94, %92  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %96 = llvm.fadd %95, %89  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %97 = llvm.add %88, %1  : i64
    %98 = llvm.icmp "eq" %97, %4 : i64
    llvm.cond_br %98, ^bb16, ^bb15(%97, %96 : i64, f64) {loop_annotation = #loop_annotation}
  ^bb16:  // pred: ^bb15
    llvm.store %96, %87 {alignment = 8 : i64, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %99 = llvm.add %82, %1  : i64
    %100 = llvm.icmp "eq" %99, %4 : i64
    llvm.cond_br %100, ^bb17, ^bb14(%99 : i64) {loop_annotation = #loop_annotation}
  ^bb17:  // pred: ^bb16
    %101 = llvm.add %78, %1  : i64
    %102 = llvm.icmp "eq" %101, %4 : i64
    llvm.cond_br %102, ^bb18, ^bb13(%101 : i64) {loop_annotation = #loop_annotation}
  ^bb18:  // pred: ^bb17
    llvm.intr.experimental.noalias.scope.decl #alias_scope9
    %103 = llvm.load %arg1 {alias_scopes = [#alias_scope9], alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %104 = llvm.getelementptr inbounds %arg1[%3, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %105 = llvm.load %104 {alias_scopes = [#alias_scope9], alignment = 4 : i64, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %106 = llvm.mul %105, %103  : i32
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
    %111 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
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
    %118 = llvm.mul %117, %4  : i64
    %119 = llvm.getelementptr inbounds %22[%118] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %120 = llvm.load %119 {alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %121 = llvm.fmul %120, %116  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %122 = llvm.getelementptr %115[%118] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %121, %122 {alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    llvm.br ^bb25(%1, %121 : i64, f64)
  ^bb25(%123: i64, %124: f64):  // 2 preds: ^bb24, ^bb25
    %125 = llvm.mul %123, %4  : i64
    %126 = llvm.getelementptr %114[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %127 = llvm.load %126 {alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %128 = llvm.getelementptr %119[%123] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %129 = llvm.load %128 {alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %130 = llvm.fmul %129, %127  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %131 = llvm.fadd %130, %124  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %132 = llvm.add %123, %1  : i64
    %133 = llvm.icmp "eq" %132, %4 : i64
    llvm.cond_br %133, ^bb26, ^bb25(%132, %131 : i64, f64) {loop_annotation = #loop_annotation}
  ^bb26:  // pred: ^bb25
    llvm.store %131, %122 {alignment = 8 : i64, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %134 = llvm.add %117, %1  : i64
    %135 = llvm.icmp "eq" %134, %4 : i64
    llvm.cond_br %135, ^bb27, ^bb24(%134 : i64) {loop_annotation = #loop_annotation}
  ^bb27:  // pred: ^bb26
    %136 = llvm.add %113, %1  : i64
    %137 = llvm.icmp "eq" %136, %4 : i64
    llvm.cond_br %137, ^bb28, ^bb23(%136 : i64) {loop_annotation = #loop_annotation}
  ^bb28:  // pred: ^bb27
    llvm.call @free(%22) : (!llvm.ptr) -> ()
    llvm.call @free(%23) : (!llvm.ptr) -> ()
    llvm.call @free(%24) : (!llvm.ptr) -> ()
    llvm.call @free(%77) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @get_posed_relatives(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {enzyme.tag = "posed_relatives base_rels", llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}) attributes {passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
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
    %15 = llvm.mlir.constant(false) : i1
    %16 = llvm.mlir.zero : !llvm.ptr
    %17 = llvm.mlir.constant(4 : i32) : i32
    %18 = llvm.call @malloc(%0) : (i64) -> !llvm.ptr
    %19 = llvm.call @malloc(%1) : (i64) -> !llvm.ptr
    llvm.store %2, %19 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
    %20 = llvm.getelementptr inbounds %19[%3, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.store %2, %20 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
    %21 = llvm.call @malloc(%5) : (i64) -> !llvm.ptr
    %22 = llvm.getelementptr inbounds %19[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.store %21, %22 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %23 = llvm.icmp "sgt" %arg0, %7 : i32
    llvm.cond_br %23, ^bb1, ^bb25
  ^bb1:  // pred: ^bb0
    %24 = llvm.getelementptr inbounds %arg2[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %25 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %26 = llvm.load %arg2 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %27 = llvm.sext %26 : i32 to i64
    %28 = llvm.zext %arg0 : i32 to i64
    llvm.br ^bb2(%3 : i64)
  ^bb2(%29: i64):  // 2 preds: ^bb1, ^bb24
    llvm.br ^bb3(%3 : i64)
  ^bb3(%30: i64):  // 2 preds: ^bb2, ^bb5
    %31 = llvm.shl %30, %8  : i64
    %32 = llvm.getelementptr %18[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb4(%3 : i64)
  ^bb4(%33: i64):  // 2 preds: ^bb3, ^bb4
    %34 = llvm.icmp "eq" %30, %33 : i64
    %35 = llvm.select %34, %9, %10 : i1, f64
    %36 = llvm.getelementptr %32[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %35, %36 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %37 = llvm.add %33, %11  : i64
    %38 = llvm.icmp "eq" %37, %12 : i64
    llvm.cond_br %38, ^bb5, ^bb4(%37 : i64) {loop_annotation = #loop_annotation}
  ^bb5:  // pred: ^bb4
    %39 = llvm.add %30, %11  : i64
    %40 = llvm.icmp "eq" %39, %12 : i64
    llvm.cond_br %40, ^bb6, ^bb3(%39 : i64) {loop_annotation = #loop_annotation}
  ^bb6:  // pred: ^bb5
    %41 = llvm.add %29, %13  : i64
    %42 = llvm.mul %41, %27  : i64
    %43 = llvm.getelementptr inbounds %25[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.call @euler_angles_to_rotation_matrix(%43, %19) : (!llvm.ptr, !llvm.ptr) -> ()
    %44 = llvm.load %20 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %45 = llvm.icmp "sgt" %44, %7 : i32
    llvm.cond_br %45, ^bb7, ^bb11
  ^bb7:  // pred: ^bb6
    %46 = llvm.load %19 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %47 = llvm.icmp "sgt" %46, %7 : i32
    %48 = llvm.sext %46 : i32 to i64
    %49 = llvm.zext %44 : i32 to i64
    %50 = llvm.zext %46 : i32 to i64
    %51 = llvm.shl %50, %13  : i64
    llvm.br ^bb8(%3 : i64)
  ^bb8(%52: i64):  // 2 preds: ^bb7, ^bb10
    llvm.cond_br %47, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %53 = llvm.shl %52, %14  : i64
    %54 = llvm.getelementptr %18[%53] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %55 = llvm.load %22 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %56 = llvm.mul %52, %48  : i64
    %57 = llvm.getelementptr %55[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%54, %57, %51) <{isVolatile = false, tbaa = [#tbaa_tag1]}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    %58 = llvm.add %52, %11  : i64
    %59 = llvm.icmp "eq" %58, %49 : i64
    llvm.cond_br %59, ^bb11, ^bb8(%58 : i64) {loop_annotation = #loop_annotation}
  ^bb11:  // 2 preds: ^bb6, ^bb10
    %60 = llvm.getelementptr inbounds %arg1[%29] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %61 = llvm.getelementptr inbounds %arg3[%29] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.intr.experimental.noalias.scope.decl #alias_scope12
    llvm.intr.experimental.noalias.scope.decl #alias_scope13
    %62 = llvm.load %60 {alias_scopes = [#alias_scope12], alignment = 8 : i64, noalias_scopes = [#alias_scope14, #alias_scope13], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %63 = llvm.load %61 {alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %64 = llvm.getelementptr inbounds %arg3[%29, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %65 = llvm.load %64 {alias_scopes = [#alias_scope13], alignment = 4 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %66 = llvm.mul %65, %63  : i32
    %67 = llvm.shl %62, %6  : i32
    %68 = llvm.icmp "eq" %66, %67 : i32
    llvm.cond_br %68, ^bb17, ^bb12
  ^bb12:  // pred: ^bb11
    %69 = llvm.getelementptr inbounds %arg3[%29, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %70 = llvm.load %69 {alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %71 = llvm.icmp "eq" %70, %16 : !llvm.ptr
    llvm.cond_br %71, ^bb14, ^bb13
  ^bb13:  // pred: ^bb12
    llvm.call @free(%70) {noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13]} : (!llvm.ptr) -> ()
    llvm.br ^bb14
  ^bb14:  // 2 preds: ^bb12, ^bb13
    %72 = llvm.icmp "sgt" %62, %7 : i32
    llvm.cond_br %72, ^bb15, ^bb16(%16 : !llvm.ptr)
  ^bb15:  // pred: ^bb14
    %73 = llvm.zext %67 : i32 to i64
    %74 = llvm.shl %73, %13  : i64
    %75 = llvm.call @malloc(%74) : (i64) -> !llvm.ptr
    llvm.br ^bb16(%75 : !llvm.ptr)
  ^bb16(%76: !llvm.ptr):  // 2 preds: ^bb14, ^bb15
    llvm.store %76, %69 {alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb17
  ^bb17:  // 2 preds: ^bb11, ^bb16
    llvm.store %17, %64 {alias_scopes = [#alias_scope13], alignment = 4 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
    llvm.store %62, %61 {alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
    %77 = llvm.icmp "sgt" %62, %7 : i32
    llvm.cond_br %77, ^bb18, ^bb24
  ^bb18:  // pred: ^bb17
    %78 = llvm.getelementptr inbounds %arg1[%29, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %79 = llvm.getelementptr inbounds %arg3[%29, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %80 = llvm.getelementptr inbounds %arg1[%29, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %81 = llvm.zext %62 : i32 to i64
    %82 = llvm.load %78 {alias_scopes = [#alias_scope12], alignment = 8 : i64, noalias_scopes = [#alias_scope14, #alias_scope13], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %83 = llvm.load %79 {alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %84 = llvm.load %80 {alias_scopes = [#alias_scope12], alignment = 4 : i64, noalias_scopes = [#alias_scope14, #alias_scope13], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %85 = llvm.icmp "sgt" %84, %4 : i32
    %86 = llvm.zext %84 : i32 to i64
    llvm.br ^bb19(%3 : i64)
  ^bb19(%87: i64):  // 2 preds: ^bb18, ^bb23
    %88 = llvm.getelementptr inbounds %82[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %89 = llvm.getelementptr %83[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb20(%3 : i64)
  ^bb20(%90: i64):  // 2 preds: ^bb19, ^bb22
    %91 = llvm.load %88 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %92 = llvm.shl %90, %8  : i64
    %93 = llvm.getelementptr inbounds %18[%92] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %94 = llvm.load %93 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %95 = llvm.fmul %94, %91  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %96 = llvm.mul %90, %81  : i64
    %97 = llvm.getelementptr %89[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %95, %97 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    llvm.cond_br %85, ^bb21(%11, %95 : i64, f64), ^bb22
  ^bb21(%98: i64, %99: f64):  // 2 preds: ^bb20, ^bb21
    %100 = llvm.mul %98, %81  : i64
    %101 = llvm.getelementptr %88[%100] {debugme} : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %102 = llvm.load %101 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %103 = llvm.getelementptr %93[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %104 = llvm.load %103 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %105 = llvm.fmul %104, %102  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %106 = llvm.fadd %105, %99  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %106, %97 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %107 = llvm.add %98, %11  : i64
    %108 = llvm.icmp "eq" %107, %86 : i64
    llvm.cond_br %108, ^bb22, ^bb21(%107, %106 : i64, f64) {loop_annotation = #loop_annotation}
  ^bb22:  // 2 preds: ^bb20, ^bb21
    %109 = llvm.add %90, %11  : i64
    %110 = llvm.icmp "eq" %109, %12 : i64
    llvm.cond_br %110, ^bb23, ^bb20(%109 : i64) {loop_annotation = #loop_annotation}
  ^bb23:  // pred: ^bb22
    %111 = llvm.add %87, %11  : i64
    %112 = llvm.icmp "eq" %111, %81 : i64
    llvm.cond_br %112, ^bb24, ^bb19(%111 : i64) {loop_annotation = #loop_annotation}
  ^bb24:  // 2 preds: ^bb17, ^bb23
    %113 = llvm.add %29, %11  : i64
    %114 = llvm.icmp "eq" %113, %28 : i64
    llvm.cond_br %114, ^bb25, ^bb2(%113 : i64) {loop_annotation = #loop_annotation}
  ^bb25:  // 2 preds: ^bb0, ^bb24
    %115 = llvm.icmp "eq" %18, %16 : !llvm.ptr
    llvm.cond_br %115, ^bb27, ^bb26
  ^bb26:  // pred: ^bb25
    llvm.call @free(%18) : (!llvm.ptr) -> ()
    llvm.br ^bb27
  ^bb27:  // 2 preds: ^bb25, ^bb26
    %116 = llvm.load %22 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %117 = llvm.icmp "eq" %116, %16 : !llvm.ptr
    llvm.cond_br %117, ^bb29, ^bb28
  ^bb28:  // pred: ^bb27
    llvm.call @free(%116) : (!llvm.ptr) -> ()
    llvm.br ^bb29
  ^bb29:  // 2 preds: ^bb27, ^bb28
    llvm.call @free(%19) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @get_skinned_vertex_positions(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: i32 {llvm.noundef}, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}, %arg9: i32 {llvm.noundef}) attributes {passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(false) : i1
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.mlir.zero : !llvm.ptr
    %8 = llvm.mlir.constant(3 : i64) : i64
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.constant(3 : i32) : i32
    %11 = llvm.mlir.constant(16 : i64) : i64
    %12 = llvm.mlir.constant(4 : i32) : i32
    %13 = llvm.sext %arg0 : i32 to i64
    %14 = llvm.shl %13, %0  : i64
    %15 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
    %16 = llvm.icmp "sgt" %arg0, %1 : i32
    llvm.cond_br %16, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %17 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
    %18 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
    llvm.br ^bb3(%18, %17 : !llvm.ptr, !llvm.ptr)
  ^bb2:  // pred: ^bb0
    %19 = llvm.zext %arg0 : i32 to i64
    %20 = llvm.shl %19, %0  : i64
    "llvm.intr.memset"(%15, %2, %20) <{isVolatile = false, tbaa = [#tbaa_tag]}> : (!llvm.ptr, i8, i64) -> ()
    %21 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
    "llvm.intr.memset"(%21, %2, %20) <{isVolatile = false, tbaa = [#tbaa_tag]}> : (!llvm.ptr, i8, i64) -> ()
    %22 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
    "llvm.intr.memset"(%22, %2, %20) <{isVolatile = false, tbaa = [#tbaa_tag]}> : (!llvm.ptr, i8, i64) -> ()
    llvm.br ^bb3(%22, %21 : !llvm.ptr, !llvm.ptr)
  ^bb3(%23: !llvm.ptr, %24: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
    llvm.call @get_posed_relatives(%arg0, %arg1, %arg7, %15) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @relatives_to_absolutes(%arg0, %15, %arg2, %24) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.cond_br %16, ^bb4, ^bb20
  ^bb4:  // pred: ^bb3
    %25 = llvm.zext %arg0 : i32 to i64
    llvm.br ^bb5(%4 : i64)
  ^bb5(%26: i64):  // 2 preds: ^bb4, ^bb19
    %27 = llvm.getelementptr inbounds %24[%26] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %28 = llvm.getelementptr inbounds %arg3[%26] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %29 = llvm.getelementptr inbounds %23[%26] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.intr.experimental.noalias.scope.decl #alias_scope15
    llvm.intr.experimental.noalias.scope.decl #alias_scope16
    llvm.intr.experimental.noalias.scope.decl #alias_scope17
    %30 = llvm.load %27 {alias_scopes = [#alias_scope15], alignment = 8 : i64, noalias_scopes = [#alias_scope16, #alias_scope17], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %31 = llvm.getelementptr inbounds %arg3[%26, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %32 = llvm.load %31 {alias_scopes = [#alias_scope16], alignment = 4 : i64, noalias_scopes = [#alias_scope15, #alias_scope17], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %33 = llvm.load %29 {alias_scopes = [#alias_scope17], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %34 = llvm.getelementptr inbounds %23[%26, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %35 = llvm.load %34 {alias_scopes = [#alias_scope17], alignment = 4 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %36 = llvm.mul %35, %33  : i32
    %37 = llvm.mul %32, %30  : i32
    %38 = llvm.icmp "eq" %36, %37 : i32
    llvm.cond_br %38, ^bb11, ^bb6
  ^bb6:  // pred: ^bb5
    %39 = llvm.getelementptr inbounds %23[%26, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %40 = llvm.load %39 {alias_scopes = [#alias_scope17], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %41 = llvm.icmp "eq" %40, %7 : !llvm.ptr
    llvm.cond_br %41, ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    llvm.call @free(%40) {noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17]} : (!llvm.ptr) -> ()
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %42 = llvm.icmp "sgt" %37, %1 : i32
    llvm.cond_br %42, ^bb9, ^bb10(%7 : !llvm.ptr)
  ^bb9:  // pred: ^bb8
    %43 = llvm.zext %37 : i32 to i64
    %44 = llvm.shl %43, %8  : i64
    %45 = llvm.call @malloc(%44) : (i64) -> !llvm.ptr
    llvm.br ^bb10(%45 : !llvm.ptr)
  ^bb10(%46: !llvm.ptr):  // 2 preds: ^bb8, ^bb9
    llvm.store %46, %39 {alias_scopes = [#alias_scope17], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb5, ^bb10
    llvm.store %32, %34 {alias_scopes = [#alias_scope17], alignment = 4 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
    llvm.store %30, %29 {alias_scopes = [#alias_scope17], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
    %47 = llvm.icmp "sgt" %30, %1 : i32
    llvm.cond_br %47, ^bb12, ^bb19
  ^bb12:  // pred: ^bb11
    %48 = llvm.icmp "sgt" %32, %1 : i32
    %49 = llvm.getelementptr inbounds %24[%26, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %50 = llvm.getelementptr inbounds %arg3[%26, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %51 = llvm.getelementptr inbounds %23[%26, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %52 = llvm.getelementptr inbounds %24[%26, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %53 = llvm.zext %30 : i32 to i64
    %54 = llvm.zext %32 : i32 to i64
    llvm.br ^bb13(%4 : i64)
  ^bb13(%55: i64):  // 2 preds: ^bb12, ^bb18
    llvm.cond_br %48, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    %56 = llvm.load %49 {alias_scopes = [#alias_scope15], alignment = 8 : i64, noalias_scopes = [#alias_scope16, #alias_scope17], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %57 = llvm.getelementptr inbounds %56[%55] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %58 = llvm.load %50 {alias_scopes = [#alias_scope16], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope17], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %59 = llvm.load %28 {alias_scopes = [#alias_scope16], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope17], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %60 = llvm.load %51 {alias_scopes = [#alias_scope17], alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %61 = llvm.load %52 {alias_scopes = [#alias_scope15], alignment = 4 : i64, noalias_scopes = [#alias_scope16, #alias_scope17], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %62 = llvm.icmp "sgt" %61, %5 : i32
    %63 = llvm.sext %59 : i32 to i64
    %64 = llvm.getelementptr %60[%55] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %65 = llvm.zext %61 : i32 to i64
    llvm.br ^bb15(%4 : i64)
  ^bb15(%66: i64):  // 2 preds: ^bb14, ^bb17
    %67 = llvm.load %57 {alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %68 = llvm.mul %66, %63  : i64
    %69 = llvm.getelementptr inbounds %58[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %70 = llvm.load %69 {alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %71 = llvm.fmul %70, %67  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %72 = llvm.mul %66, %53  : i64
    %73 = llvm.getelementptr %64[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %71, %73 {alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    llvm.cond_br %62, ^bb16(%9, %71 : i64, f64), ^bb17
  ^bb16(%74: i64, %75: f64):  // 2 preds: ^bb15, ^bb16
    %76 = llvm.mul %74, %53  : i64
    %77 = llvm.getelementptr %57[%76] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %78 = llvm.load %77 {alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %79 = llvm.getelementptr %69[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %80 = llvm.load %79 {alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %81 = llvm.fmul %80, %78  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %82 = llvm.fadd %81, %75  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %82, %73 {alignment = 8 : i64, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %83 = llvm.add %74, %9  : i64
    %84 = llvm.icmp "eq" %83, %65 : i64
    llvm.cond_br %84, ^bb17, ^bb16(%83, %82 : i64, f64) {loop_annotation = #loop_annotation}
  ^bb17:  // 2 preds: ^bb15, ^bb16
    %85 = llvm.add %66, %9  : i64
    %86 = llvm.icmp "eq" %85, %54 : i64
    llvm.cond_br %86, ^bb18, ^bb15(%85 : i64) {loop_annotation = #loop_annotation}
  ^bb18:  // 2 preds: ^bb13, ^bb17
    %87 = llvm.add %55, %9  : i64
    %88 = llvm.icmp "eq" %87, %53 : i64
    llvm.cond_br %88, ^bb19, ^bb13(%87 : i64) {loop_annotation = #loop_annotation}
  ^bb19:  // 2 preds: ^bb11, ^bb18
    %89 = llvm.add %26, %9  : i64
    %90 = llvm.icmp "eq" %89, %25 : i64
    llvm.cond_br %90, ^bb20, ^bb5(%89 : i64) {loop_annotation = #loop_annotation}
  ^bb20:  // 2 preds: ^bb3, ^bb19
    %91 = llvm.getelementptr inbounds %arg4[%4, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %92 = llvm.load %91 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %93 = llvm.load %arg8 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %94 = llvm.getelementptr inbounds %arg8[%4, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %95 = llvm.load %94 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %96 = llvm.mul %95, %93  : i32
    %97 = llvm.mul %92, %10  : i32
    %98 = llvm.icmp "eq" %96, %97 : i32
    llvm.cond_br %98, ^bb26, ^bb21
  ^bb21:  // pred: ^bb20
    %99 = llvm.getelementptr inbounds %arg8[%4, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %100 = llvm.load %99 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %101 = llvm.icmp "eq" %100, %7 : !llvm.ptr
    llvm.cond_br %101, ^bb23, ^bb22
  ^bb22:  // pred: ^bb21
    llvm.call @free(%100) : (!llvm.ptr) -> ()
    llvm.br ^bb23
  ^bb23:  // 2 preds: ^bb21, ^bb22
    %102 = llvm.icmp "sgt" %92, %1 : i32
    llvm.cond_br %102, ^bb24, ^bb25(%7 : !llvm.ptr)
  ^bb24:  // pred: ^bb23
    %103 = llvm.zext %97 : i32 to i64
    %104 = llvm.shl %103, %8  : i64
    %105 = llvm.call @malloc(%104) : (i64) -> !llvm.ptr
    llvm.br ^bb25(%105 : !llvm.ptr)
  ^bb25(%106: !llvm.ptr):  // 2 preds: ^bb23, ^bb24
    llvm.store %106, %99 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb26
  ^bb26:  // 2 preds: ^bb20, ^bb25
    llvm.store %92, %94 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
    llvm.store %10, %arg8 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
    %107 = llvm.icmp "sgt" %92, %1 : i32
    llvm.cond_br %107, ^bb27, ^bb28
  ^bb27:  // pred: ^bb26
    %108 = llvm.getelementptr inbounds %arg8[%4, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %109 = llvm.load %108 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %110 = llvm.zext %97 : i32 to i64
    %111 = llvm.shl %110, %8  : i64
    "llvm.intr.memset"(%109, %2, %111) <{isVolatile = false, tbaa = [#tbaa_tag1]}> : (!llvm.ptr, i8, i64) -> ()
    llvm.br ^bb28
  ^bb28:  // 2 preds: ^bb26, ^bb27
    %112 = llvm.call @malloc(%11) : (i64) -> !llvm.ptr
    llvm.store %12, %112 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
    %113 = llvm.getelementptr inbounds %112[%4, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.store %92, %113 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
    %114 = llvm.shl %92, %6  : i32
    %115 = llvm.sext %114 : i32 to i64
    %116 = llvm.shl %115, %8  : i64
    %117 = llvm.call @malloc(%116) : (i64) -> !llvm.ptr
    %118 = llvm.getelementptr inbounds %112[%4, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.store %117, %118 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    llvm.cond_br %16, ^bb29, ^bb50
  ^bb29:  // pred: ^bb28
    %119 = llvm.getelementptr inbounds %arg4[%4, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %120 = llvm.zext %92 : i32 to i64
    %121 = llvm.getelementptr inbounds %arg5[%4, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %122 = llvm.getelementptr inbounds %arg8[%4, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %123 = llvm.zext %arg0 : i32 to i64
    llvm.br ^bb30(%117, %117, %4, %12 : !llvm.ptr, !llvm.ptr, i64, i32)
  ^bb30(%124: !llvm.ptr, %125: !llvm.ptr, %126: i64, %127: i32):  // 2 preds: ^bb29, ^bb49
    %128 = llvm.getelementptr inbounds %23[%126] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    llvm.intr.experimental.noalias.scope.decl #alias_scope18
    llvm.intr.experimental.noalias.scope.decl #alias_scope19
    llvm.intr.experimental.noalias.scope.decl #alias_scope20
    %129 = llvm.load %128 {alias_scopes = [#alias_scope18], alignment = 8 : i64, noalias_scopes = [#alias_scope19, #alias_scope20], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %130 = llvm.mul %92, %127  : i32
    %131 = llvm.mul %129, %92  : i32
    %132 = llvm.icmp "eq" %130, %131 : i32
    llvm.cond_br %132, ^bb36(%124, %125 : !llvm.ptr, !llvm.ptr), ^bb31
  ^bb31:  // pred: ^bb30
    %133 = llvm.icmp "eq" %125, %7 : !llvm.ptr
    llvm.cond_br %133, ^bb33, ^bb32
  ^bb32:  // pred: ^bb31
    llvm.call @free(%125) {noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20]} : (!llvm.ptr) -> ()
    llvm.br ^bb33
  ^bb33:  // 2 preds: ^bb31, ^bb32
    %134 = llvm.icmp "sgt" %131, %1 : i32
    llvm.cond_br %134, ^bb34, ^bb35(%7 : !llvm.ptr)
  ^bb34:  // pred: ^bb33
    %135 = llvm.zext %131 : i32 to i64
    %136 = llvm.shl %135, %8  : i64
    %137 = llvm.call @malloc(%136) : (i64) -> !llvm.ptr
    llvm.br ^bb35(%137 : !llvm.ptr)
  ^bb35(%138: !llvm.ptr):  // 2 preds: ^bb33, ^bb34
    llvm.store %138, %118 {alias_scopes = [#alias_scope20], alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb36(%138, %138 : !llvm.ptr, !llvm.ptr)
  ^bb36(%139: !llvm.ptr, %140: !llvm.ptr):  // 2 preds: ^bb30, ^bb35
    %141 = llvm.icmp "sgt" %129, %1 : i32
    llvm.cond_br %141, ^bb37, ^bb44(%140 : !llvm.ptr)
  ^bb37:  // pred: ^bb36
    %142 = llvm.getelementptr inbounds %23[%126, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %143 = llvm.getelementptr inbounds %23[%126, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %144 = llvm.zext %129 : i32 to i64
    llvm.br ^bb38(%140, %4 : !llvm.ptr, i64)
  ^bb38(%145: !llvm.ptr, %146: i64):  // 2 preds: ^bb37, ^bb43
    llvm.cond_br %107, ^bb39, ^bb43(%145 : !llvm.ptr)
  ^bb39:  // pred: ^bb38
    %147 = llvm.load %142 {alias_scopes = [#alias_scope18], alignment = 8 : i64, noalias_scopes = [#alias_scope19, #alias_scope20], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %148 = llvm.getelementptr inbounds %147[%146] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %149 = llvm.load %119 {alias_scopes = [#alias_scope19], alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope20], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %150 = llvm.load %arg4 {alias_scopes = [#alias_scope19], alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope20], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %151 = llvm.load %143 {alias_scopes = [#alias_scope18], alignment = 4 : i64, noalias_scopes = [#alias_scope19, #alias_scope20], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %152 = llvm.icmp "sgt" %151, %5 : i32
    %153 = llvm.sext %150 : i32 to i64
    %154 = llvm.getelementptr %139[%146] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %155 = llvm.zext %151 : i32 to i64
    llvm.br ^bb40(%4 : i64)
  ^bb40(%156: i64):  // 2 preds: ^bb39, ^bb42
    %157 = llvm.load %148 {alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %158 = llvm.mul %156, %153  : i64
    %159 = llvm.getelementptr inbounds %149[%158] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %160 = llvm.load %159 {alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %161 = llvm.fmul %160, %157  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %162 = llvm.mul %156, %144  : i64
    %163 = llvm.getelementptr %154[%162] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %161, %163 {alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    llvm.cond_br %152, ^bb41(%9, %161 : i64, f64), ^bb42
  ^bb41(%164: i64, %165: f64):  // 2 preds: ^bb40, ^bb41
    %166 = llvm.mul %164, %144  : i64
    %167 = llvm.getelementptr %148[%166] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %168 = llvm.load %167 {alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %169 = llvm.getelementptr %159[%164] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %170 = llvm.load %169 {alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %171 = llvm.fmul %170, %168  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %172 = llvm.fadd %171, %165  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %172, %163 {alignment = 8 : i64, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %173 = llvm.add %164, %9  : i64
    %174 = llvm.icmp "eq" %173, %155 : i64
    llvm.cond_br %174, ^bb42, ^bb41(%173, %172 : i64, f64) {loop_annotation = #loop_annotation}
  ^bb42:  // 2 preds: ^bb40, ^bb41
    %175 = llvm.add %156, %9  : i64
    %176 = llvm.icmp "eq" %175, %120 : i64
    llvm.cond_br %176, ^bb43(%139 : !llvm.ptr), ^bb40(%175 : i64) {loop_annotation = #loop_annotation}
  ^bb43(%177: !llvm.ptr):  // 2 preds: ^bb38, ^bb42
    %178 = llvm.add %146, %9  : i64
    %179 = llvm.icmp "eq" %178, %144 : i64
    llvm.cond_br %179, ^bb44(%177 : !llvm.ptr), ^bb38(%177, %178 : !llvm.ptr, i64) {loop_annotation = #loop_annotation}
  ^bb44(%180: !llvm.ptr):  // 2 preds: ^bb36, ^bb43
    llvm.cond_br %107, ^bb45, ^bb49(%139, %180 : !llvm.ptr, !llvm.ptr)
  ^bb45:  // pred: ^bb44
    %181 = llvm.load %118 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %182 = llvm.load %121 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %183 = llvm.load %arg5 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %184 = llvm.load %122 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %185 = llvm.sext %129 : i32 to i64
    %186 = llvm.sext %183 : i32 to i64
    %187 = llvm.getelementptr %182[%126] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb46(%4 : i64)
  ^bb46(%188: i64):  // 2 preds: ^bb45, ^bb48
    %189 = llvm.mul %188, %185  : i64
    %190 = llvm.mul %188, %186  : i64
    %191 = llvm.getelementptr %187[%190] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %192 = llvm.mul %188, %8  : i64
    %193 = llvm.getelementptr %181[%189] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %194 = llvm.getelementptr %184[%192] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb47(%4 : i64)
  ^bb47(%195: i64):  // 2 preds: ^bb46, ^bb47
    %196 = llvm.getelementptr %193[%195] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %197 = llvm.load %196 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %198 = llvm.load %191 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %199 = llvm.fmul %198, %197  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %200 = llvm.getelementptr %194[%195] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %201 = llvm.load %200 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %202 = llvm.fadd %201, %199  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %202, %200 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %203 = llvm.add %195, %9  : i64
    %204 = llvm.icmp "eq" %203, %8 : i64
    llvm.cond_br %204, ^bb48, ^bb47(%203 : i64) {loop_annotation = #loop_annotation}
  ^bb48:  // pred: ^bb47
    %205 = llvm.add %188, %9  : i64
    %206 = llvm.icmp "eq" %205, %120 : i64
    llvm.cond_br %206, ^bb49(%181, %181 : !llvm.ptr, !llvm.ptr), ^bb46(%205 : i64) {loop_annotation = #loop_annotation}
  ^bb49(%207: !llvm.ptr, %208: !llvm.ptr):  // 2 preds: ^bb44, ^bb48
    %209 = llvm.add %126, %9  : i64
    %210 = llvm.icmp "eq" %209, %123 : i64
    llvm.cond_br %210, ^bb50, ^bb30(%207, %208, %209, %129 : !llvm.ptr, !llvm.ptr, i64, i32) {loop_annotation = #loop_annotation}
  ^bb50:  // 2 preds: ^bb28, ^bb49
    %211 = llvm.icmp "ne" %arg6, %1 : i32
    %212 = llvm.and %211, %107  : i1
    llvm.cond_br %212, ^bb51, ^bb53
  ^bb51:  // pred: ^bb50
    %213 = llvm.getelementptr inbounds %arg8[%4, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %214 = llvm.load %213 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %215 = llvm.zext %92 : i32 to i64
    llvm.br ^bb52(%4 : i64)
  ^bb52(%216: i64):  // 2 preds: ^bb51, ^bb52
    %217 = llvm.mul %216, %8  : i64
    %218 = llvm.getelementptr inbounds %214[%217] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %219 = llvm.load %218 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %220 = llvm.fneg %219  {fastmathFlags = #llvm.fastmath<fast>} : f64
    llvm.store %220, %218 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %221 = llvm.add %216, %9  : i64
    %222 = llvm.icmp "eq" %221, %215 : i64
    llvm.cond_br %222, ^bb53, ^bb52(%221 : i64) {loop_annotation = #loop_annotation}
  ^bb53:  // 2 preds: ^bb50, ^bb52
    %223 = llvm.icmp "eq" %arg9, %1 : i32
    llvm.cond_br %223, ^bb55, ^bb54
  ^bb54:  // pred: ^bb53
    llvm.call @apply_global_transform(%arg7, %arg8) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb55
  ^bb55:  // 2 preds: ^bb53, ^bb54
    %224 = llvm.load %118 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %225 = llvm.icmp "eq" %224, %7 : !llvm.ptr
    llvm.cond_br %225, ^bb57, ^bb56
  ^bb56:  // pred: ^bb55
    llvm.call @free(%224) : (!llvm.ptr) -> ()
    llvm.br ^bb57
  ^bb57:  // 2 preds: ^bb55, ^bb56
    llvm.call @free(%112) : (!llvm.ptr) -> ()
    llvm.cond_br %16, ^bb59, ^bb58
  ^bb58:  // pred: ^bb57
    llvm.call @free(%15) : (!llvm.ptr) -> ()
    llvm.call @free(%24) : (!llvm.ptr) -> ()
    llvm.br ^bb71
  ^bb59:  // pred: ^bb57
    %226 = llvm.zext %arg0 : i32 to i64
    llvm.br ^bb60(%4 : i64)
  ^bb60(%227: i64):  // 2 preds: ^bb59, ^bb62
    %228 = llvm.getelementptr inbounds %15[%227, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %229 = llvm.load %228 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %230 = llvm.icmp "eq" %229, %7 : !llvm.ptr
    llvm.cond_br %230, ^bb62, ^bb61
  ^bb61:  // pred: ^bb60
    llvm.call @free(%229) : (!llvm.ptr) -> ()
    llvm.br ^bb62
  ^bb62:  // 2 preds: ^bb60, ^bb61
    %231 = llvm.add %227, %9  : i64
    %232 = llvm.icmp "eq" %231, %226 : i64
    llvm.cond_br %232, ^bb63, ^bb60(%231 : i64) {loop_annotation = #loop_annotation}
  ^bb63:  // pred: ^bb62
    llvm.call @free(%15) : (!llvm.ptr) -> ()
    llvm.br ^bb64(%4 : i64)
  ^bb64(%233: i64):  // 2 preds: ^bb63, ^bb66
    %234 = llvm.getelementptr inbounds %24[%233, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %235 = llvm.load %234 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %236 = llvm.icmp "eq" %235, %7 : !llvm.ptr
    llvm.cond_br %236, ^bb66, ^bb65
  ^bb65:  // pred: ^bb64
    llvm.call @free(%235) : (!llvm.ptr) -> ()
    llvm.br ^bb66
  ^bb66:  // 2 preds: ^bb64, ^bb65
    %237 = llvm.add %233, %9  : i64
    %238 = llvm.icmp "eq" %237, %226 : i64
    llvm.cond_br %238, ^bb67, ^bb64(%237 : i64) {loop_annotation = #loop_annotation}
  ^bb67:  // pred: ^bb66
    llvm.call @free(%24) : (!llvm.ptr) -> ()
    llvm.br ^bb68(%4 : i64)
  ^bb68(%239: i64):  // 2 preds: ^bb67, ^bb70
    %240 = llvm.getelementptr inbounds %23[%239, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %241 = llvm.load %240 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %242 = llvm.icmp "eq" %241, %7 : !llvm.ptr
    llvm.cond_br %242, ^bb70, ^bb69
  ^bb69:  // pred: ^bb68
    llvm.call @free(%241) : (!llvm.ptr) -> ()
    llvm.br ^bb70
  ^bb70:  // 2 preds: ^bb68, ^bb69
    %243 = llvm.add %239, %9  : i64
    %244 = llvm.icmp "eq" %243, %226 : i64
    llvm.cond_br %244, ^bb71, ^bb68(%243 : i64) {loop_annotation = #loop_annotation}
  ^bb71:  // 2 preds: ^bb58, ^bb70
    llvm.call @free(%23) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @to_pose_params(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.readnone}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}) attributes {passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(3 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(-3 : i32) : i32
    %6 = llvm.mlir.constant(3 : i64) : i64
    %7 = llvm.mlir.constant(0 : i8) : i8
    %8 = llvm.mlir.constant(false) : i1
    %9 = llvm.mlir.constant(24 : i64) : i64
    %10 = llvm.mlir.constant(6 : i64) : i64
    %11 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %12 = llvm.mlir.constant(1 : i64) : i64
    %13 = llvm.mlir.constant(0 : i32) : i32
    %14 = llvm.mlir.constant(5 : i32) : i32
    %15 = llvm.mlir.constant(6 : i32) : i32
    %16 = llvm.add %arg0, %0  : i32
    %17 = llvm.load %arg3 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %18 = llvm.getelementptr inbounds %arg3[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %19 = llvm.load %18 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
    %20 = llvm.mul %19, %17  : i32
    %21 = llvm.mul %16, %0  : i32
    %22 = llvm.icmp "eq" %20, %21 : i32
    llvm.cond_br %22, ^bb6, ^bb1
  ^bb1:  // pred: ^bb0
    %23 = llvm.getelementptr inbounds %arg3[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %24 = llvm.load %23 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %25 = llvm.icmp "eq" %24, %4 : !llvm.ptr
    llvm.cond_br %25, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    llvm.call @free(%24) : (!llvm.ptr) -> ()
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %26 = llvm.icmp "sgt" %arg0, %5 : i32
    llvm.cond_br %26, ^bb4, ^bb5(%4 : !llvm.ptr)
  ^bb4:  // pred: ^bb3
    %27 = llvm.zext %21 : i32 to i64
    %28 = llvm.shl %27, %6  : i64
    %29 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr
    llvm.br ^bb5(%29 : !llvm.ptr)
  ^bb5(%30: !llvm.ptr):  // 2 preds: ^bb3, ^bb4
    llvm.store %30, %23 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb0, ^bb5
    llvm.store %16, %18 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
    llvm.store %0, %arg3 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
    %31 = llvm.icmp "sgt" %arg0, %5 : i32
    %32 = llvm.getelementptr inbounds %arg3[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %33 = llvm.load %32 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    llvm.cond_br %31, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %34 = llvm.zext %21 : i32 to i64
    %35 = llvm.shl %34, %6  : i64
    "llvm.intr.memset"(%33, %7, %35) <{isVolatile = false, tbaa = [#tbaa_tag1]}> : (!llvm.ptr, i8, i64) -> ()
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    "llvm.intr.memcpy"(%33, %arg1, %9) <{isVolatile = false, tbaa = [#tbaa_tag1]}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %36 = llvm.getelementptr %33[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb9(%1 : i64)
  ^bb9(%37: i64):  // 2 preds: ^bb8, ^bb9
    %38 = llvm.add %37, %6  : i64
    %39 = llvm.getelementptr inbounds %33[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %11, %39 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %40 = llvm.getelementptr inbounds %arg1[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %41 = llvm.load %40 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %42 = llvm.getelementptr %36[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %41, %42 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %43 = llvm.add %37, %12  : i64
    %44 = llvm.icmp "eq" %43, %6 : i64
    llvm.cond_br %44, ^bb10, ^bb9(%43 : i64) {loop_annotation = #loop_annotation}
  ^bb10:  // pred: ^bb9
    %45 = llvm.getelementptr %33[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb11(%13, %14, %15 : i32, i32, i32)
  ^bb11(%46: i32, %47: i32, %48: i32):  // 2 preds: ^bb10, ^bb15
    %49 = llvm.sext %47 : i32 to i64
    %50 = llvm.add %47, %0  : i32
    llvm.br ^bb12(%49, %3, %48 : i64, i32, i32)
  ^bb12(%51: i64, %52: i32, %53: i32):  // 2 preds: ^bb11, ^bb14
    %54 = llvm.sext %53 : i32 to i64
    %55 = llvm.getelementptr inbounds %arg1[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %56 = llvm.load %55 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %57 = llvm.mul %51, %6  : i64
    %58 = llvm.getelementptr inbounds %33[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %56, %58 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %59 = llvm.add %53, %2  : i32
    %60 = llvm.icmp "eq" %52, %3 : i32
    llvm.cond_br %60, ^bb13, ^bb14(%59 : i32)
  ^bb13:  // pred: ^bb12
    %61 = llvm.sext %59 : i32 to i64
    %62 = llvm.getelementptr inbounds %arg1[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %63 = llvm.load %62 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %64 = llvm.getelementptr %45[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %63, %64 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %65 = llvm.add %53, %3  : i32
    llvm.br ^bb14(%65 : i32)
  ^bb14(%66: i32):  // 2 preds: ^bb12, ^bb13
    %67 = llvm.add %51, %12  : i64
    %68 = llvm.add %52, %2  : i32
    %69 = llvm.trunc %67 : i64 to i32
    %70 = llvm.icmp "eq" %50, %69 : i32
    llvm.cond_br %70, ^bb15, ^bb12(%67, %68, %66 : i64, i32, i32) {loop_annotation = #loop_annotation}
  ^bb15:  // pred: ^bb14
    %71 = llvm.trunc %51 : i64 to i32
    %72 = llvm.add %71, %3  : i32
    %73 = llvm.add %46, %2  : i32
    %74 = llvm.icmp "eq" %73, %14 : i32
    llvm.cond_br %74, ^bb16, ^bb11(%73, %72, %66 : i32, i32, i32) {loop_annotation = #loop_annotation}
  ^bb16:  // pred: ^bb15
    llvm.return
  }
  llvm.func @hand_objective(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: i32 {llvm.noundef}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.readnone}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {enzyme.tag = "base_rels", llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.readnone}, %arg9: i32 {llvm.noundef}, %arg10: i32 {llvm.noundef}, %arg11: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg12: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg13: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(16 : i64) : i64
    %2 = llvm.mlir.poison : !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.mlir.constant(3 : i64) : i64
    %8 = llvm.mlir.zero : !llvm.ptr
    %9 = llvm.call @calloc(%0, %1) : (i64, i64) -> !llvm.ptr
    llvm.call @to_pose_params(%arg1, %arg0, %2, %9) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %10 = llvm.call @calloc(%0, %1) : (i64, i64) -> !llvm.ptr
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
    %21 = llvm.mul %20, %18  : i64
    %22 = llvm.getelementptr inbounds %arg11[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %23 = llvm.load %22 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %24 = llvm.mul %17, %23  : i32
    %25 = llvm.mul %20, %7  : i64
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
    %37 = llvm.add %30, %0  : i64
    %38 = llvm.icmp "eq" %37, %7 : i64
    llvm.cond_br %38, ^bb4, ^bb3(%37 : i64) {loop_annotation = #loop_annotation}
  ^bb4:  // pred: ^bb3
    %39 = llvm.add %20, %0  : i64
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
  llvm.func local_unnamed_addr @hand_objective_complicated(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: i32 {llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg7: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg9: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg10: i32 {llvm.noundef}, %arg11: i32 {llvm.noundef}, %arg12: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg13: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg14: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(16 : i64) : i64
    %2 = llvm.mlir.poison : !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.mlir.constant(2 : i64) : i64
    %8 = llvm.mlir.constant(3 : i64) : i64
    %9 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %10 = llvm.call @calloc(%0, %1) : (i64, i64) -> !llvm.ptr
    llvm.call @to_pose_params(%arg2, %arg0, %2, %10) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %11 = llvm.call @calloc(%0, %1) : (i64, i64) -> !llvm.ptr
    llvm.call @get_skinned_vertex_positions(%arg2, %arg5, %arg4, %arg6, %arg7, %arg8, %arg10, %10, %11, %3) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32) -> ()
    %12 = llvm.icmp "sgt" %arg11, %4 : i32
    llvm.cond_br %12, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    %13 = llvm.getelementptr inbounds %11[%5, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %14 = llvm.load %13 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %15 = llvm.load %11 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %16 = llvm.getelementptr inbounds %arg13[%5, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
    %17 = llvm.load %16 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.load %arg13 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %19 = llvm.sext %18 : i32 to i64
    %20 = llvm.zext %arg11 : i32 to i64
    llvm.br ^bb2(%5 : i64)
  ^bb2(%21: i64):  // 2 preds: ^bb1, ^bb4
    %22 = llvm.getelementptr inbounds %arg12[%21] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %23 = llvm.load %22 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %24 = llvm.sext %23 : i32 to i64
    %25 = llvm.getelementptr inbounds %arg9[%24] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Triangle", (array<3 x i32>)>
    %26 = llvm.shl %21, %0  : i64
    %27 = llvm.getelementptr inbounds %arg1[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %28 = llvm.load %25 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %29 = llvm.mul %15, %28  : i32
    %30 = llvm.getelementptr inbounds %27[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %31 = llvm.getelementptr inbounds %25[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %32 = llvm.load %31 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %33 = llvm.mul %32, %15  : i32
    %34 = llvm.getelementptr inbounds %25[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %35 = llvm.load %34 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %36 = llvm.mul %35, %15  : i32
    %37 = llvm.mul %21, %19  : i64
    %38 = llvm.mul %21, %8  : i64
    %39 = llvm.sext %29 : i32 to i64
    %40 = llvm.sext %33 : i32 to i64
    %41 = llvm.sext %36 : i32 to i64
    %42 = llvm.getelementptr %14[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %43 = llvm.getelementptr %14[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %44 = llvm.getelementptr %14[%41] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %45 = llvm.getelementptr %17[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %46 = llvm.getelementptr %arg14[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb3(%5 : i64)
  ^bb3(%47: i64):  // 2 preds: ^bb2, ^bb3
    %48 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %49 = llvm.getelementptr %42[%47] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %50 = llvm.load %49 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %51 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %52 = llvm.getelementptr %43[%47] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %53 = llvm.load %52 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %54 = llvm.fadd %48, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %55 = llvm.fsub %9, %54  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %56 = llvm.getelementptr %44[%47] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %57 = llvm.load %56 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %58 = llvm.getelementptr %45[%47] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %59 = llvm.load %58 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
    %60 = llvm.fmul %50, %48  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %61 = llvm.fmul %53, %51  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %62 = llvm.fmul %57, %55  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %63 = llvm.fadd %61, %60  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %64 = llvm.fadd %63, %62  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %65 = llvm.fsub %59, %64  {fastmathFlags = #llvm.fastmath<fast>} : f64
    %66 = llvm.getelementptr %46[%47] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %65, %66 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
    %67 = llvm.add %47, %0  : i64
    %68 = llvm.icmp "eq" %67, %8 : i64
    llvm.cond_br %68, ^bb4, ^bb3(%67 : i64) {loop_annotation = #loop_annotation}
  ^bb4:  // pred: ^bb3
    %69 = llvm.add %21, %0  : i64
    %70 = llvm.icmp "eq" %69, %20 : i64
    llvm.cond_br %70, ^bb5, ^bb2(%69 : i64) {loop_annotation = #loop_annotation}
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.return
  }
  llvm.func local_unnamed_addr @dhand_objective(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: !llvm.ptr {llvm.noundef}, %arg4: !llvm.ptr {llvm.noundef}, %arg5: !llvm.ptr {llvm.noundef}, %arg6: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg7: !llvm.ptr {llvm.noundef}, %arg8: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg9: !llvm.ptr {llvm.noundef}, %arg10: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg11: !llvm.ptr {llvm.noundef}, %arg12: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg13: !llvm.ptr {llvm.noundef}, %arg14: i32 {llvm.noundef}, %arg15: i32 {llvm.noundef}, %arg16: !llvm.ptr {llvm.noundef}, %arg17: !llvm.ptr {llvm.noundef}, %arg18: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg19: !llvm.ptr {llvm.noundef}, %arg20: !llvm.ptr {llvm.noundef}) attributes {passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]]} {
    %0 = llvm.mlir.addressof @enzyme_dup : !llvm.ptr
    %1 = llvm.mlir.addressof @enzyme_const : !llvm.ptr
    %2 = llvm.mlir.addressof @enzyme_dupnoneed : !llvm.ptr
    %3 = llvm.mlir.addressof @hand_objective : !llvm.ptr
    %4 = llvm.load %0 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %5 = llvm.load %1 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %6 = llvm.load %2 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    llvm.call @__enzyme_autodiff(%3, %4, %arg0, %arg1, %5, %arg2, %5, %arg3, %5, %arg4, %5, %arg5, %5, %arg7, %5, %arg9, %5, %arg11, %5, %arg13, %5, %arg14, %5, %arg15, %5, %arg16, %5, %arg17, %6, %arg19, %arg20) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @__enzyme_autodiff(!llvm.ptr {llvm.noundef}, ...) attributes {passthrough = [["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"}
  llvm.func local_unnamed_addr @calloc(i64 {llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, passthrough = ["nofree", "nounwind", "willreturn", ["allockind", "17"], ["allocsize", "1"], ["alloc-family", "malloc"]], sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func local_unnamed_addr @angle_axis_to_rotation_matrix(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {enzyme.icv = array<i1: false, false>, memory = #llvm.memory_effects<other = write, argMem = readwrite, inaccessibleMem = none>, passthrough = ["nofree", "nosync", "nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
// CHECK-NEXT:    %0 = llvm.mlir.constant(1 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %1 = llvm.mlir.constant(3 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %2 = llvm.mlir.constant(1.000000e-04 : f64) {enzyme.ici = true, enzyme.icv = true} : f64
// CHECK-NEXT:    %3 = llvm.mlir.constant(2 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %4 = llvm.mlir.constant(1.000000e+00 : f64) {enzyme.ici = true, enzyme.icv = true} : f64
// CHECK-NEXT:    %5 = llvm.mlir.constant(0 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %6 = llvm.mlir.constant(2 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %7 = llvm.mlir.constant(1 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %8 = llvm.mlir.constant(0 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %9 = llvm.mlir.constant(0.000000e+00 : f64) {enzyme.ici = true, enzyme.icv = true} : f64
// CHECK-NEXT:    %10 = llvm.load %arg0 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %11 = llvm.fmul %10, %10  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    llvm.br ^bb1(%0, %11 : i64, f64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb1(%12: i64, %13: f64):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:    %14 = llvm.getelementptr inbounds %arg0[%12] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %15 = llvm.load %14 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %16 = llvm.fmul %15, %15  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %17 = llvm.fadd %16, %13  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %18 = llvm.add %12, %0  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %19 = llvm.icmp "eq" %18, %1 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %19, ^bb2, ^bb1(%18, %17 : i64, f64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    %20 = llvm.intr.sqrt(%17)  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
// CHECK-NEXT:    %21 = llvm.fcmp "olt" %20, %2 {enzyme.ici = true, enzyme.icv = true, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    llvm.cond_br %21, ^bb3, ^bb9 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb3:  // pred: ^bb2
// CHECK-NEXT:    %22 = llvm.getelementptr inbounds %arg1[%5, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %23 = llvm.load %22 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %24 = llvm.icmp "sgt" %23, %8 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %24, ^bb4, ^bb10 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    %25 = llvm.load %arg1 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %26 = llvm.icmp "sgt" %25, %8 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %27 = llvm.getelementptr inbounds %arg1[%5, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %28 = llvm.sext %25 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %29 = llvm.zext %23 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %30 = llvm.zext %25 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb5(%5 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb5(%31: i64):  // 2 preds: ^bb4, ^bb8
// CHECK-NEXT:    llvm.cond_br %26, ^bb6, ^bb8 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb6:  // pred: ^bb5
// CHECK-NEXT:    %32 = llvm.mul %31, %28  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %33 = llvm.load %27 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %34 = llvm.getelementptr %33[%32] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb7(%5 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb7(%35: i64):  // 2 preds: ^bb6, ^bb7
// CHECK-NEXT:    %36 = llvm.icmp "eq" %31, %35 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %37 = llvm.select %36, %4, %9 {enzyme.ici = true, enzyme.icv = true} : i1, f64
// CHECK-NEXT:    %38 = llvm.getelementptr %34[%35] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %37, %38 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %39 = llvm.add %35, %0  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %40 = llvm.icmp "eq" %39, %30 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %40, ^bb8, ^bb7(%39 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb8:  // 2 preds: ^bb5, ^bb7
// CHECK-NEXT:    %41 = llvm.add %31, %0  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %42 = llvm.icmp "eq" %41, %29 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %42, ^bb10, ^bb5(%41 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb9:  // pred: ^bb2
// CHECK-NEXT:    %43 = llvm.fdiv %10, %20  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %44 = llvm.getelementptr inbounds %arg0[%0] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %45 = llvm.load %44 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %46 = llvm.fdiv %45, %20  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %47 = llvm.getelementptr inbounds %arg0[%3] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %48 = llvm.load %47 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %49 = llvm.fdiv %48, %20  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %50 = llvm.intr.sin(%20)  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
// CHECK-NEXT:    %51 = llvm.intr.cos(%20)  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
// CHECK-NEXT:    %52 = llvm.fmul %43, %43  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %53 = llvm.fsub %4, %52  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %54 = llvm.fmul %53, %51  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %55 = llvm.fadd %54, %52  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %56 = llvm.getelementptr inbounds %arg1[%5, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %57 = llvm.load %56 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.store %55, %57 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %58 = llvm.fsub %4, %51  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %59 = llvm.fmul %58, %43  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %60 = llvm.fmul %59, %46  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %61 = llvm.fmul %49, %50  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %62 = llvm.fsub %60, %61  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %63 = llvm.load %arg1 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %64 = llvm.sext %63 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %65 = llvm.getelementptr inbounds %57[%64] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %62, %65 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %66 = llvm.fmul %59, %49  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %67 = llvm.fmul %46, %50  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %68 = llvm.fadd %66, %67  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %69 = llvm.shl %63, %7  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %70 = llvm.sext %69 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %71 = llvm.getelementptr inbounds %57[%70] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %68, %71 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %72 = llvm.fadd %60, %61  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %73 = llvm.getelementptr inbounds %57[%0] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %72, %73 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %74 = llvm.fmul %46, %46  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %75 = llvm.fsub %4, %74  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %76 = llvm.fmul %75, %51  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %77 = llvm.fadd %76, %74  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %78 = llvm.add %63, %7  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %79 = llvm.sext %78 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %80 = llvm.getelementptr inbounds %57[%79] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %77, %80 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %81 = llvm.fmul %46, %58  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %82 = llvm.fmul %81, %49  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %83 = llvm.fmul %43, %50  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %84 = llvm.fsub %82, %83  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %85 = llvm.or %69, %7  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %86 = llvm.sext %85 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %87 = llvm.getelementptr inbounds %57[%86] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %84, %87 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %88 = llvm.fsub %66, %67  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %89 = llvm.getelementptr inbounds %57[%3] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %88, %89 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %90 = llvm.fadd %82, %83  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %91 = llvm.add %63, %6  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %92 = llvm.sext %91 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %93 = llvm.getelementptr inbounds %57[%92] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %90, %93 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %94 = llvm.fmul %49, %49  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %95 = llvm.fsub %4, %94  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %96 = llvm.fmul %95, %51  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %97 = llvm.fadd %96, %94  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %98 = llvm.add %69, %6  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %99 = llvm.sext %98 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %100 = llvm.getelementptr inbounds %57[%99] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %97, %100 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb10 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb10:  // 3 preds: ^bb3, ^bb8, ^bb9
// CHECK-NEXT:    llvm.return {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  }

// CHECK-LABEL: llvm.func local_unnamed_addr @apply_global_transform(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {enzyme.icv = array<i1: false, false>, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
// CHECK-NEXT:    %0 = llvm.mlir.constant(16 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %1 = llvm.mlir.constant(3 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %2 = llvm.mlir.constant(0 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %3 = llvm.mlir.constant(1 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %4 = llvm.mlir.constant(72 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %5 = llvm.mlir.constant(2 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %6 = llvm.mlir.constant(3 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %7 = llvm.mlir.constant(1 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %8 = llvm.mlir.constant(0 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %9 = llvm.mlir.zero {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    %10 = llvm.mlir.constant(4294967295 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %11 = llvm.call @malloc(%0) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %1, %11 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
// CHECK-NEXT:    %12 = llvm.getelementptr inbounds %11[%2, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    llvm.store %1, %12 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
// CHECK-NEXT:    %13 = llvm.call @malloc(%4) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %14 = llvm.getelementptr inbounds %11[%2, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    llvm.store %13, %14 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %15 = llvm.getelementptr inbounds %arg0[%2, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %16 = llvm.load %15 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.call @angle_axis_to_rotation_matrix(%16, %11) {enzyme.ici = false, enzyme.icv = true} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    %17 = llvm.load %15 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %18 = llvm.load %arg0 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %19 = llvm.sext %18 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %20 = llvm.getelementptr %17[%19] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb1(%2 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb3
// CHECK-NEXT:    %22 = llvm.getelementptr %20[%21] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %23 = llvm.mul %21, %6  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %24 = llvm.getelementptr %13[%23] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb2(%2 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb2(%25: i64):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:    %26 = llvm.load %22 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %27 = llvm.getelementptr %24[%25] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %28 = llvm.load %27 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %29 = llvm.fmul %28, %26  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    llvm.store %29, %27 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %30 = llvm.add %25, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %31 = llvm.icmp "eq" %30, %6 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %31, ^bb3, ^bb2(%30 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb3:  // pred: ^bb2
// CHECK-NEXT:    %32 = llvm.add %21, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %33 = llvm.icmp "eq" %32, %6 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %33, ^bb4, ^bb1(%32 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    %34 = llvm.getelementptr inbounds %arg1[%2, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %35 = llvm.load %34 {alias_scopes = [#alias_scope], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %36 = llvm.icmp "sgt" %35, %8 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %36, ^bb5, ^bb6(%9 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb5:  // pred: ^bb4
// CHECK-NEXT:    %37 = llvm.mul %35, %1  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %38 = llvm.zext %37 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %39 = llvm.shl %38, %6  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %40 = llvm.call @malloc(%39) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb6(%40 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb6(%41: !llvm.ptr):  // 2 preds: ^bb4, ^bb5
// CHECK-NEXT:    %42 = llvm.icmp "sgt" %35, %8 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %43 = llvm.getelementptr inbounds %arg1[%2, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %44 = llvm.zext %35 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb7(%2 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb7(%45: i64):  // 2 preds: ^bb6, ^bb12
// CHECK-NEXT:    llvm.cond_br %42, ^bb8, ^bb12 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb8:  // pred: ^bb7
// CHECK-NEXT:    %46 = llvm.getelementptr inbounds %13[%45] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %47 = llvm.load %43 {alias_scopes = [#alias_scope], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %48 = llvm.load %arg1 {alias_scopes = [#alias_scope], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %49 = llvm.sext %48 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %50 = llvm.getelementptr %41[%45] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %51 = llvm.load %46 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    llvm.br ^bb9(%2 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb9(%52: i64):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:    %53 = llvm.mul %52, %49  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %54 = llvm.getelementptr inbounds %47[%53] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %55 = llvm.load %54 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %56 = llvm.fmul %55, %51  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %57 = llvm.mul %52, %6  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %58 = llvm.getelementptr %50[%57] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %56, %58 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb10(%7, %56 : i64, f64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb10(%59: i64, %60: f64):  // 2 preds: ^bb9, ^bb10
// CHECK-NEXT:    %61 = llvm.mul %59, %6  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %62 = llvm.getelementptr %46[%61] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %63 = llvm.load %62 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %64 = llvm.getelementptr %54[%59] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %65 = llvm.load %64 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %66 = llvm.fmul %65, %63  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %67 = llvm.fadd %66, %60  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    llvm.store %67, %58 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope1, #alias_scope, #alias_scope2], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %68 = llvm.add %59, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %69 = llvm.icmp "eq" %68, %6 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %69, ^bb11, ^bb10(%68, %67 : i64, f64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb11:  // pred: ^bb10
// CHECK-NEXT:    %70 = llvm.add %52, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %71 = llvm.icmp "eq" %70, %44 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %71, ^bb12, ^bb9(%70 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb12:  // 2 preds: ^bb7, ^bb11
// CHECK-NEXT:    %72 = llvm.add %45, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %73 = llvm.icmp "eq" %72, %6 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %73, ^bb13, ^bb7(%72 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb13:  // pred: ^bb12
// CHECK-NEXT:    %74 = llvm.load %34 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %75 = llvm.icmp "sgt" %74, %8 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %75, ^bb14, ^bb19 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb14:  // pred: ^bb13
// CHECK-NEXT:    %76 = llvm.load %arg1 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %77 = llvm.icmp "sgt" %76, %8 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %78 = llvm.sext %76 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %79 = llvm.zext %74 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %80 = llvm.shl %18, %3  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %81 = llvm.sext %80 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %82 = llvm.zext %76 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %83 = llvm.getelementptr %17[%81] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb15(%2 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb15(%84: i64):  // 2 preds: ^bb14, ^bb18
// CHECK-NEXT:    llvm.cond_br %77, ^bb16, ^bb18 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb16:  // pred: ^bb15
// CHECK-NEXT:    %85 = llvm.mul %84, %6  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %86 = llvm.load %43 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %87 = llvm.mul %84, %78  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %88 = llvm.getelementptr %86[%87] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb17(%2 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb17(%89: i64):  // 2 preds: ^bb16, ^bb17
// CHECK-NEXT:    %90 = llvm.add %89, %85  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %91 = llvm.and %90, %10  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %92 = llvm.getelementptr inbounds %41[%91] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %93 = llvm.load %92 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %94 = llvm.getelementptr %83[%89] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %95 = llvm.load %94 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %96 = llvm.fadd %95, %93  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %97 = llvm.getelementptr %88[%89] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %96, %97 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %98 = llvm.add %89, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %99 = llvm.icmp "eq" %98, %82 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %99, ^bb18, ^bb17(%98 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb18:  // 2 preds: ^bb15, ^bb17
// CHECK-NEXT:    %100 = llvm.add %84, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %101 = llvm.icmp "eq" %100, %79 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %101, ^bb19, ^bb15(%100 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb19:  // 2 preds: ^bb13, ^bb18
// CHECK-NEXT:    %102 = llvm.icmp "eq" %13, %9 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %102, ^bb21, ^bb20 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb20:  // pred: ^bb19
// CHECK-NEXT:    llvm.call @free(%13) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb21 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb21:  // 2 preds: ^bb19, ^bb20
// CHECK-NEXT:    llvm.call @free(%11) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    %103 = llvm.icmp "eq" %41, %9 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %103, ^bb23, ^bb22 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb22:  // pred: ^bb21
// CHECK-NEXT:    llvm.call @free(%41) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb23 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb23:  // 2 preds: ^bb21, ^bb22
// CHECK-NEXT:    llvm.return {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  }

// CHECK-LABEL: llvm.func local_unnamed_addr @relatives_to_absolutes(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {enzyme.icv = array<i1: true, false, true, false>, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
// CHECK-NEXT:    %0 = llvm.mlir.constant(0 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %1 = llvm.mlir.constant(0 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %2 = llvm.mlir.constant(-1 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %3 = llvm.mlir.constant(1 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %4 = llvm.mlir.constant(2 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %5 = llvm.mlir.zero {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    %6 = llvm.mlir.constant(3 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %7 = llvm.mlir.constant(1 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %8 = llvm.icmp "sgt" %arg0, %0 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %8, ^bb1, ^bb23 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %9 = llvm.zext %arg0 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb2(%1 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb2(%10: i64):  // 2 preds: ^bb1, ^bb22
// CHECK-NEXT:    %11 = llvm.getelementptr inbounds %arg2[%10] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:    %12 = llvm.load %11 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
// CHECK-NEXT:    %13 = llvm.icmp "eq" %12, %2 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %13, ^bb3, ^bb8 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb3:  // pred: ^bb2
// CHECK-NEXT:    %14 = llvm.getelementptr inbounds %arg3[%10] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %15 = llvm.getelementptr inbounds %arg1[%10] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %16 = llvm.getelementptr inbounds %arg3[%10, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %17 = llvm.load %16 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %18 = llvm.icmp "eq" %17, %5 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %18, ^bb5, ^bb4 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    llvm.call @free(%17) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb5 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb5:  // 2 preds: ^bb3, ^bb4
// CHECK-NEXT:    %19 = llvm.getelementptr inbounds %arg1[%10, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %20 = llvm.load %19 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %21 = llvm.getelementptr inbounds %arg3[%10, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    llvm.store %20, %21 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
// CHECK-NEXT:    %22 = llvm.load %15 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    llvm.store %22, %14 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
// CHECK-NEXT:    %23 = llvm.mul %22, %20  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %24 = llvm.sext %23 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %25 = llvm.shl %24, %6  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %26 = llvm.call @malloc(%25) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %26, %16 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %27 = llvm.icmp "sgt" %23, %0 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %27, ^bb6, ^bb22 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb6:  // pred: ^bb5
// CHECK-NEXT:    %28 = llvm.getelementptr inbounds %arg1[%10, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %29 = llvm.load %28 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %30 = llvm.zext %23 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb7(%1 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb7(%31: i64):  // 2 preds: ^bb6, ^bb7
// CHECK-NEXT:    %32 = llvm.getelementptr inbounds %29[%31] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %33 = llvm.load %32 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %34 = llvm.getelementptr inbounds %26[%31] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %33, %34 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %35 = llvm.add %31, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %36 = llvm.icmp "eq" %35, %30 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %36, ^bb22, ^bb7(%35 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb8:  // pred: ^bb2
// CHECK-NEXT:    %37 = llvm.sext %12 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %38 = llvm.getelementptr inbounds %arg3[%37] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %39 = llvm.getelementptr inbounds %arg1[%10] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %40 = llvm.getelementptr inbounds %arg3[%10] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope3 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope4 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope5 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    %41 = llvm.load %38 {alias_scopes = [#alias_scope3], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope4, #alias_scope5], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %42 = llvm.getelementptr inbounds %arg1[%10, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %43 = llvm.load %42 {alias_scopes = [#alias_scope4], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope3, #alias_scope5], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %44 = llvm.load %40 {alias_scopes = [#alias_scope5], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %45 = llvm.getelementptr inbounds %arg3[%10, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %46 = llvm.load %45 {alias_scopes = [#alias_scope5], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %47 = llvm.mul %46, %44  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %48 = llvm.mul %43, %41  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %49 = llvm.icmp "eq" %47, %48 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %49, ^bb14, ^bb9 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb9:  // pred: ^bb8
// CHECK-NEXT:    %50 = llvm.getelementptr inbounds %arg3[%10, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %51 = llvm.load %50 {alias_scopes = [#alias_scope5], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %52 = llvm.icmp "eq" %51, %5 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %52, ^bb11, ^bb10 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb10:  // pred: ^bb9
// CHECK-NEXT:    llvm.call @free(%51) {enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5]} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb11 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb11:  // 2 preds: ^bb9, ^bb10
// CHECK-NEXT:    %53 = llvm.icmp "sgt" %48, %0 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %53, ^bb12, ^bb13(%5 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb12:  // pred: ^bb11
// CHECK-NEXT:    %54 = llvm.zext %48 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %55 = llvm.shl %54, %6  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %56 = llvm.call @malloc(%55) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb13(%56 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb13(%57: !llvm.ptr):  // 2 preds: ^bb11, ^bb12
// CHECK-NEXT:    llvm.store %57, %50 {alias_scopes = [#alias_scope5], alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb14 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb14:  // 2 preds: ^bb8, ^bb13
// CHECK-NEXT:    llvm.store %43, %45 {alias_scopes = [#alias_scope5], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
// CHECK-NEXT:    llvm.store %41, %40 {alias_scopes = [#alias_scope5], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
// CHECK-NEXT:    %58 = llvm.icmp "sgt" %41, %0 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %58, ^bb15, ^bb22 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb15:  // pred: ^bb14
// CHECK-NEXT:    %59 = llvm.icmp "sgt" %43, %0 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %60 = llvm.getelementptr inbounds %arg3[%37, 2] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %61 = llvm.getelementptr inbounds %arg1[%10, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %62 = llvm.getelementptr inbounds %arg3[%10, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %63 = llvm.getelementptr inbounds %arg3[%37, 1] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %64 = llvm.zext %41 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %65 = llvm.zext %43 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb16(%1 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb16(%66: i64):  // 2 preds: ^bb15, ^bb21
// CHECK-NEXT:    llvm.cond_br %59, ^bb17, ^bb21 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb17:  // pred: ^bb16
// CHECK-NEXT:    %67 = llvm.load %60 {alias_scopes = [#alias_scope3], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, noalias_scopes = [#alias_scope4, #alias_scope5], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %68 = llvm.getelementptr inbounds %67[%66] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %69 = llvm.load %61 {alias_scopes = [#alias_scope4], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, noalias_scopes = [#alias_scope3, #alias_scope5], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %70 = llvm.load %39 {alias_scopes = [#alias_scope4], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope3, #alias_scope5], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %71 = llvm.load %62 {alias_scopes = [#alias_scope5], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, noalias_scopes = [#alias_scope3, #alias_scope4], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %72 = llvm.load %63 {alias_scopes = [#alias_scope3], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope4, #alias_scope5], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %73 = llvm.icmp "sgt" %72, %3 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %74 = llvm.sext %70 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %75 = llvm.getelementptr %71[%66] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %76 = llvm.zext %72 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb18(%1 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb18(%77: i64):  // 2 preds: ^bb17, ^bb20
// CHECK-NEXT:    %78 = llvm.load %68 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %79 = llvm.mul %77, %74  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %80 = llvm.getelementptr inbounds %69[%79] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %81 = llvm.load %80 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %82 = llvm.fmul %81, %78  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %83 = llvm.mul %77, %64  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %84 = llvm.getelementptr %75[%83] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %82, %84 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %73, ^bb19(%7, %82 : i64, f64), ^bb20 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb19(%85: i64, %86: f64):  // 2 preds: ^bb18, ^bb19
// CHECK-NEXT:    %87 = llvm.mul %85, %64  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %88 = llvm.getelementptr %68[%87] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %89 = llvm.load %88 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %90 = llvm.getelementptr %80[%85] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %91 = llvm.load %90 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %92 = llvm.fmul %91, %89  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %93 = llvm.fadd %92, %86  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    llvm.store %93, %84 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope3, #alias_scope4, #alias_scope5], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %94 = llvm.add %85, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %95 = llvm.icmp "eq" %94, %76 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %95, ^bb20, ^bb19(%94, %93 : i64, f64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb20:  // 2 preds: ^bb18, ^bb19
// CHECK-NEXT:    %96 = llvm.add %77, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %97 = llvm.icmp "eq" %96, %65 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %97, ^bb21, ^bb18(%96 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb21:  // 2 preds: ^bb16, ^bb20
// CHECK-NEXT:    %98 = llvm.add %66, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %99 = llvm.icmp "eq" %98, %64 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %99, ^bb22, ^bb16(%98 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb22:  // 4 preds: ^bb5, ^bb7, ^bb14, ^bb21
// CHECK-NEXT:    %100 = llvm.add %10, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %101 = llvm.icmp "eq" %100, %9 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %101, ^bb23, ^bb2(%100 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb23:  // 2 preds: ^bb0, ^bb22
// CHECK-NEXT:    llvm.return {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func local_unnamed_addr @euler_angles_to_rotation_matrix(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}) attributes {enzyme.icv = array<i1: false, false>, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
// CHECK-NEXT:    %0 = llvm.mlir.constant(2 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %1 = llvm.mlir.constant(1 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %2 = llvm.mlir.constant(72 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %3 = llvm.mlir.constant(0 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %4 = llvm.mlir.constant(3 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %5 = llvm.mlir.constant(1.000000e+00 : f64) {enzyme.ici = true, enzyme.icv = true} : f64
// CHECK-NEXT:    %6 = llvm.mlir.constant(0.000000e+00 : f64) {enzyme.ici = true, enzyme.icv = true} : f64
// CHECK-NEXT:    %7 = llvm.mlir.constant(4 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %8 = llvm.mlir.constant(5 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %9 = llvm.mlir.constant(7 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %10 = llvm.mlir.constant(8 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %11 = llvm.mlir.constant(6 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %12 = llvm.mlir.constant(1 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %13 = llvm.mlir.constant(9 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %14 = llvm.mlir.constant(2 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %15 = llvm.mlir.zero {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    %16 = llvm.mlir.constant(3 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %17 = llvm.load %arg0 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %18 = llvm.getelementptr inbounds %arg0[%0] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %19 = llvm.load %18 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %20 = llvm.getelementptr inbounds %arg0[%1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %21 = llvm.load %20 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %22 = llvm.call @malloc(%2) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %23 = llvm.call @malloc(%2) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %24 = llvm.call @malloc(%2) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb1(%25: i64):  // 2 preds: ^bb0, ^bb3
// CHECK-NEXT:    %26 = llvm.mul %25, %4  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %27 = llvm.getelementptr %22[%26] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb2(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb2(%28: i64):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:    %29 = llvm.icmp "eq" %25, %28 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %30 = llvm.select %29, %5, %6 {enzyme.ici = true, enzyme.icv = true} : i1, f64
// CHECK-NEXT:    %31 = llvm.getelementptr %27[%28] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %30, %31 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %32 = llvm.add %28, %1  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %33 = llvm.icmp "eq" %32, %4 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %33, ^bb3, ^bb2(%32 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb3:  // pred: ^bb2
// CHECK-NEXT:    %34 = llvm.add %25, %1  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %35 = llvm.icmp "eq" %34, %4 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %35, ^bb4, ^bb1(%34 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    %36 = llvm.intr.cos(%17)  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
// CHECK-NEXT:    %37 = llvm.getelementptr inbounds %22[%7] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %36, %37 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %38 = llvm.intr.sin(%17)  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
// CHECK-NEXT:    %39 = llvm.getelementptr inbounds %22[%8] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %38, %39 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %40 = llvm.fneg %38  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %41 = llvm.getelementptr inbounds %22[%9] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %40, %41 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %42 = llvm.getelementptr inbounds %22[%10] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %36, %42 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb5(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb5(%43: i64):  // 2 preds: ^bb4, ^bb7
// CHECK-NEXT:    %44 = llvm.mul %43, %4  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %45 = llvm.getelementptr %23[%44] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb6(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb6(%46: i64):  // 2 preds: ^bb5, ^bb6
// CHECK-NEXT:    %47 = llvm.icmp "eq" %43, %46 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %48 = llvm.select %47, %5, %6 {enzyme.ici = true, enzyme.icv = true} : i1, f64
// CHECK-NEXT:    %49 = llvm.getelementptr %45[%46] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %48, %49 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %50 = llvm.add %46, %1  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %51 = llvm.icmp "eq" %50, %4 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %51, ^bb7, ^bb6(%50 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb7:  // pred: ^bb6
// CHECK-NEXT:    %52 = llvm.add %43, %1  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %53 = llvm.icmp "eq" %52, %4 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %53, ^bb8, ^bb5(%52 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb8:  // pred: ^bb7
// CHECK-NEXT:    %54 = llvm.intr.cos(%19)  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
// CHECK-NEXT:    llvm.store %54, %23 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %55 = llvm.intr.sin(%19)  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
// CHECK-NEXT:    %56 = llvm.getelementptr inbounds %23[%11] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %55, %56 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %57 = llvm.fneg %55  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %58 = llvm.getelementptr inbounds %23[%0] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %57, %58 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %59 = llvm.getelementptr inbounds %23[%10] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %54, %59 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb9(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb9(%60: i64):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:    %61 = llvm.mul %60, %4  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %62 = llvm.getelementptr %24[%61] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb10(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb10(%63: i64):  // 2 preds: ^bb9, ^bb10
// CHECK-NEXT:    %64 = llvm.icmp "eq" %60, %63 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %65 = llvm.select %64, %5, %6 {enzyme.ici = true, enzyme.icv = true} : i1, f64
// CHECK-NEXT:    %66 = llvm.getelementptr %62[%63] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %65, %66 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %67 = llvm.add %63, %1  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %68 = llvm.icmp "eq" %67, %4 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %68, ^bb11, ^bb10(%67 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb11:  // pred: ^bb10
// CHECK-NEXT:    %69 = llvm.add %60, %1  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %70 = llvm.icmp "eq" %69, %4 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %70, ^bb12, ^bb9(%69 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb12:  // pred: ^bb11
// CHECK-NEXT:    %71 = llvm.intr.cos(%21)  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
// CHECK-NEXT:    llvm.store %71, %24 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %72 = llvm.intr.sin(%21)  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
// CHECK-NEXT:    %73 = llvm.getelementptr inbounds %24[%1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %72, %73 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %74 = llvm.fneg %72  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %75 = llvm.getelementptr inbounds %24[%4] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %74, %75 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %76 = llvm.getelementptr inbounds %24[%7] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %71, %76 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %77 = llvm.call @malloc(%2) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb13(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb13(%78: i64):  // 2 preds: ^bb12, ^bb17
// CHECK-NEXT:    %79 = llvm.getelementptr inbounds %24[%78] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %80 = llvm.getelementptr %77[%78] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %81 = llvm.load %79 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    llvm.br ^bb14(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb14(%82: i64):  // 2 preds: ^bb13, ^bb16
// CHECK-NEXT:    %83 = llvm.mul %82, %4  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %84 = llvm.getelementptr inbounds %23[%83] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %85 = llvm.load %84 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %86 = llvm.fmul %85, %81  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %87 = llvm.getelementptr %80[%83] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb15(%1, %86 : i64, f64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb15(%88: i64, %89: f64):  // 2 preds: ^bb14, ^bb15
// CHECK-NEXT:    %90 = llvm.mul %88, %4  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %91 = llvm.getelementptr %79[%90] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %92 = llvm.load %91 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %93 = llvm.getelementptr %84[%88] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %94 = llvm.load %93 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %95 = llvm.fmul %94, %92  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %96 = llvm.fadd %95, %89  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %97 = llvm.add %88, %1  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %98 = llvm.icmp "eq" %97, %4 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %98, ^bb16, ^bb15(%97, %96 : i64, f64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb16:  // pred: ^bb15
// CHECK-NEXT:    llvm.store %96, %87 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope6, #alias_scope7, #alias_scope8], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %99 = llvm.add %82, %1  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %100 = llvm.icmp "eq" %99, %4 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %100, ^bb17, ^bb14(%99 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb17:  // pred: ^bb16
// CHECK-NEXT:    %101 = llvm.add %78, %1  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %102 = llvm.icmp "eq" %101, %4 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %102, ^bb18, ^bb13(%101 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb18:  // pred: ^bb17
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope9 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    %103 = llvm.load %arg1 {alias_scopes = [#alias_scope9], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %104 = llvm.getelementptr inbounds %arg1[%3, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %105 = llvm.load %104 {alias_scopes = [#alias_scope9], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %106 = llvm.mul %105, %103  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %107 = llvm.icmp "eq" %106, %13 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %108 = llvm.getelementptr inbounds %arg1[%3, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %109 = llvm.load %108 {alias_scopes = [#alias_scope9], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %107, ^bb22(%109 : !llvm.ptr), ^bb19 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb19:  // pred: ^bb18
// CHECK-NEXT:    %110 = llvm.icmp "eq" %109, %15 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %110, ^bb21, ^bb20 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb20:  // pred: ^bb19
// CHECK-NEXT:    llvm.call @free(%109) {enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9]} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb21 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb21:  // 2 preds: ^bb19, ^bb20
// CHECK-NEXT:    %111 = llvm.call @malloc(%2) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %111, %108 {alias_scopes = [#alias_scope9], alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb22(%111 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb22(%112: !llvm.ptr):  // 2 preds: ^bb18, ^bb21
// CHECK-NEXT:    llvm.store %16, %104 {alias_scopes = [#alias_scope9], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
// CHECK-NEXT:    llvm.store %16, %arg1 {alias_scopes = [#alias_scope9], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope10, #alias_scope11], tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb23(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb23(%113: i64):  // 2 preds: ^bb22, ^bb27
// CHECK-NEXT:    %114 = llvm.getelementptr inbounds %77[%113] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %115 = llvm.getelementptr %112[%113] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %116 = llvm.load %114 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    llvm.br ^bb24(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb24(%117: i64):  // 2 preds: ^bb23, ^bb26
// CHECK-NEXT:    %118 = llvm.mul %117, %4  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %119 = llvm.getelementptr inbounds %22[%118] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %120 = llvm.load %119 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %121 = llvm.fmul %120, %116  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %122 = llvm.getelementptr %115[%118] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %121, %122 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb25(%1, %121 : i64, f64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb25(%123: i64, %124: f64):  // 2 preds: ^bb24, ^bb25
// CHECK-NEXT:    %125 = llvm.mul %123, %4  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %126 = llvm.getelementptr %114[%125] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %127 = llvm.load %126 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %128 = llvm.getelementptr %119[%123] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %129 = llvm.load %128 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %130 = llvm.fmul %129, %127  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %131 = llvm.fadd %130, %124  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %132 = llvm.add %123, %1  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %133 = llvm.icmp "eq" %132, %4 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %133, ^bb26, ^bb25(%132, %131 : i64, f64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb26:  // pred: ^bb25
// CHECK-NEXT:    llvm.store %131, %122 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope10, #alias_scope11, #alias_scope9], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %134 = llvm.add %117, %1  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %135 = llvm.icmp "eq" %134, %4 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %135, ^bb27, ^bb24(%134 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb27:  // pred: ^bb26
// CHECK-NEXT:    %136 = llvm.add %113, %1  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %137 = llvm.icmp "eq" %136, %4 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %137, ^bb28, ^bb23(%136 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb28:  // pred: ^bb27
// CHECK-NEXT:    llvm.call @free(%22) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.call @free(%23) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.call @free(%24) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.call @free(%77) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  }
// CHECK-LABEL:  llvm.func local_unnamed_addr @get_posed_relatives(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}) attributes {enzyme.icv = array<i1: true, true, false, false>, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
// CHECK-NEXT:    %0 = llvm.mlir.constant(128 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %1 = llvm.mlir.constant(16 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %2 = llvm.mlir.constant(3 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %3 = llvm.mlir.constant(0 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %4 = llvm.mlir.constant(1 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %5 = llvm.mlir.constant(72 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %6 = llvm.mlir.constant(2 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %7 = llvm.mlir.constant(0 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %8 = llvm.mlir.constant(2 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %9 = llvm.mlir.constant(1.000000e+00 : f64) {enzyme.ici = true, enzyme.icv = true} : f64
// CHECK-NEXT:    %10 = llvm.mlir.constant(0.000000e+00 : f64) {enzyme.ici = true, enzyme.icv = true} : f64
// CHECK-NEXT:    %11 = llvm.mlir.constant(1 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %12 = llvm.mlir.constant(4 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %13 = llvm.mlir.constant(3 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %14 = llvm.mlir.constant(5 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %15 = llvm.mlir.constant(false) {enzyme.ici = true, enzyme.icv = true} : i1
// CHECK-NEXT:    %16 = llvm.mlir.zero {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    %17 = llvm.mlir.constant(4 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %18 = llvm.call @malloc(%0) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %19 = llvm.call @malloc(%1) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %2, %19 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
// CHECK-NEXT:    %20 = llvm.getelementptr inbounds %19[%3, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    llvm.store %2, %20 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
// CHECK-NEXT:    %21 = llvm.call @malloc(%5) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %22 = llvm.getelementptr inbounds %19[%3, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    llvm.store %21, %22 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %23 = llvm.icmp "sgt" %arg0, %7 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %23, ^bb1, ^bb25 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %24 = llvm.getelementptr inbounds %arg2[%3, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %25 = llvm.load %24 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %26 = llvm.load %arg2 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %27 = llvm.sext %26 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %28 = llvm.zext %arg0 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb2(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb2(%29: i64):  // 2 preds: ^bb1, ^bb24
// CHECK-NEXT:    llvm.br ^bb3(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb3(%30: i64):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:    %31 = llvm.shl %30, %8  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %32 = llvm.getelementptr %18[%31] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb4(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb4(%33: i64):  // 2 preds: ^bb3, ^bb4
// CHECK-NEXT:    %34 = llvm.icmp "eq" %30, %33 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %35 = llvm.select %34, %9, %10 {enzyme.ici = true, enzyme.icv = true} : i1, f64
// CHECK-NEXT:    %36 = llvm.getelementptr %32[%33] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %35, %36 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %37 = llvm.add %33, %11  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %38 = llvm.icmp "eq" %37, %12 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %38, ^bb5, ^bb4(%37 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb5:  // pred: ^bb4
// CHECK-NEXT:    %39 = llvm.add %30, %11  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %40 = llvm.icmp "eq" %39, %12 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %40, ^bb6, ^bb3(%39 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb6:  // pred: ^bb5
// CHECK-NEXT:    %41 = llvm.add %29, %13  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %42 = llvm.mul %41, %27  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %43 = llvm.getelementptr inbounds %25[%42] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.call @euler_angles_to_rotation_matrix(%43, %19) {enzyme.ici = false, enzyme.icv = true} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    %44 = llvm.load %20 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %45 = llvm.icmp "sgt" %44, %7 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %45, ^bb7, ^bb11 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb7:  // pred: ^bb6
// CHECK-NEXT:    %46 = llvm.load %19 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %47 = llvm.icmp "sgt" %46, %7 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %48 = llvm.sext %46 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %49 = llvm.zext %44 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %50 = llvm.zext %46 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %51 = llvm.shl %50, %13  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.br ^bb8(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb8(%52: i64):  // 2 preds: ^bb7, ^bb10
// CHECK-NEXT:    llvm.cond_br %47, ^bb9, ^bb10 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb9:  // pred: ^bb8
// CHECK-NEXT:    %53 = llvm.shl %52, %14  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %54 = llvm.getelementptr %18[%53] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:    %55 = llvm.load %22 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %56 = llvm.mul %52, %48  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %57 = llvm.getelementptr %55[%56] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    "llvm.intr.memcpy"(%54, %57, %51) <{isVolatile = false, tbaa = [#tbaa_tag1]}> {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    llvm.br ^bb10 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb10:  // 2 preds: ^bb8, ^bb9
// CHECK-NEXT:    %58 = llvm.add %52, %11  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %59 = llvm.icmp "eq" %58, %49 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %59, ^bb11, ^bb8(%58 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb11:  // 2 preds: ^bb6, ^bb10
// CHECK-NEXT:    %60 = llvm.getelementptr inbounds %arg1[%29] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %61 = llvm.getelementptr inbounds %arg3[%29] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope12 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope13 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    %62 = llvm.load %60 {alias_scopes = [#alias_scope12], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope14, #alias_scope13], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %63 = llvm.load %61 {alias_scopes = [#alias_scope13], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %64 = llvm.getelementptr inbounds %arg3[%29, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %65 = llvm.load %64 {alias_scopes = [#alias_scope13], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %66 = llvm.mul %65, %63  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %67 = llvm.shl %62, %6  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %68 = llvm.icmp "eq" %66, %67 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %68, ^bb17, ^bb12 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb12:  // pred: ^bb11
// CHECK-NEXT:    %69 = llvm.getelementptr inbounds %arg3[%29, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %70 = llvm.load %69 {alias_scopes = [#alias_scope13], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %71 = llvm.icmp "eq" %70, %16 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %71, ^bb14, ^bb13 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb13:  // pred: ^bb12
// CHECK-NEXT:    llvm.call @free(%70) {enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13]} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb14 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb14:  // 2 preds: ^bb12, ^bb13
// CHECK-NEXT:    %72 = llvm.icmp "sgt" %62, %7 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %72, ^bb15, ^bb16(%16 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb15:  // pred: ^bb14
// CHECK-NEXT:    %73 = llvm.zext %67 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %74 = llvm.shl %73, %13  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %75 = llvm.call @malloc(%74) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb16(%75 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb16(%76: !llvm.ptr):  // 2 preds: ^bb14, ^bb15
// CHECK-NEXT:    llvm.store %76, %69 {alias_scopes = [#alias_scope13], alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb17 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb17:  // 2 preds: ^bb11, ^bb16
// CHECK-NEXT:    llvm.store %17, %64 {alias_scopes = [#alias_scope13], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
// CHECK-NEXT:    llvm.store %62, %61 {alias_scopes = [#alias_scope13], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
// CHECK-NEXT:    %77 = llvm.icmp "sgt" %62, %7 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %77, ^bb18, ^bb24 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb18:  // pred: ^bb17
// CHECK-NEXT:    %78 = llvm.getelementptr inbounds %arg1[%29, 2] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %79 = llvm.getelementptr inbounds %arg3[%29, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %80 = llvm.getelementptr inbounds %arg1[%29, 1] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %81 = llvm.zext %62 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %82 = llvm.load %78 {alias_scopes = [#alias_scope12], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope14, #alias_scope13], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %83 = llvm.load %79 {alias_scopes = [#alias_scope13], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %84 = llvm.load %80 {alias_scopes = [#alias_scope12], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope14, #alias_scope13], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %85 = llvm.icmp "sgt" %84, %4 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %86 = llvm.zext %84 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb19(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb19(%87: i64):  // 2 preds: ^bb18, ^bb23
// CHECK-NEXT:    %88 = llvm.getelementptr inbounds %82[%87] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %89 = llvm.getelementptr %83[%87] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb20(%3 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb20(%90: i64):  // 2 preds: ^bb19, ^bb22
// CHECK-NEXT:    %91 = llvm.load %88 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %92 = llvm.shl %90, %8  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %93 = llvm.getelementptr inbounds %18[%92] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %94 = llvm.load %93 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %95 = llvm.fmul %94, %91  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %96 = llvm.mul %90, %81  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %97 = llvm.getelementptr %89[%96] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %95, %97 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %85, ^bb21(%11, %95 : i64, f64), ^bb22 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb21(%98: i64, %99: f64):  // 2 preds: ^bb20, ^bb21
// CHECK-NEXT:    %100 = llvm.mul %98, %81  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %101 = llvm.getelementptr %88[%100] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %102 = llvm.load %101 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %103 = llvm.getelementptr %93[%98] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %104 = llvm.load %103 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %105 = llvm.fmul %104, %102  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %106 = llvm.fadd %105, %99  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    llvm.store %106, %97 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %107 = llvm.add %98, %11  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %108 = llvm.icmp "eq" %107, %86 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %108, ^bb22, ^bb21(%107, %106 : i64, f64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb22:  // 2 preds: ^bb20, ^bb21
// CHECK-NEXT:    %109 = llvm.add %90, %11  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %110 = llvm.icmp "eq" %109, %12 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %110, ^bb23, ^bb20(%109 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb23:  // pred: ^bb22
// CHECK-NEXT:    %111 = llvm.add %87, %11  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %112 = llvm.icmp "eq" %111, %81 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %112, ^bb24, ^bb19(%111 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb24:  // 2 preds: ^bb17, ^bb23
// CHECK-NEXT:    %113 = llvm.add %29, %11  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %114 = llvm.icmp "eq" %113, %28 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %114, ^bb25, ^bb2(%113 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb25:  // 2 preds: ^bb0, ^bb24
// CHECK-NEXT:    %115 = llvm.icmp "eq" %18, %16 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %115, ^bb27, ^bb26 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb26:  // pred: ^bb25
// CHECK-NEXT:    llvm.call @free(%18) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb27 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb27:  // 2 preds: ^bb25, ^bb26
// CHECK-NEXT:    %116 = llvm.load %22 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %117 = llvm.icmp "eq" %116, %16 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %117, ^bb29, ^bb28 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb28:  // pred: ^bb27
// CHECK-NEXT:    llvm.call @free(%116) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb29 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb29:  // 2 preds: ^bb27, ^bb28
// CHECK-NEXT:    llvm.call @free(%19) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func local_unnamed_addr @get_skinned_vertex_positions(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: i32 {llvm.noundef}, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}, %arg9: i32 {llvm.noundef}) attributes {enzyme.icv = array<i1: true, true, true, true, true, true, true, false, false, true>, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
// CHECK-NEXT:    %0 = llvm.mlir.constant(4 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %1 = llvm.mlir.constant(0 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %2 = llvm.mlir.constant(0 : i8) {enzyme.ici = true, enzyme.icv = true} : i8
// CHECK-NEXT:    %3 = llvm.mlir.constant(false) {enzyme.ici = true, enzyme.icv = true} : i1
// CHECK-NEXT:    %4 = llvm.mlir.constant(0 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %5 = llvm.mlir.constant(1 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %6 = llvm.mlir.constant(2 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %7 = llvm.mlir.zero {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    %8 = llvm.mlir.constant(3 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %9 = llvm.mlir.constant(1 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %10 = llvm.mlir.constant(3 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %11 = llvm.mlir.constant(16 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %12 = llvm.mlir.constant(4 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %13 = llvm.sext %arg0 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %14 = llvm.shl %13, %0  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %15 = llvm.call @malloc(%14) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %16 = llvm.icmp "sgt" %arg0, %1 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %16, ^bb2, ^bb1 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %17 = llvm.call @malloc(%14) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %18 = llvm.call @malloc(%14) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb3(%18, %17 : !llvm.ptr, !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb2:  // pred: ^bb0
// CHECK-NEXT:    %19 = llvm.zext %arg0 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %20 = llvm.shl %19, %0  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    "llvm.intr.memset"(%15, %2, %20) <{isVolatile = false, tbaa = [#tbaa_tag]}> {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:    %21 = llvm.call @malloc(%14) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    "llvm.intr.memset"(%21, %2, %20) <{isVolatile = false, tbaa = [#tbaa_tag]}> {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:    %22 = llvm.call @malloc(%14) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    "llvm.intr.memset"(%22, %2, %20) <{isVolatile = false, tbaa = [#tbaa_tag]}> {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:    llvm.br ^bb3(%22, %21 : !llvm.ptr, !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb3(%23: !llvm.ptr, %24: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:    llvm.call @get_posed_relatives(%arg0, %arg1, %arg7, %15) {enzyme.ici = false, enzyme.icv = true} : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.call @relatives_to_absolutes(%arg0, %15, %arg2, %24) {enzyme.ici = false, enzyme.icv = true} : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.cond_br %16, ^bb4, ^bb20 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    %25 = llvm.zext %arg0 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb5(%4 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb5(%26: i64):  // 2 preds: ^bb4, ^bb19
// CHECK-NEXT:    %27 = llvm.getelementptr inbounds %24[%26] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %28 = llvm.getelementptr inbounds %arg3[%26] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %29 = llvm.getelementptr inbounds %23[%26] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope15 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope16 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope17 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    %30 = llvm.load %27 {alias_scopes = [#alias_scope15], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope16, #alias_scope17], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %31 = llvm.getelementptr inbounds %arg3[%26, 1] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %32 = llvm.load %31 {alias_scopes = [#alias_scope16], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope17], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %33 = llvm.load %29 {alias_scopes = [#alias_scope17], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %34 = llvm.getelementptr inbounds %23[%26, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %35 = llvm.load %34 {alias_scopes = [#alias_scope17], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %36 = llvm.mul %35, %33  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %37 = llvm.mul %32, %30  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %38 = llvm.icmp "eq" %36, %37 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %38, ^bb11, ^bb6 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb6:  // pred: ^bb5
// CHECK-NEXT:    %39 = llvm.getelementptr inbounds %23[%26, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %40 = llvm.load %39 {alias_scopes = [#alias_scope17], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %41 = llvm.icmp "eq" %40, %7 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %41, ^bb8, ^bb7 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb7:  // pred: ^bb6
// CHECK-NEXT:    llvm.call @free(%40) {enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17]} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb8 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb8:  // 2 preds: ^bb6, ^bb7
// CHECK-NEXT:    %42 = llvm.icmp "sgt" %37, %1 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %42, ^bb9, ^bb10(%7 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb9:  // pred: ^bb8
// CHECK-NEXT:    %43 = llvm.zext %37 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %44 = llvm.shl %43, %8  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %45 = llvm.call @malloc(%44) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb10(%45 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb10(%46: !llvm.ptr):  // 2 preds: ^bb8, ^bb9
// CHECK-NEXT:    llvm.store %46, %39 {alias_scopes = [#alias_scope17], alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb11 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb11:  // 2 preds: ^bb5, ^bb10
// CHECK-NEXT:    llvm.store %32, %34 {alias_scopes = [#alias_scope17], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
// CHECK-NEXT:    llvm.store %30, %29 {alias_scopes = [#alias_scope17], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
// CHECK-NEXT:    %47 = llvm.icmp "sgt" %30, %1 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %47, ^bb12, ^bb19 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb12:  // pred: ^bb11
// CHECK-NEXT:    %48 = llvm.icmp "sgt" %32, %1 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %49 = llvm.getelementptr inbounds %24[%26, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %50 = llvm.getelementptr inbounds %arg3[%26, 2] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %51 = llvm.getelementptr inbounds %23[%26, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %52 = llvm.getelementptr inbounds %24[%26, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %53 = llvm.zext %30 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %54 = llvm.zext %32 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb13(%4 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb13(%55: i64):  // 2 preds: ^bb12, ^bb18
// CHECK-NEXT:    llvm.cond_br %48, ^bb14, ^bb18 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb14:  // pred: ^bb13
// CHECK-NEXT:    %56 = llvm.load %49 {alias_scopes = [#alias_scope15], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, noalias_scopes = [#alias_scope16, #alias_scope17], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %57 = llvm.getelementptr inbounds %56[%55] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %58 = llvm.load %50 {alias_scopes = [#alias_scope16], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope17], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %59 = llvm.load %28 {alias_scopes = [#alias_scope16], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope17], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %60 = llvm.load %51 {alias_scopes = [#alias_scope17], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, noalias_scopes = [#alias_scope15, #alias_scope16], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %61 = llvm.load %52 {alias_scopes = [#alias_scope15], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope16, #alias_scope17], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %62 = llvm.icmp "sgt" %61, %5 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %63 = llvm.sext %59 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %64 = llvm.getelementptr %60[%55] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %65 = llvm.zext %61 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb15(%4 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb15(%66: i64):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:    %67 = llvm.load %57 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %68 = llvm.mul %66, %63  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %69 = llvm.getelementptr inbounds %58[%68] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %70 = llvm.load %69 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %71 = llvm.fmul %70, %67  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %72 = llvm.mul %66, %53  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %73 = llvm.getelementptr %64[%72] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %71, %73 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %62, ^bb16(%9, %71 : i64, f64), ^bb17 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb16(%74: i64, %75: f64):  // 2 preds: ^bb15, ^bb16
// CHECK-NEXT:    %76 = llvm.mul %74, %53  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %77 = llvm.getelementptr %57[%76] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %78 = llvm.load %77 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %79 = llvm.getelementptr %69[%74] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %80 = llvm.load %79 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %81 = llvm.fmul %80, %78  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %82 = llvm.fadd %81, %75  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    llvm.store %82, %73 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope15, #alias_scope16, #alias_scope17], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %83 = llvm.add %74, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %84 = llvm.icmp "eq" %83, %65 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %84, ^bb17, ^bb16(%83, %82 : i64, f64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb17:  // 2 preds: ^bb15, ^bb16
// CHECK-NEXT:    %85 = llvm.add %66, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %86 = llvm.icmp "eq" %85, %54 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %86, ^bb18, ^bb15(%85 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb18:  // 2 preds: ^bb13, ^bb17
// CHECK-NEXT:    %87 = llvm.add %55, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %88 = llvm.icmp "eq" %87, %53 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %88, ^bb19, ^bb13(%87 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb19:  // 2 preds: ^bb11, ^bb18
// CHECK-NEXT:    %89 = llvm.add %26, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %90 = llvm.icmp "eq" %89, %25 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %90, ^bb20, ^bb5(%89 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb20:  // 2 preds: ^bb3, ^bb19
// CHECK-NEXT:    %91 = llvm.getelementptr inbounds %arg4[%4, 1] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %92 = llvm.load %91 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %93 = llvm.load %arg8 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %94 = llvm.getelementptr inbounds %arg8[%4, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %95 = llvm.load %94 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %96 = llvm.mul %95, %93  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %97 = llvm.mul %92, %10  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %98 = llvm.icmp "eq" %96, %97 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %98, ^bb26, ^bb21 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb21:  // pred: ^bb20
// CHECK-NEXT:    %99 = llvm.getelementptr inbounds %arg8[%4, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %100 = llvm.load %99 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %101 = llvm.icmp "eq" %100, %7 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %101, ^bb23, ^bb22 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb22:  // pred: ^bb21
// CHECK-NEXT:    llvm.call @free(%100) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb23 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb23:  // 2 preds: ^bb21, ^bb22
// CHECK-NEXT:    %102 = llvm.icmp "sgt" %92, %1 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %102, ^bb24, ^bb25(%7 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb24:  // pred: ^bb23
// CHECK-NEXT:    %103 = llvm.zext %97 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %104 = llvm.shl %103, %8  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %105 = llvm.call @malloc(%104) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb25(%105 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb25(%106: !llvm.ptr):  // 2 preds: ^bb23, ^bb24
// CHECK-NEXT:    llvm.store %106, %99 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb26 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb26:  // 2 preds: ^bb20, ^bb25
// CHECK-NEXT:    llvm.store %92, %94 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
// CHECK-NEXT:    llvm.store %10, %arg8 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
// CHECK-NEXT:    %107 = llvm.icmp "sgt" %92, %1 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %107, ^bb27, ^bb28 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb27:  // pred: ^bb26
// CHECK-NEXT:    %108 = llvm.getelementptr inbounds %arg8[%4, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %109 = llvm.load %108 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %110 = llvm.zext %97 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %111 = llvm.shl %110, %8  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    "llvm.intr.memset"(%109, %2, %111) <{isVolatile = false, tbaa = [#tbaa_tag1]}> {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:    llvm.br ^bb28 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb28:  // 2 preds: ^bb26, ^bb27
// CHECK-NEXT:    %112 = llvm.call @malloc(%11) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %12, %112 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
// CHECK-NEXT:    %113 = llvm.getelementptr inbounds %112[%4, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    llvm.store %92, %113 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
// CHECK-NEXT:    %114 = llvm.shl %92, %6  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %115 = llvm.sext %114 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %116 = llvm.shl %115, %8  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %117 = llvm.call @malloc(%116) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %118 = llvm.getelementptr inbounds %112[%4, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    llvm.store %117, %118 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %16, ^bb29, ^bb50 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb29:  // pred: ^bb28
// CHECK-NEXT:    %119 = llvm.getelementptr inbounds %arg4[%4, 2] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %120 = llvm.zext %92 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %121 = llvm.getelementptr inbounds %arg5[%4, 2] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %122 = llvm.getelementptr inbounds %arg8[%4, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %123 = llvm.zext %arg0 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb30(%117, %117, %4, %12 : !llvm.ptr, !llvm.ptr, i64, i32) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb30(%124: !llvm.ptr, %125: !llvm.ptr, %126: i64, %127: i32):  // 2 preds: ^bb29, ^bb49
// CHECK-NEXT:    %128 = llvm.getelementptr inbounds %23[%126] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope18 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope19 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope20 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:    %129 = llvm.load %128 {alias_scopes = [#alias_scope18], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope19, #alias_scope20], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %130 = llvm.mul %92, %127  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %131 = llvm.mul %129, %92  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %132 = llvm.icmp "eq" %130, %131 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %132, ^bb36(%124, %125 : !llvm.ptr, !llvm.ptr), ^bb31 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb31:  // pred: ^bb30
// CHECK-NEXT:    %133 = llvm.icmp "eq" %125, %7 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %133, ^bb33, ^bb32 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb32:  // pred: ^bb31
// CHECK-NEXT:    llvm.call @free(%125) {enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20]} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb33 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb33:  // 2 preds: ^bb31, ^bb32
// CHECK-NEXT:    %134 = llvm.icmp "sgt" %131, %1 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %134, ^bb34, ^bb35(%7 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb34:  // pred: ^bb33
// CHECK-NEXT:    %135 = llvm.zext %131 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %136 = llvm.shl %135, %8  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %137 = llvm.call @malloc(%136) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb35(%137 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb35(%138: !llvm.ptr):  // 2 preds: ^bb33, ^bb34
// CHECK-NEXT:    llvm.store %138, %118 {alias_scopes = [#alias_scope20], alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope18, #alias_scope19], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb36(%138, %138 : !llvm.ptr, !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb36(%139: !llvm.ptr, %140: !llvm.ptr):  // 2 preds: ^bb30, ^bb35
// CHECK-NEXT:    %141 = llvm.icmp "sgt" %129, %1 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %141, ^bb37, ^bb44(%140 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb37:  // pred: ^bb36
// CHECK-NEXT:    %142 = llvm.getelementptr inbounds %23[%126, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %143 = llvm.getelementptr inbounds %23[%126, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %144 = llvm.zext %129 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb38(%140, %4 : !llvm.ptr, i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb38(%145: !llvm.ptr, %146: i64):  // 2 preds: ^bb37, ^bb43
// CHECK-NEXT:    llvm.cond_br %107, ^bb39, ^bb43(%145 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb39:  // pred: ^bb38
// CHECK-NEXT:    %147 = llvm.load %142 {alias_scopes = [#alias_scope18], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, noalias_scopes = [#alias_scope19, #alias_scope20], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %148 = llvm.getelementptr inbounds %147[%146] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %149 = llvm.load %119 {alias_scopes = [#alias_scope19], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope18, #alias_scope20], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %150 = llvm.load %arg4 {alias_scopes = [#alias_scope19], alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope18, #alias_scope20], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %151 = llvm.load %143 {alias_scopes = [#alias_scope18], alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope19, #alias_scope20], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %152 = llvm.icmp "sgt" %151, %5 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %153 = llvm.sext %150 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %154 = llvm.getelementptr %139[%146] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %155 = llvm.zext %151 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb40(%4 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb40(%156: i64):  // 2 preds: ^bb39, ^bb42
// CHECK-NEXT:    %157 = llvm.load %148 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %158 = llvm.mul %156, %153  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %159 = llvm.getelementptr inbounds %149[%158] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %160 = llvm.load %159 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %161 = llvm.fmul %160, %157  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %162 = llvm.mul %156, %144  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %163 = llvm.getelementptr %154[%162] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %161, %163 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %152, ^bb41(%9, %161 : i64, f64), ^bb42 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb41(%164: i64, %165: f64):  // 2 preds: ^bb40, ^bb41
// CHECK-NEXT:    %166 = llvm.mul %164, %144  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %167 = llvm.getelementptr %148[%166] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %168 = llvm.load %167 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %169 = llvm.getelementptr %159[%164] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %170 = llvm.load %169 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %171 = llvm.fmul %170, %168  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %172 = llvm.fadd %171, %165  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    llvm.store %172, %163 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, noalias_scopes = [#alias_scope18, #alias_scope19, #alias_scope20], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %173 = llvm.add %164, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %174 = llvm.icmp "eq" %173, %155 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %174, ^bb42, ^bb41(%173, %172 : i64, f64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb42:  // 2 preds: ^bb40, ^bb41
// CHECK-NEXT:    %175 = llvm.add %156, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %176 = llvm.icmp "eq" %175, %120 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %176, ^bb43(%139 : !llvm.ptr), ^bb40(%175 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb43(%177: !llvm.ptr):  // 2 preds: ^bb38, ^bb42
// CHECK-NEXT:    %178 = llvm.add %146, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %179 = llvm.icmp "eq" %178, %144 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %179, ^bb44(%177 : !llvm.ptr), ^bb38(%177, %178 : !llvm.ptr, i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb44(%180: !llvm.ptr):  // 2 preds: ^bb36, ^bb43
// CHECK-NEXT:    llvm.cond_br %107, ^bb45, ^bb49(%139, %180 : !llvm.ptr, !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb45:  // pred: ^bb44
// CHECK-NEXT:    %181 = llvm.load %118 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %182 = llvm.load %121 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %183 = llvm.load %arg5 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %184 = llvm.load %122 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %185 = llvm.sext %129 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %186 = llvm.sext %183 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %187 = llvm.getelementptr %182[%126] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb46(%4 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb46(%188: i64):  // 2 preds: ^bb45, ^bb48
// CHECK-NEXT:    %189 = llvm.mul %188, %185  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %190 = llvm.mul %188, %186  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %191 = llvm.getelementptr %187[%190] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %192 = llvm.mul %188, %8  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %193 = llvm.getelementptr %181[%189] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %194 = llvm.getelementptr %184[%192] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb47(%4 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb47(%195: i64):  // 2 preds: ^bb46, ^bb47
// CHECK-NEXT:    %196 = llvm.getelementptr %193[%195] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %197 = llvm.load %196 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %198 = llvm.load %191 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %199 = llvm.fmul %198, %197  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %200 = llvm.getelementptr %194[%195] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %201 = llvm.load %200 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %202 = llvm.fadd %201, %199  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    llvm.store %202, %200 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %203 = llvm.add %195, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %204 = llvm.icmp "eq" %203, %8 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %204, ^bb48, ^bb47(%203 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb48:  // pred: ^bb47
// CHECK-NEXT:    %205 = llvm.add %188, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %206 = llvm.icmp "eq" %205, %120 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %206, ^bb49(%181, %181 : !llvm.ptr, !llvm.ptr), ^bb46(%205 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb49(%207: !llvm.ptr, %208: !llvm.ptr):  // 2 preds: ^bb44, ^bb48
// CHECK-NEXT:    %209 = llvm.add %126, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %210 = llvm.icmp "eq" %209, %123 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %210, ^bb50, ^bb30(%207, %208, %209, %129 : !llvm.ptr, !llvm.ptr, i64, i32) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb50:  // 2 preds: ^bb28, ^bb49
// CHECK-NEXT:    %211 = llvm.icmp "ne" %arg6, %1 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %212 = llvm.and %211, %107  {enzyme.ici = true, enzyme.icv = true} : i1
// CHECK-NEXT:    llvm.cond_br %212, ^bb51, ^bb53 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb51:  // pred: ^bb50
// CHECK-NEXT:    %213 = llvm.getelementptr inbounds %arg8[%4, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %214 = llvm.load %213 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %215 = llvm.zext %92 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb52(%4 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb52(%216: i64):  // 2 preds: ^bb51, ^bb52
// CHECK-NEXT:    %217 = llvm.mul %216, %8  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %218 = llvm.getelementptr inbounds %214[%217] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %219 = llvm.load %218 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %220 = llvm.fneg %219  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    llvm.store %220, %218 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %221 = llvm.add %216, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %222 = llvm.icmp "eq" %221, %215 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %222, ^bb53, ^bb52(%221 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb53:  // 2 preds: ^bb50, ^bb52
// CHECK-NEXT:    %223 = llvm.icmp "eq" %arg9, %1 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %223, ^bb55, ^bb54 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb54:  // pred: ^bb53
// CHECK-NEXT:    llvm.call @apply_global_transform(%arg7, %arg8) {enzyme.ici = false, enzyme.icv = true} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb55 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb55:  // 2 preds: ^bb53, ^bb54
// CHECK-NEXT:    %224 = llvm.load %118 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %225 = llvm.icmp "eq" %224, %7 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %225, ^bb57, ^bb56 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb56:  // pred: ^bb55
// CHECK-NEXT:    llvm.call @free(%224) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb57 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb57:  // 2 preds: ^bb55, ^bb56
// CHECK-NEXT:    llvm.call @free(%112) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.cond_br %16, ^bb59, ^bb58 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb58:  // pred: ^bb57
// CHECK-NEXT:    llvm.call @free(%15) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.call @free(%24) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb71 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb59:  // pred: ^bb57
// CHECK-NEXT:    %226 = llvm.zext %arg0 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb60(%4 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb60(%227: i64):  // 2 preds: ^bb59, ^bb62
// CHECK-NEXT:    %228 = llvm.getelementptr inbounds %15[%227, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %229 = llvm.load %228 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %230 = llvm.icmp "eq" %229, %7 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %230, ^bb62, ^bb61 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb61:  // pred: ^bb60
// CHECK-NEXT:    llvm.call @free(%229) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb62 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb62:  // 2 preds: ^bb60, ^bb61
// CHECK-NEXT:    %231 = llvm.add %227, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %232 = llvm.icmp "eq" %231, %226 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %232, ^bb63, ^bb60(%231 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb63:  // pred: ^bb62
// CHECK-NEXT:    llvm.call @free(%15) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb64(%4 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb64(%233: i64):  // 2 preds: ^bb63, ^bb66
// CHECK-NEXT:    %234 = llvm.getelementptr inbounds %24[%233, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %235 = llvm.load %234 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %236 = llvm.icmp "eq" %235, %7 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %236, ^bb66, ^bb65 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb65:  // pred: ^bb64
// CHECK-NEXT:    llvm.call @free(%235) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb66 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb66:  // 2 preds: ^bb64, ^bb65
// CHECK-NEXT:    %237 = llvm.add %233, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %238 = llvm.icmp "eq" %237, %226 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %238, ^bb67, ^bb64(%237 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb67:  // pred: ^bb66
// CHECK-NEXT:    llvm.call @free(%24) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb68(%4 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb68(%239: i64):  // 2 preds: ^bb67, ^bb70
// CHECK-NEXT:    %240 = llvm.getelementptr inbounds %23[%239, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %241 = llvm.load %240 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %242 = llvm.icmp "eq" %241, %7 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %242, ^bb70, ^bb69 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb69:  // pred: ^bb68
// CHECK-NEXT:    llvm.call @free(%241) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb70 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb70:  // 2 preds: ^bb68, ^bb69
// CHECK-NEXT:    %243 = llvm.add %239, %9  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %244 = llvm.icmp "eq" %243, %226 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %244, ^bb71, ^bb68(%243 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb71:  // 2 preds: ^bb58, ^bb70
// CHECK-NEXT:    llvm.call @free(%23) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func local_unnamed_addr @to_pose_params(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.readnone}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}) attributes {enzyme.icv = array<i1: true, false, true, false>, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"} {
// CHECK-NEXT:    %0 = llvm.mlir.constant(3 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %1 = llvm.mlir.constant(0 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %2 = llvm.mlir.constant(1 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %3 = llvm.mlir.constant(2 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %4 = llvm.mlir.zero {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    %5 = llvm.mlir.constant(-3 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %6 = llvm.mlir.constant(3 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %7 = llvm.mlir.constant(0 : i8) {enzyme.ici = true, enzyme.icv = true} : i8
// CHECK-NEXT:    %8 = llvm.mlir.constant(false) {enzyme.ici = true, enzyme.icv = true} : i1
// CHECK-NEXT:    %9 = llvm.mlir.constant(24 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %10 = llvm.mlir.constant(6 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %11 = llvm.mlir.constant(1.000000e+00 : f64) {enzyme.ici = true, enzyme.icv = true} : f64
// CHECK-NEXT:    %12 = llvm.mlir.constant(1 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %13 = llvm.mlir.constant(0 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %14 = llvm.mlir.constant(5 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %15 = llvm.mlir.constant(6 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %16 = llvm.add %arg0, %0  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %17 = llvm.load %arg3 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %18 = llvm.getelementptr inbounds %arg3[%1, 1] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %19 = llvm.load %18 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
// CHECK-NEXT:    %20 = llvm.mul %19, %17  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %21 = llvm.mul %16, %0  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %22 = llvm.icmp "eq" %20, %21 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %22, ^bb6, ^bb1 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %23 = llvm.getelementptr inbounds %arg3[%1, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %24 = llvm.load %23 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %25 = llvm.icmp "eq" %24, %4 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %25, ^bb3, ^bb2 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    llvm.call @free(%24) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb3 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:    %26 = llvm.icmp "sgt" %arg0, %5 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %26, ^bb4, ^bb5(%4 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    %27 = llvm.zext %21 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %28 = llvm.shl %27, %6  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %29 = llvm.call @malloc(%28) {enzyme.ici = true, enzyme.icv = false} : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb5(%29 : !llvm.ptr) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb5(%30: !llvm.ptr):  // 2 preds: ^bb3, ^bb4
// CHECK-NEXT:    llvm.store %30, %23 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb6 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb6:  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:    llvm.store %16, %18 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
// CHECK-NEXT:    llvm.store %0, %arg3 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
// CHECK-NEXT:    %31 = llvm.icmp "sgt" %arg0, %5 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %32 = llvm.getelementptr inbounds %arg3[%1, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %33 = llvm.load %32 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %31, ^bb7, ^bb8 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb7:  // pred: ^bb6
// CHECK-NEXT:    %34 = llvm.zext %21 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %35 = llvm.shl %34, %6  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    "llvm.intr.memset"(%33, %7, %35) <{isVolatile = false, tbaa = [#tbaa_tag1]}> {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:    llvm.br ^bb8 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb8:  // 2 preds: ^bb6, ^bb7
// CHECK-NEXT:    "llvm.intr.memcpy"(%33, %arg1, %9) <{isVolatile = false, tbaa = [#tbaa_tag1]}> {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %36 = llvm.getelementptr %33[%10] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb9(%1 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb9(%37: i64):  // 2 preds: ^bb8, ^bb9
// CHECK-NEXT:    %38 = llvm.add %37, %6  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %39 = llvm.getelementptr inbounds %33[%38] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %11, %39 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %40 = llvm.getelementptr inbounds %arg1[%38] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %41 = llvm.load %40 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %42 = llvm.getelementptr %36[%37] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %41, %42 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %43 = llvm.add %37, %12  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %44 = llvm.icmp "eq" %43, %6 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %44, ^bb10, ^bb9(%43 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb10:  // pred: ^bb9
// CHECK-NEXT:    %45 = llvm.getelementptr %33[%12] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb11(%13, %14, %15 : i32, i32, i32) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb11(%46: i32, %47: i32, %48: i32):  // 2 preds: ^bb10, ^bb15
// CHECK-NEXT:    %49 = llvm.sext %47 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %50 = llvm.add %47, %0  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.br ^bb12(%49, %3, %48 : i64, i32, i32) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb12(%51: i64, %52: i32, %53: i32):  // 2 preds: ^bb11, ^bb14
// CHECK-NEXT:    %54 = llvm.sext %53 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %55 = llvm.getelementptr inbounds %arg1[%54] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %56 = llvm.load %55 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %57 = llvm.mul %51, %6  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %58 = llvm.getelementptr inbounds %33[%57] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %56, %58 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %59 = llvm.add %53, %2  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %60 = llvm.icmp "eq" %52, %3 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %60, ^bb13, ^bb14(%59 : i32) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb13:  // pred: ^bb12
// CHECK-NEXT:    %61 = llvm.sext %59 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %62 = llvm.getelementptr inbounds %arg1[%61] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %63 = llvm.load %62 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %64 = llvm.getelementptr %45[%57] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %63, %64 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %65 = llvm.add %53, %3  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.br ^bb14(%65 : i32) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb14(%66: i32):  // 2 preds: ^bb12, ^bb13
// CHECK-NEXT:    %67 = llvm.add %51, %12  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %68 = llvm.add %52, %2  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %69 = llvm.trunc %67 {enzyme.ici = true, enzyme.icv = true} : i64 to i32
// CHECK-NEXT:    %70 = llvm.icmp "eq" %50, %69 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %70, ^bb15, ^bb12(%67, %68, %66 : i64, i32, i32) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb15:  // pred: ^bb14
// CHECK-NEXT:    %71 = llvm.trunc %51 {enzyme.ici = true, enzyme.icv = true} : i64 to i32
// CHECK-NEXT:    %72 = llvm.add %71, %3  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %73 = llvm.add %46, %2  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %74 = llvm.icmp "eq" %73, %14 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %74, ^bb16, ^bb11(%73, %72, %66 : i32, i32, i32) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb16:  // pred: ^bb15
// CHECK-NEXT:    llvm.return {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func @hand_objective(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: i32 {llvm.noundef}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.readnone}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.readnone}, %arg9: i32 {llvm.noundef}, %arg10: i32 {llvm.noundef}, %arg11: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg12: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg13: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {enzyme.icv = array<i1: false, true, true, true, true, true, true, true, true, true, true, true, true, false>, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]]} {
// CHECK-NEXT:    %0 = llvm.mlir.constant(1 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %1 = llvm.mlir.constant(16 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %2 = llvm.mlir.poison {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    %3 = llvm.mlir.constant(1 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %4 = llvm.mlir.constant(0 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %5 = llvm.mlir.constant(0 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %6 = llvm.mlir.constant(2 : i32) {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %7 = llvm.mlir.constant(3 : i64) {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %8 = llvm.mlir.zero {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    %9 = llvm.call @calloc(%0, %1) {enzyme.ici = true, enzyme.icv = false} : (i64, i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.call @to_pose_params(%arg1, %arg0, %2, %9) {enzyme.ici = false, enzyme.icv = true} : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    %10 = llvm.call @calloc(%0, %1) {enzyme.ici = true, enzyme.icv = false} : (i64, i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.call @get_skinned_vertex_positions(%arg1, %arg4, %arg3, %arg5, %arg6, %arg7, %arg9, %9, %10, %3) {enzyme.ici = false, enzyme.icv = true} : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK-NEXT:    %11 = llvm.icmp "sgt" %arg10, %4 {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    llvm.cond_br %11, ^bb1, ^bb5 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %12 = llvm.getelementptr inbounds %arg12[%5, 2] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %13 = llvm.load %12 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %14 = llvm.load %arg12 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %15 = llvm.getelementptr inbounds %10[%5, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %16 = llvm.load %15 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %17 = llvm.load %10 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
// CHECK-NEXT:    %18 = llvm.sext %14 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %19 = llvm.zext %arg10 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    llvm.br ^bb2(%5 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb2(%20: i64):  // 2 preds: ^bb1, ^bb4
// CHECK-NEXT:    %21 = llvm.mul %20, %18  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %22 = llvm.getelementptr inbounds %arg11[%20] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:    %23 = llvm.load %22 {alignment = 4 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
// CHECK-NEXT:    %24 = llvm.mul %17, %23  {enzyme.ici = true, enzyme.icv = true} : i32
// CHECK-NEXT:    %25 = llvm.mul %20, %7  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %26 = llvm.sext %24 {enzyme.ici = true, enzyme.icv = true} : i32 to i64
// CHECK-NEXT:    %27 = llvm.getelementptr %13[%21] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %28 = llvm.getelementptr %16[%26] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %29 = llvm.getelementptr %arg13[%25] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.br ^bb3(%5 : i64) {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb3(%30: i64):  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:    %31 = llvm.getelementptr %27[%30] {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %32 = llvm.load %31 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = true, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %33 = llvm.getelementptr %28[%30] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    %34 = llvm.load %33 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = false, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:    %35 = llvm.fsub %32, %34  {enzyme.ici = false, enzyme.icv = false, fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:    %36 = llvm.getelementptr %29[%30] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:    llvm.store %35, %36 {alignment = 8 : i64, enzyme.ici = false, enzyme.icv = true, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:    %37 = llvm.add %30, %0  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %38 = llvm.icmp "eq" %37, %7 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %38, ^bb4, ^bb3(%37 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    %39 = llvm.add %20, %0  {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    %40 = llvm.icmp "eq" %39, %19 {enzyme.ici = true, enzyme.icv = true} : i64
// CHECK-NEXT:    llvm.cond_br %40, ^bb5, ^bb2(%39 : i64) {enzyme.ici = true, enzyme.icv = true, loop_annotation = #loop_annotation}
// CHECK-NEXT:  ^bb5:  // 2 preds: ^bb0, ^bb4
// CHECK-NEXT:    %41 = llvm.getelementptr inbounds %9[%5, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %42 = llvm.load %41 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %43 = llvm.icmp "eq" %42, %8 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %43, ^bb7, ^bb6 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb6:  // pred: ^bb5
// CHECK-NEXT:    llvm.call @free(%42) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb7 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb7:  // 2 preds: ^bb5, ^bb6
// CHECK-NEXT:    llvm.call @free(%9) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    %44 = llvm.getelementptr inbounds %10[%5, 2] {enzyme.ici = true, enzyme.icv = false} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
// CHECK-NEXT:    %45 = llvm.load %44 {alignment = 8 : i64, enzyme.ici = true, enzyme.icv = false, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %46 = llvm.icmp "eq" %45, %8 {enzyme.ici = true, enzyme.icv = true} : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %46, ^bb9, ^bb8 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb8:  // pred: ^bb7
// CHECK-NEXT:    llvm.call @free(%45) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb9 {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  ^bb9:  // 2 preds: ^bb7, ^bb8
// CHECK-NEXT:    llvm.call @free(%10) {enzyme.ici = true, enzyme.icv = true} : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return {enzyme.ici = true, enzyme.icv = true}
// CHECK-NEXT:  }
