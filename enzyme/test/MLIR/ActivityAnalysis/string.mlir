// RUN: %eopt --print-activity-analysis=funcs=_Z2fnv %s 2>&1 | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "any pointer", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "long", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", members = {<#tbaa_type_desc1, 0>}>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_type_desc4 = #llvm.tbaa_type_desc<id = "_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", members = {<#tbaa_type_desc3, 0>, <#tbaa_type_desc2, 8>, <#tbaa_type_desc, 16>}>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc2, offset = 8>
#tbaa_tag3 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc1, offset = 0>
module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>,
    #dlti.dl_entry<i8, dense<8> : vector<2xi64>>,
    #dlti.dl_entry<i16, dense<16> : vector<2xi64>>,
    #dlti.dl_entry<i1, dense<8> : vector<2xi64>>,
    #dlti.dl_entry<f16, dense<16> : vector<2xi64>>,
    #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi64>>,
    #dlti.dl_entry<i32, dense<32> : vector<2xi64>>,
    #dlti.dl_entry<f128, dense<128> : vector<2xi64>>,
    #dlti.dl_entry<f64, dense<64> : vector<2xi64>>,
    #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.mlir.global private unnamed_addr constant @".str"("test string\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.2"("%f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str("Home\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  // CHECK-LABEL: @_Z2fnv:
  // CHECK:         "alloc": Constant
  // CHECK:         "casted": Constant
  // CHECK:         "loaded": Constant
  llvm.func @_Z2fnv() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant("test string\00") : !llvm.array<12 x i8>
    %4 = llvm.mlir.addressof @".str" : !llvm.ptr
    %5 = llvm.mlir.constant(11 : i64) : i64
    %6 = llvm.mlir.constant(false) : i1
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(3 : i64) : i64
    %9 = llvm.mlir.constant(0 : i8) : i8
    %10 = llvm.alloca %0 x !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)> {alignment = 8 : i64, tag = "alloc"} : (i32) -> !llvm.ptr
    %11 = llvm.bitcast %10 : !llvm.ptr to !llvm.ptr
    %12 = llvm.getelementptr inbounds %10[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %13 = llvm.bitcast %10 : !llvm.ptr to !llvm.ptr
    llvm.store %12, %13 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr, !llvm.ptr
    %14 = llvm.bitcast %12 {tag = "casted"} : !llvm.ptr to !llvm.ptr
    "llvm.intr.memcpy"(%14, %4, %5) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %15 = llvm.getelementptr inbounds %10[%1, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %16 = llvm.getelementptr inbounds %10[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    llvm.store %5, %16 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : i64, !llvm.ptr
    %17 = llvm.getelementptr inbounds %10[%1, 2, 1, %8] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    llvm.store %9, %17 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : i8, !llvm.ptr
    %18 = llvm.call @printf(%14) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %19 = llvm.load %15 {alignment = 8 : i64, tbaa = [#tbaa_tag3], tag = "loaded"} : !llvm.ptr -> !llvm.ptr
    %20 = llvm.icmp "eq" %19, %14 : !llvm.ptr
    llvm.cond_br %20, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @_ZdlPv(%19) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return
  }
  llvm.func @printf(!llvm.ptr {llvm.nocapture, llvm.readonly}, ...) -> i32
  llvm.func local_unnamed_addr @_ZdlPv(!llvm.ptr) attributes {passthrough = ["nobuiltin", "nounwind"]}
}
