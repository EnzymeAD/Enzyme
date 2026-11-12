# poseidon-mlir-opt: Poseidon's rewriting of floating-point chains, on MLIR.
# It uses the MLIR that the top-level CMakeLists finds, the same one Enzyme's
# own MLIR code is built against.
if(NOT MLIR_FOUND)
  message(
    FATAL_ERROR "POSEIDON_MLIR needs MLIR: build LLVM with it, or set MLIR_DIR")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Poseidon)
add_public_tablegen_target(MLIRPoseidonPassIncGen)

add_mlir_library(
  MLIRPoseidonTransforms
  HerbieResults.cpp
  PoseidonRewrite.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRPoseidonPassIncGen
  LINK_LIBS
  PUBLIC
  LLVMSupport
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  PoseidonHerbie)

set_source_files_properties(
  PoseidonRewrite.cpp
  PROPERTIES COMPILE_DEFINITIONS HERBIE_BINARY="${POSEIDON_HERBIE_BINARY_PATH}")

add_llvm_executable(poseidon-mlir-opt poseidon-mlir-opt.cpp
                    PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(poseidon-mlir-opt)
target_link_libraries(
  poseidon-mlir-opt
  PRIVATE MLIRPoseidonTransforms MLIROptLib MLIRTransforms)

install(TARGETS poseidon-mlir-opt RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
