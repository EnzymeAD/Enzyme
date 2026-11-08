# poseidon-mlir-opt: Poseidon's rewriting of floating-point chains, on MLIR.
# It needs the MLIR of the LLVM build Poseidon is built against, found here
# rather than through ENZYME_MLIR, which would also build Enzyme's MLIR tree.
find_package(MLIR REQUIRED CONFIG HINTS "${LLVM_DIR}/../mlir")
message(STATUS "Poseidon: MLIR from ${MLIR_CMAKE_DIR}")
# A cached MLIR_DIR is what the top-level CMakeLists takes as ENZYME_MLIR's
# MLIR on the next configure, adding its definitions and include paths to
# every target. Found again here each time instead.
unset(MLIR_DIR CACHE)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)

include_directories(${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(${MLIR_DEFINITIONS})

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
  MLIRTensorDialect)

add_llvm_executable(poseidon-mlir-opt poseidon-mlir-opt.cpp
                    PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(poseidon-mlir-opt)
target_link_libraries(
  poseidon-mlir-opt
  PRIVATE MLIRPoseidonTransforms MLIROptLib MLIRTransforms)

install(TARGETS poseidon-mlir-opt RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
