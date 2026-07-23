set(LLVM_TARGET_DEFINITIONS AffineDerivatives.td)
enzyme_tablegen(AffineDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(AffineDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS ArithDerivatives.td)
enzyme_tablegen(ArithDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(ArithDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS LLVMDerivatives.td)
enzyme_tablegen(LLVMDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(LLVMDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS LLVMExtDerivatives.td)
enzyme_tablegen(LLVMExtDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(LLVMExtDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS LinalgDerivatives.td)
enzyme_tablegen(LinalgDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(LinalgDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS NVVMDerivatives.td)
enzyme_tablegen(NVVMDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(NVVMDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS SCFDerivatives.td)
enzyme_tablegen(SCFDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(SCFDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS CFDerivatives.td)
enzyme_tablegen(CFDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(CFDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS MemRefDerivatives.td)
enzyme_tablegen(MemRefDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(MemRefDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS ComplexDerivatives.td)
enzyme_tablegen(ComplexDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(ComplexDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS MathDerivatives.td)
enzyme_tablegen(MathDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(MathDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS FuncDerivatives.td)
enzyme_tablegen(FuncDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(FuncDerivativesIncGen)

set(LLVM_TARGET_DEFINITIONS EnzymeDerivatives.td)
enzyme_tablegen(EnzymeDerivatives.inc -gen-mlir-derivatives)
add_public_tablegen_target(EnzymeDerivativesIncGen)

add_mlir_library(MLIREnzymeImplementations
  AffineAutoDiffOpInterfaceImpl.cpp
  ArithAutoDiffOpInterfaceImpl.cpp
  CoreDialectsAutoDiffImplementations.cpp
  CoreDialectsAutoDiffRegistration.cpp
  LLVMAutoDiffOpInterfaceImpl.cpp
  LLVMExtAutoDiffOpInterfaceImpl.cpp
  NVVMAutoDiffOpInterfaceImpl.cpp
  MemRefAutoDiffOpInterfaceImpl.cpp
  ComplexAutoDiffOpInterfaceImpl.cpp
  FuncAutoDiffOpInterfaceImpl.cpp
  LinalgAutoDiffOpInterfaceImpl.cpp
  BuiltinAutoDiffTypeInterfaceImpl.cpp
  SCFAutoDiffOpInterfaceImpl.cpp
  CFAutoDiffOpInterfaceImpl.cpp
  MathAutoDiffOpInterfaceImpl.cpp
  GPUAutoDiffOpInterfaceImpl.cpp
  EnzymeAutoDiffOpInterfaceImpl.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRAutoDiffOpInterfaceIncGen
  AffineDerivativesIncGen
  ArithDerivativesIncGen
  ComplexDerivativesIncGen
  LLVMDerivativesIncGen
  LLVMExtDerivativesIncGen
  LinalgDerivativesIncGen
  FuncDerivativesIncGen
  NVVMDerivativesIncGen
  SCFDerivativesIncGen
  CFDerivativesIncGen
  MemRefDerivativesIncGen
  MathDerivativesIncGen
  EnzymeDerivativesIncGen
  MLIREnzymeEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRComplexDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRGPUDialect
  MLIREnzymeAutoDiffInterface
  MLIRIR
  MLIRSCFDialect
  MLIRLinalgDialect
)

# Flang-dependent Enzyme plumbing for Fortran. Kept in a separate library (and
# behind ENZYME_FIR_TOOL, which resolves Flang) so the core Enzyme-MLIR build
# never gains a Flang dependency. This layer holds only the dialect-agnostic
# plumbing -- the `enzyme-lower-fortran-calls` pass that rewrites the Fortran
# f__enzyme_* hook calls into enzyme.* ops. The Tier-1 hlfir.*/fir.* autodiff
# models are added to this same library by the FIR autodiff layer. Linked into
# fir-enzyme-opt. See HLFIREnzymeCallLowering.cpp.
if (ENZYME_FIR_TOOL)
  add_mlir_library(MLIREnzymeHLFIRImplementations
    HLFIREnzymeCallLowering.cpp
    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRAutoDiffOpInterfaceIncGen
    MLIREnzymeEnumsIncGen

    LINK_LIBS PUBLIC
    MLIREnzymeImplementations
    MLIREnzymeAutoDiffInterface
    HLFIRDialect
    FIRBuilder
    MLIRIR
  )
  target_include_directories(MLIREnzymeHLFIRImplementations PRIVATE ${FLANG_INCLUDE_DIRS})
endif()
