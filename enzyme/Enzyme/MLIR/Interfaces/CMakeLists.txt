add_mlir_interface(AutoDiffOpInterface)
add_mlir_interface(NoOffsetViewInterface)

set(LLVM_TARGET_DEFINITIONS AutoDiffTypeInterface.td)
mlir_tablegen(AutoDiffTypeInterface.h.inc -gen-type-interface-decls)
mlir_tablegen(AutoDiffTypeInterface.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRAutoDiffTypeInterfaceIncGen)

add_mlir_library(
  MLIREnzymeAutoDiffInterface
  AutoDiffOpInterface.cpp
  AutoDiffTypeInterface.cpp
  CloneFunction.cpp
  GradientUtils.cpp
  GradientUtilsReverse.cpp
  HMCUtils.cpp
  ImpulseUtils.cpp
  NoOffsetViewInterface.cpp
  TransformUtils.cpp
  EnzymeLogic.cpp
  EnzymeLogicReverse.cpp
  Utils.cpp
  DEPENDS
  MLIRAutoDiffOpInterfaceIncGen
  MLIRAutoDiffTypeInterfaceIncGen
  MLIRNoOffsetViewInterfaceIncGen
  MLIREnzymeEnumsIncGen
  MLIRImpulseOpsIncGen
  MLIRImpulseEnumsIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRViewLikeInterface
  MLIREnzymeAnalysis
  MLIRImpulse)
