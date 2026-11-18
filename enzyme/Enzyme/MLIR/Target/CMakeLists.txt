add_mlir_translation_library(
  MLIREnzymeToLLVMIRTranslation
  EnzymeToLLVMIRTranslation.cpp

  DEPENDS
  MLIREnzymeOpsIncGen
  MLIREnzymeEnumsIncGen
  MLIREnzymeAttributesIncGen

  LINK_LIBS PUBLIC
  MLIREnzyme
  MLIRIR
  MLIRLLVMDialect
  MLIRTargetLLVMIRExport)

target_include_directories(
  MLIREnzymeToLLVMIRTranslation
  PRIVATE ${ENZYME_SOURCE_DIR}/Enzyme)
