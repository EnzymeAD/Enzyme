add_mlir_library(MLIREnzymeAnalysis
  ActivityAnalysis.cpp
  ActivityAnnotation.cpp
  AliasAnalysis.cpp
  DataFlowActivityAnalysis.cpp

  DEPENDS
  MLIRAutoDiffTypeInterfaceIncGen
  MLIRAutoDiffOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
