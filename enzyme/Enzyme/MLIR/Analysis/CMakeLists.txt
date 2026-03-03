add_mlir_library(MLIREnzymeAnalysis
  ActivityAnalysis.cpp
  ActivityAnnotations.cpp
  DataFlowAliasAnalysis.cpp
  DataFlowActivityAnalysis.cpp
  DataFlowLattice.cpp
  SampleDependenceAnalysis.cpp

  DEPENDS
  MLIRAutoDiffTypeInterfaceIncGen
  MLIRAutoDiffOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRTransformUtils
)
