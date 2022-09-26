add_mlir_interface(AutoDiffOpInterface)

add_mlir_library(MLIREnzymeAutoDiffInterface
  AutoDiffOpInterface.cpp
  GradientUtils.cpp

  DEPENDS
  MLIRAutoDiffOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
