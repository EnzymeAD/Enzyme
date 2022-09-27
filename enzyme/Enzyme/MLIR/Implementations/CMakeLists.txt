add_mlir_library(MLIREnzymeImplementations
  ArithAutoDiffOpInterfaceImpl.cpp
  BuiltinAutoDiffTypeInterfaceImpl.cpp
  SCFAutoDiffOpInterfaceImpl.cpp

  DEPENDS
  MLIRAutoDiffOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIREnzymeAutoDiffInterface
  MLIRIR
  MLIRSCFDialect
)
