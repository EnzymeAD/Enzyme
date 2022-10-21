add_mlir_library(MLIREnzymeImplementations
  ArithAutoDiffOpInterfaceImpl.cpp
  MemRefAutoDiffOpInterfaceImpl.cpp
  BuiltinAutoDiffTypeInterfaceImpl.cpp
  SCFAutoDiffOpInterfaceImpl.cpp

  DEPENDS
  MLIRAutoDiffOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRMemRefDialect
  MLIREnzymeAutoDiffInterface
  MLIRIR
  MLIRSCFDialect
)
