add_mlir_public_c_api_library(MLIRCAPIEnzyme
  EnzymeMLIR.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIREnzyme
)
