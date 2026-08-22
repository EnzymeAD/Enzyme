add_mlir_public_c_api_library(MLIRCAPIEnzyme
  EnzymeMLIR.cpp

  DEPENDS
  MLIRImpulseEnumsIncGen
  MLIRImpulseAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCAPIIR

  LINK_LIBS PRIVATE
  MLIREnzyme
  MLIRImpulse
  MLIREnzymeTransforms
  MLIREnzymeAnalysis
  MLIREnzymeAutoDiffInterface
  MLIREnzymeImplementations
)
