add_mlir_dialect(EnzymeOps enzyme)
add_mlir_doc(EnzymeDialect -gen-dialect-doc EnzymeDialect Enzyme/)
add_mlir_doc(EnzymeOps -gen-op-doc EnzymeOps Enzyme/)

add_mlir_dialect_library(MLIREnzyme
Dialect.cpp
Ops.cpp

ADDITIONAL_HEADER_DIRS

DEPENDS
MLIREnzymeOpsIncGen

LINK_LIBS PUBLIC
MLIRIR
MLIRMemRefDialect
)
