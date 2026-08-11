add_mlir_dialect(ImpulseOps impulse)
add_mlir_doc(Dialect ImpulseDialect Impulse/ -gen-dialect-doc)
add_mlir_doc(ImpulseOps ImpulseOps Impulse/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS ImpulseOps.td)
mlir_tablegen(ImpulseAttributeInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(ImpulseAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=impulse)
mlir_tablegen(ImpulseAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=impulse)
add_public_tablegen_target(MLIRImpulseAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS ImpulseEnums.td)
mlir_tablegen(ImpulseEnums.h.inc -gen-enum-decls)
mlir_tablegen(ImpulseEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRImpulseEnumsIncGen)

add_mlir_dialect_library(MLIRImpulse
  ImpulseDialect.cpp
  ImpulseOps.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  MLIRImpulseOpsIncGen
  MLIRImpulseEnumsIncGen
  MLIRImpulseAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIREnzyme
)
