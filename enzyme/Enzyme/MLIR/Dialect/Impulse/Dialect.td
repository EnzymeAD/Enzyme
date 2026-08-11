#ifndef IMPULSE_DIALECT_TD
#define IMPULSE_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Impulse dialect definition.
//===----------------------------------------------------------------------===//

def Impulse_Dialect : Dialect {
  let name = "impulse";
  let summary = "A probabilistic programming dialect";
  let cppNamespace = "::mlir::impulse";
  let dependentDialects = [
    "mlir::enzyme::EnzymeDialect",
  ];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Impulse operation definition.
//===----------------------------------------------------------------------===//

class Impulse_Op<string mnemonic, list<Trait> traits = []>
    : Op<Impulse_Dialect, mnemonic, traits>;

class Impulse_Type<string name> : TypeDef<Impulse_Dialect, name>;

#endif // IMPULSE_DIALECT_TD
