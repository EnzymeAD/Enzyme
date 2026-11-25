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
  let description = [{
    Probabilistic programs and inference over them. A model is a function
    whose random choices are `impulse.sample` sites. A model takes an RNG
    state as its first argument and returns the advanced state as its first
    result; likewise, an op that runs a model or a sampler takes the RNG state
    as the first of its `inputs`, and the first of its `outputs`, if it has
    them, is the advanced state.

    A site is named within its model by its `symbol`. Its address is the array
    of the symbols of the sites on the call path from the outermost model down
    to the site, ending with its own.

    A trace of an address array is a `tensor<1xNxT>` that holds the flattened
    values of the sites of the array, in the order of the array. A score-tail
    trace holds, after these N values, one more column per sample site of the
    model, in the order the sites appear in the model, with the log density of
    that site.

    The `name` attribute of an op is a label without semantics.
  }];
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
