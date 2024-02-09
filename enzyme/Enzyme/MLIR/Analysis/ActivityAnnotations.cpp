#include "ActivityAnnotations.h"
#include "AliasAnalysis.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
/// Starting from callee, compute a reverse (bottom-up) topological sorting of
/// all functions transitively called from callee.
void reverseToposortCallgraph(CallableOpInterface callee,
                              SymbolTableCollection *symbolTable,
                              SmallVectorImpl<CallableOpInterface> &sorted) {
  DenseSet<CallableOpInterface> permanent;
  DenseSet<CallableOpInterface> temporary;
  std::function<void(CallableOpInterface)> visit =
      [&](CallableOpInterface node) {
        if (permanent.contains(node))
          return;
        if (temporary.contains(node))
          assert(false && "unimplemented cycle in call graph");

        temporary.insert(node);
        node.walk([&](CallOpInterface call) {
          auto neighbour =
              cast<CallableOpInterface>(call.resolveCallable(symbolTable));
          visit(neighbour);
        });

        temporary.erase(node);
        permanent.insert(node);
        sorted.push_back(node);
      };

  visit(callee);
}
} // namespace

void enzyme::runActivityAnnotations(FunctionOpInterface callee) {
  SymbolTableCollection symbolTable;
  SmallVector<CallableOpInterface> sorted;
  reverseToposortCallgraph(callee, &symbolTable, sorted);
  raw_ostream &os = llvm::outs();

  for (CallableOpInterface node : sorted) {
    if (!node.getCallableRegion() || node->hasAttr("p2psummary"))
      continue;
    auto funcOp = cast<FunctionOpInterface>(node.getOperation());
    os << "[ata] processing function @" << funcOp.getName() << "\n";
    DataFlowConfig config;
    config.setInterprocedural(false);
    DataFlowSolver solver(config);

    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<enzyme::AliasAnalysis>(callee.getContext(), /*relative=*/true);
    solver.load<enzyme::PointsToPointerAnalysis>();

    if (failed(solver.initializeAndRun(node))) {
      assert(false && "dataflow solver failed");
    }

    // Create the overall summary by joining sets at all return sites.
    enzyme::PointsToSets p2sets(nullptr);
    for (Operation &op : node.getCallableRegion()->getOps()) {
      if (op.hasTrait<OpTrait::ReturnLike>()) {
        (void)p2sets.join(*solver.lookupState<enzyme::PointsToSets>(&op));
      }
    }

    node->setAttr("p2psummary", p2sets.serialize(node.getContext()));
    os << "[ata] p2p summary:\n";
    for (ArrayAttr pair :
         node->getAttrOfType<ArrayAttr>("p2psummary").getAsRange<ArrayAttr>()) {
      os << "     " << pair[0] << " -> " << pair[1] << "\n";
    }
  }
}
