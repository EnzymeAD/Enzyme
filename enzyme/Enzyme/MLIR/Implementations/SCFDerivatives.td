include "Common.td"

def : ControlFlowOp<"scf", "ForOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    scf::ForOpAdaptor adaptor(remappedOperands);
    auto repFor = scf::ForOp::create(
        builder, op->getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitArgs());
    return repFor;
  }
}]>;

def : ControlFlowOp<"scf", "ParallelOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    // This should work but doesn't because ParallelOpAdaptor returns empty ranges
    // for lower/upper bounds, step, and init-vals.  So we do it manually instead.
    // Perhaps we need to also pass in the properties?

    // auto dict = DictionaryAttr::get(op->getContext(), op->getAttrs());
    // scf::ParallelOpAdaptor adaptor(remappedOperands, dict);
    // auto repParallel = scf::ParallelOp::create(
    //     builder, op->getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
    //     adaptor.getStep(), adaptor.getInitVals());
    
    mlir::scf::ParallelOp par_op = dyn_cast<mlir::scf::ParallelOp>(op);
    auto rob = remappedOperands.begin();
    auto roe = std::next(rob, par_op.getLowerBound().size());
    ValueRange lb = ValueRange(rob, roe);
    rob = roe;
    roe = std::next(roe, par_op.getUpperBound().size());
    ValueRange ub = ValueRange(rob, roe);
    rob = roe;
    roe = std::next(roe, par_op.getStep().size());
    ValueRange st = ValueRange(rob, roe);
    rob = roe;
    roe = std::next(roe, rettys.size());  // Use rettys instead of par_op.getInitVals() to account for shadows
    ValueRange in = ValueRange(rob, roe);
    auto repParallel = scf::ParallelOp::create(
        builder, op->getLoc(), lb, ub, st, in);

    return repParallel;
  }
}]>;

def : ControlFlowOp<"scf", "IfOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    scf::IfOpAdaptor adaptor(remappedOperands);
    auto repIf = scf::IfOp::create(
        builder, op->getLoc(), rettys, adaptor.getCondition());
    return repIf;
  }
}]>;

def : ControlFlowOp<"scf", "WhileOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    return scf::WhileOp::create(builder, original->getLoc(), rettys,
                                remappedOperands, original->getAttrs());
  }
}]>;

def : ControlFlowOp<"scf", "ExecuteRegionOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    auto repIf = scf::ExecuteRegionOp::create(
        builder, op->getLoc(), rettys);
    return repIf;
  }
}]>;

def : RegionTerminatorOp<"scf", "YieldOp">;
def : RegionTerminatorOp<"scf", "ConditionOp">;
def : RegionTerminatorOp<"scf", "ReduceOp">;
def : RegionTerminatorOp<"scf", "ReduceReturnOp">;
