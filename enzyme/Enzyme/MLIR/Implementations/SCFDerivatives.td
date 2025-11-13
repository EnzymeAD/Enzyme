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
