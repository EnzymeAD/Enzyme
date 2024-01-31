include "Common.td"

def : ControlFlowOp<"scf", "ForOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    scf::ForOpAdaptor adaptor(remappedOperands);
    auto repFor = builder.create<scf::ForOp>(
        op->getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
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
    auto repIf = builder.create<scf::IfOp>(
        op->getLoc(), rettys, adaptor.getCondition());
    return repIf;
  }
}]>;

def : ControlFlowOp<"scf", "WhileOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    return builder.create<scf::WhileOp>(original->getLoc(), rettys,
                                        remappedOperands, original->getAttrs());
  }
}]>;

def : ControlFlowOp<"scf", "ExecuteRegionOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    auto repIf = builder.create<scf::ExecuteRegionOp>(
        op->getLoc(), rettys);
    return repIf;
  }
}]>;

def : RegionTerminatorOp<"scf", "YieldOp">;
def : RegionTerminatorOp<"scf", "ConditionOp">;
