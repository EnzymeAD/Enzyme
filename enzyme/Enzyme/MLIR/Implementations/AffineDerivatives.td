include "Common.td"

def : ControlFlowOp<"affine", "AffineForOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    return createAffineForWithShadows(op, builder, gutils, original,
                                      remappedOperands, rettys);
  }
}]>;

def : ControlFlowOp<"affine", "AffineIfOp", [{
    Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    return createAffineIfWithShadows(op, builder, gutils,
                                     cast<affine::AffineIfOp>(original),
                                     remappedOperands, rettys);
  }
}]>;
