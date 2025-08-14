//===- Passes.h - Enzyme pass include header  -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_PASSES_H
#define ENZYME_PASSES_H

#include "../../Utils.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Pass/Pass.h"
#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Dialect/Dialect.h"

namespace mlir {
class PatternRewriter;
class RewritePatternSet;
class DominanceInfo;
} // namespace mlir

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithDialect;
} // end namespace arith

namespace complex {
class ComplexDialect;
} // end namespace complex

namespace cf {
class ControlFlowDialect;
} // end namespace cf

namespace scf {
class SCFDialect;
} // end namespace scf

namespace memref {
class MemRefDialect;
} // end namespace memref

namespace func {
class FuncDialect;
} // end namespace func

namespace affine {
class AffineDialect;
} // end namespace affine

namespace tensor {
class TensorDialect;
} // end namespace tensor

namespace LLVM {
class LLVMDialect;
} // end namespace LLVM

namespace tensor {
class TensorDialect;
} // end namespace tensor

namespace enzyme {

#define GEN_PASS_DECL
#include "Passes/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Passes/Passes.h.inc"
}

} // end namespace mlir

#endif // ENZYME_PASSES_H
