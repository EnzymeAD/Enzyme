//===- DataFlowActivityAnalysis.h - Declaration of Activity Analysis ------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of Activity Analysis -- an AD-specific
// analysis that deduces if a given instruction or value can impact the
// calculation of a derivative. This file formulates activity analysis within
// a dataflow framework.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_MLIR_ANALYSIS_DATAFLOW_ACTIVITYANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_DATAFLOW_ACTIVITYANALYSIS_H

#include "mlir/IR/Block.h"

namespace mlir {
class FunctionOpInterface;

namespace enzyme {

enum class Activity : uint32_t;

void runDataFlowActivityAnalysis(FunctionOpInterface callee,
                                 ArrayRef<enzyme::Activity> argumentActivity,
                                 bool print = false, bool verbose = false,
                                 bool annotate = false,
                                 bool intraprocedural = false);

} // namespace enzyme
} // namespace mlir

#endif
