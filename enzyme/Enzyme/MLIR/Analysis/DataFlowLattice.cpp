//===- DataFlowLattice.h - Implementation of common dataflow lattices -----===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @inproceedings{NEURIPS2020_9332c513,
// author = {Moses, William and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems},
// editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H.
// Lin}, pages = {12472--12485}, publisher = {Curran Associates, Inc.}, title =
// {Instead of Rewriting Foreign Code for Machine Learning, Automatically
// Synthesize Fast Gradients}, url =
// {https://proceedings.neurips.cc/paper/2020/file/9332c513ef44b682e9347822c2e457ac-Paper.pdf},
// volume = {33},
// year = {2020}
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of reusable lattices in dataflow
// analyses.
//
//===----------------------------------------------------------------------===//

#include <Analysis/DataFlowLattice.h>

#include <algorithm>

using namespace mlir;

bool enzyme::sortAttributes(Attribute a, Attribute b) {
  std::string strA, strB;
  llvm::raw_string_ostream sstreamA(strA), sstreamB(strB);
  sstreamA << a;
  sstreamB << b;
  return strA < strB;
}

bool enzyme::sortArraysLexicographic(ArrayAttr a, ArrayAttr b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
                                      sortAttributes);
}
