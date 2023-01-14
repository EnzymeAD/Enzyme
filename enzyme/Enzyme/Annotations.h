//===- Annotations.h - Wrappers determining the context in which a LLVM value is
// used
//---===//
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
// This file declares a base helper class CacheUtility that manages the cache
// of values from the forward pass for later use.
//
//===----------------------------------------------------------------------===//

#ifndef ANNOTATIONS_H
#define ANNOTATIONS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Support/Casting.h"

#include "GradientUtils.h"

using namespace llvm;

template <typename T> struct Primal {
private:
  T *value;

public:
  Primal(T *value) : value(value) {}

  Value *getValue(IRBuilder<> &Builder, std::map<Value*, Value*> &map, GradientUtils *gutils, Value *i) {
    unsigned width = gutils->getWidth();
    if (width == 1)
      return value;
      
    return Builder.CreateLoad(value->getType(), map[value]);
  }
  
  Value *getValue() {
    return value;
  }
  
  Type *getType() {
    return value->getType();
  }
};


template <> struct Primal<Constant> {
private:
  Constant *c;

public:
  Primal(Constant *c) : c(c) {}

  Constant *getValue(IRBuilder<> &Builder, std::map<Value*, Value*> &map, GradientUtils *gutils, Value *i) {
    return c;
  }
  
  Value *getValue() {
    return c;
  }
  
  Type *getType() {
    return c->getType();
  }
};

template <> struct Primal<ConstantVector> {
private:
  ConstantVector *cv;

public:
  Primal(ConstantVector *cv) : cv(cv) {}

  ConstantVector *getValue(IRBuilder<> &Builder, std::map<Value*, Value*> &map, GradientUtils *gutils, Value *i) {
    return cv;
  }
  
  Value *getValue() {
    return cv;
  }
  
  Type *getType() {
    return cv->getType();
  }
};

template <> struct Primal<ConstantDataVector> {
private:
  ConstantDataVector *cv;

public:
  Primal(ConstantDataVector *cv) : cv(cv) {}

  ConstantDataVector *getValue(IRBuilder<> &Builder, std::map<Value*, Value*> &map, GradientUtils *gutils, Value *i) {
    return cv;
  }
  
  Value *getValue() {
    return cv;
  }
  
  Type *getType() {
    return cv->getType();
  }
};

template <typename T> struct Gradient {
private:
  T *value;

public:
  Gradient(T *value) : value(value) {}

  T *getValue(IRBuilder<> &Builder, std::map<Value*, Value*> &map, GradientUtils *gutils, Value *i) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();

    if (width == 1)
      return value;

    switch (memoryLayout) {
      case VectorModeMemoryLayout::VectorizeAtRootNode: {
        assert(cast<ArrayType>(value->getType())->getNumElements() == width);
        auto gep = Builder.CreateInBoundsGEP(map[value], {Builder.getInt64(0), i});
        auto aty = cast<ArrayType>(value->getType());
        return Builder.CreateLoad(aty->getElementType(), gep);
      }
      case VectorModeMemoryLayout::VectorizeAtLeafNodes:
        // TODO: compute the correct offset!!!
        return Builder.CreateInBoundsGEP(map[value], {i});
    }
  }
  
  Value *getValue() {
    return value;
  }
  
  Type *getType() {
    return value->getType();
  }
};

template <> struct Gradient<Constant> {
private:
  Constant *value;

public:
  Gradient(Constant *value) : value(value) {}

  Constant *getValue(IRBuilder<> &Builder, std::map<Value*, Value*> &map, GradientUtils *gutils, Value *i) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();

    return value;
  }
  
  Value *getValue() {
    return value;
  }
  
  Type *getType() {
    return value->getType();
  }
};

template <> struct Gradient<ArrayRef<Constant *>> {
private:
  ArrayRef<Constant *> values;

public:
  Gradient(ArrayRef<Constant *> values) : values(values) {}

  std::vector<Constant *> getValue(IRBuilder<> &Builder, std::map<Value*, Value*> &map, GradientUtils *gutils, Value *i) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();


    return values;
  }
  
  Value *getValue() {
    return nullptr;
  }
  
  Type *getType() {
    return nullptr;
  }
};

#endif
