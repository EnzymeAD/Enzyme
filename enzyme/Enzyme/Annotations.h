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

  Value *getValue(IRBuilder<> &Builder, GradientUtils *gutils) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();

    if (width == 1)
      return value;

    switch (memoryLayout) {
    case VectorModeMemoryLayout::VectorizeAtRootNode:
      return value;
    case VectorModeMemoryLayout::VectorizeAtLeafNodes:
      if (auto vty = dyn_cast<VectorType>(value->getType())) {
#if LLVM_VERSION_MAJOR >= 12
        unsigned vector_width = vty->getElementCount().getKnownMinValue();
#else
        unsigned vector_width = vty->getNumElements();
#endif
        return Builder.CreateShuffleVector(
            value, UndefValue::get(value->getType()),
            GradientUtils::CreateVectorSplatMask(vector_width, width),
            value->getName() + ".vecsplat");
      } else if (auto sty = dyn_cast<StructType>(value->getType())) {
        auto vsty = GradientUtils::getShadowType(
            *Builder.GetInsertBlock()->getModule(), sty, width, memoryLayout);
        Value *vecstruct = UndefValue::get(vsty);
        for (unsigned i = 0; i < sty->getNumElements(); ++i) {
          auto elem = Builder.CreateExtractValue(value, {i});
          auto splat = Builder.CreateVectorSplat(width, elem);
          vecstruct = Builder.CreateInsertValue(vecstruct, splat, {i});
        }
        return vecstruct;
      }
      return Builder.CreateVectorSplat(width, value);
    }
  }

  Value *getValue(IRBuilder<> &Builder, GradientUtils *gutils, unsigned i) {
    return value;
  }
};

template <> struct Primal<Type> {
private:
  Type *type;

public:
  Primal(Type *type) : type(type) {}

  Type *getValue(IRBuilder<> &Builder, GradientUtils *gutils) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();

    if (width == 1)
      return type;

    switch (memoryLayout) {
    case VectorModeMemoryLayout::VectorizeAtRootNode:
      return type;
    case VectorModeMemoryLayout::VectorizeAtLeafNodes:
      return GradientUtils::getShadowType(
          *Builder.GetInsertBlock()->getModule(), type, width, memoryLayout);
    }
  }

  Type *getValue(IRBuilder<> &Builder, GradientUtils *gutils, unsigned i) {
    return type;
  }
};

template <> struct Primal<ArrayType> {
private:
  ArrayType *type;

public:
  Primal(ArrayType *type) : type(type) {}

  ArrayType *getValue(IRBuilder<> &Builder, GradientUtils *gutils) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();

    if (width == 1)
      return type;

    switch (memoryLayout) {
    case VectorModeMemoryLayout::VectorizeAtRootNode:
      return type;
    case VectorModeMemoryLayout::VectorizeAtLeafNodes: {
      Type *ty = GradientUtils::getShadowType(
          *Builder.GetInsertBlock()->getModule(), type->getElementType(), width,
          memoryLayout);
      return ArrayType::get(ty, type->getNumElements());
    }
    }
  }

  ArrayType *getValue(IRBuilder<> &Builder, GradientUtils *gutils, unsigned i) {
    return type;
  }
};

template <> struct Primal<Constant> {
private:
  Constant *c;

public:
  Primal(Constant *c) : c(c) {}

  Constant *getValue(IRBuilder<> &Builder, GradientUtils *gutils) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();

    if (width == 1)
      return c;

    switch (memoryLayout) {
    case VectorModeMemoryLayout::VectorizeAtRootNode:
      return c;
    case VectorModeMemoryLayout::VectorizeAtLeafNodes:
      std::vector<Constant *> cs(width, c);
      return ConstantVector::get(cs);
    }
  }

  Constant *getValue(IRBuilder<> &Builder, GradientUtils *gutils, unsigned i) {
    return c;
  }
};

template <> struct Primal<ConstantVector> {
private:
  ConstantVector *cv;

public:
  Primal(ConstantVector *cv) : cv(cv) {}

  ConstantVector *getValue(IRBuilder<> &Builder, GradientUtils *gutils) {
    return cv;
  }

  ConstantVector *getValue(IRBuilder<> &Builder, GradientUtils *gutils,
                           unsigned i) {
    return cv;
  }
};

template <> struct Primal<ConstantDataVector> {
private:
  ConstantDataVector *cv;

public:
  Primal(ConstantDataVector *cv) : cv(cv) {}

  ConstantDataVector *getValue(IRBuilder<> &Builder, GradientUtils *gutils) {
    return cv;
  }

  ConstantDataVector *getValue(IRBuilder<> &Builder, GradientUtils *gutils,
                               unsigned i) {
    return cv;
  }
};

template <typename T> struct Gradient {
private:
  T *value;

public:
  Gradient(T *value) : value(value) {}

  T *getValue(IRBuilder<> &Builder, GradientUtils *gutils, unsigned i) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();

    if (width == 1 || !value)
      return value;

    if (!value)
      return nullptr;

    switch (memoryLayout) {
    case VectorModeMemoryLayout::VectorizeAtRootNode:
      assert(cast<ArrayType>(value->getType())->getNumElements() == width);
      return GradientUtils::extractMeta(Builder, value, i);
    case VectorModeMemoryLayout::VectorizeAtLeafNodes:
      if (auto vty = dyn_cast<VectorType>(value->getType())) {
#if LLVM_VERSION_MAJOR >= 12
        unsigned vector_width = vty->getElementCount().getKnownMinValue();
#else
        unsigned vector_width = vty->getNumElements();
#endif
        if (vector_width / width > 1) {
          return Builder.CreateShuffleVector(
              value, UndefValue::get(value->getType()),
              GradientUtils::CreateExtractVectorMask(vector_width, width, i),
              value->getName() + ".subvector." + Twine(i));
        } else {
          return Builder.CreateExtractElement(value, i);
        }
      } else if (auto pty = dyn_cast<PointerType>(value->getType())) {
#if LLVM_VERSION_MAJOR >= 15
        return value;
#else
        if (auto vty = dyn_cast<VectorType>(pty->getElementType())) {
#if LLVM_VERSION_MAJOR >= 12
          unsigned vector_width = vty->getElementCount().getKnownMinValue();
          Type *res_type =
              FixedVectorType::get(vty->getElementType(), vector_width / width);
#else
          unsigned vector_width = vty->getNumElements();
          Type *res_type =
              VectorType::get(vty->getElementType(), vector_width / width);
#endif
          if (vector_width / width > 1) {
            Type *gep_type = PointerType::get(res_type, pty->getAddressSpace());
            Value *idx[2] = {Builder.getInt32(0),
                             Builder.getInt32(i * vector_width / width)};
            auto gep = Builder.CreateInBoundsGEP(vty, value, idx);
            return Builder.CreatePointerCast(gep, gep_type);
          }
        }
        Value *idx[2] = {Builder.getInt32(0), Builder.getInt32(i)};
        return Builder.CreateInBoundsGEP(pty->getElementType(), value, idx);
#endif
      }
      return value;
    }

    return value;
  }

  T *getValue(IRBuilder<> &Builder, GradientUtils *gutils) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();

    if (width == 1 || !value)
      return value;

    if (!value)
      return nullptr;

    if (value && memoryLayout == VectorModeMemoryLayout::VectorizeAtRootNode)
      assert(cast<ArrayType>(value->getType())->getNumElements() == width);

    return value;
  }
};

template <> struct Gradient<Constant> {
private:
  Constant *value;

public:
  Gradient(Constant *value) : value(value) {}

  Constant *getValue(IRBuilder<> &Builder, GradientUtils *gutils, unsigned i) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();

    if (width == 1 || !value)
      return value;

    if (!value)
      return nullptr;

    switch (memoryLayout) {
    case VectorModeMemoryLayout::VectorizeAtRootNode:
      return value;
    case VectorModeMemoryLayout::VectorizeAtLeafNodes:
      if (auto vty = dyn_cast<VectorType>(value->getType())) {
#if LLVM_VERSION_MAJOR >= 12
        unsigned vector_width = vty->getElementCount().getKnownMinValue();
#else
        unsigned vector_width = vty->getNumElements();
#endif
        if (vector_width / width > 1) {
#if LLVM_VERSION_MAJOR >= 11
          auto Mask =
              GradientUtils::CreateExtractVectorMask(vector_width, width, i);
#else
          auto MaskArray =
              GradientUtils::CreateExtractVectorMask(vector_width, width, i);
          SmallVector<Constant *, 8> ConstantArray;
          for (auto elem : MaskArray)
            ConstantArray.push_back(Builder.getInt32(elem));
          auto Mask = ConstantVector::get(ConstantArray);
#endif
          Constant *res = ConstantExpr::getShuffleVector(
              value, UndefValue::get(value->getType()), Mask);
          res->setName(value->getName() + ".subvector." + Twine(i));
          return res;
        } else {
          return ConstantExpr::getExtractElement(value, Builder.getInt64(i));
        }
      } else if (auto pty = dyn_cast<PointerType>(value->getType())) {
#if LLVM_VERSION_MAJOR >= 15
        return value;
#else
        if (auto vty = dyn_cast<VectorType>(pty->getElementType())) {
#if LLVM_VERSION_MAJOR >= 12
          unsigned vector_width = vty->getElementCount().getKnownMinValue();
          Type *res_type =
              FixedVectorType::get(vty->getElementType(), vector_width / width);
#else
          unsigned vector_width = vty->getNumElements();
          Type *res_type =
              VectorType::get(vty->getElementType(), vector_width / width);
#endif
          if (vector_width / width > 1) {
            Type *gep_type = PointerType::get(res_type, pty->getAddressSpace());
            Constant *idx[2] = {Builder.getInt32(0),
                                Builder.getInt32(i * vector_width / width)};
            auto gep = ConstantExpr::getInBoundsGetElementPtr(vty, value, idx);
            return ConstantExpr::getPointerCast(gep, gep_type);
          }
        }
        Constant *idx[2] = {Builder.getInt32(0), Builder.getInt32(i)};
        return ConstantExpr::getInBoundsGetElementPtr(pty->getElementType(),
                                                      value, idx);
#endif
      }
      return value;
    }

    return value;
  }

  Constant *getValue(IRBuilder<> &Builder, GradientUtils *gutils) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();

    if (width == 1 || !value)
      return value;

    if (!value)
      return nullptr;

    if (value && memoryLayout == VectorModeMemoryLayout::VectorizeAtRootNode)
      assert(cast<ArrayType>(value->getType())->getNumElements() == width);

    return value;
  }
};

template <> struct Gradient<ArrayRef<Constant *>> {
private:
  ArrayRef<Constant *> values;

public:
  Gradient(ArrayRef<Constant *> values) : values(values) {}

  std::vector<Constant *> getValue(IRBuilder<> &Builder, GradientUtils *gutils,
                                   unsigned i) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();

    if (width == 1)
      return values;

    switch (memoryLayout) {
    case VectorModeMemoryLayout::VectorizeAtRootNode: {
      std::vector<Constant *> vals;
      for (auto &&val : values) {
        if (val)
          vals.push_back(
              cast<Constant>(GradientUtils::extractMeta(Builder, val, i)));
        else
          vals.push_back(nullptr);
      }
      return vals;
    }
    case VectorModeMemoryLayout::VectorizeAtLeafNodes: {
      std::vector<Constant *> vals;
      for (auto &&val : values) {
        vals.push_back(cast_or_null<Constant>(val));
      }
      return vals;
    }
    }

    return values;
  }

  ArrayRef<Constant *> getValue(IRBuilder<> &Builder, GradientUtils *gutils) {
    VectorModeMemoryLayout memoryLayout = gutils->memoryLayout;
    unsigned width = gutils->getWidth();

    if (width == 1)
      return values;

    switch (memoryLayout) {
    case VectorModeMemoryLayout::VectorizeAtLeafNodes:
      break;
    case VectorModeMemoryLayout::VectorizeAtRootNode:
      for (auto &&val : values) {
        assert(cast<ArrayType>(val->getType())->getNumElements() == width);
      }
      break;
    }

    return values;
  }
};

#endif
