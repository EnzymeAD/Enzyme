//===- DataFlowLattice.h - Declaration of common dataflow lattices --------===//
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
// This file contains the declaration of reusable lattices in dataflow analyses.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_ANALYSIS_DATAFLOW_LATTICE_H
#define ENZYME_MLIR_ANALYSIS_DATAFLOW_LATTICE_H

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir {
namespace enzyme {

constexpr llvm::StringLiteral undefinedSetString = "<undefined>";
constexpr llvm::StringLiteral unknownSetString = "<unknown>";

//===----------------------------------------------------------------------===//
// SetLattice
//
// A data structure representing a set of elements. It may be undefined, meaning
// the analysis has no information about it, or unknown, meaning the analysis
// has conservatively assumed it could contain anything.
//===----------------------------------------------------------------------===//

template <typename ValueT> class SetLattice {
public:
  enum class State {
    Undefined, ///< Has not been analyzed yet (lattice bottom).
    Defined,   ///< Has specific elements.
    Unknown    ///< Analyzed and may contain anything (lattice top).
  };

  SetLattice() : state(State::Undefined) {}

  SetLattice(ValueT single) : state(State::Defined) { elements.insert(single); }

  // TODO(zinenko): deprecate this and use a visitor instead.
  DenseSet<ValueT> &getElements() {
    assert(state == State::Defined);
    return elements;
  }

  const DenseSet<ValueT> &getElements() const {
    return const_cast<SetLattice<ValueT> *>(this)->getElements();
  }

  bool isUnknown() const { return state == State::Unknown; }
  bool isUndefined() const { return state == State::Undefined; }

  ChangeResult join(const SetLattice<ValueT> &other) {
    if (isUnknown())
      return ChangeResult::NoChange;
    if (isUndefined() && other.isUndefined())
      return ChangeResult::NoChange;
    if (other.isUnknown()) {
      state = State::Unknown;
      return ChangeResult::Change;
    }

    ChangeResult result = updateStateToDefined();
    return insert(other.elements) | result;
  }

  ChangeResult insert(const DenseSet<ValueT> &newElements) {
    if (isUnknown())
      return ChangeResult::NoChange;

    size_t oldSize = elements.size();
    elements.insert(newElements.begin(), newElements.end());
    ChangeResult result = elements.size() == oldSize ? ChangeResult::NoChange
                                                     : ChangeResult::Change;
    return updateStateToDefined() | result;
  }

  ChangeResult markUnknown() {
    if (isUnknown())
      return ChangeResult::NoChange;

    state = State::Unknown;
    elements.clear();
    return ChangeResult::Change;
  }

  /// Returns true if this set is in the canonical form, i.e. either the state
  /// is `State::Defined` or the explicit list of classes is empty, but not
  /// both.
  bool isCanonical() const {
    return state == State::Defined || elements.empty();
  }

  /// Returns an instance of SetLattice known not to have any elements.
  /// This is different from "undefined" and "unknown". The instance is *not* a
  /// classical singleton.
  static const SetLattice<ValueT> &getEmpty() {
    static const SetLattice<ValueT> empty(State::Defined);
    return empty;
  }

  /// Returns an instance of SetLattice in "undefined" state, i.e. without a set
  /// of elements. This is different from empty set, which indicates that the
  /// set is known not to contain any elements. The instance is *not* a
  /// classical singleton, there are other ways of obtaining it.
  static const SetLattice<ValueT> &getUndefined() { return undefinedSet; }

  /// Returns an instance of SetLattice for the "unknown" class. The instance
  /// is *not* a classical singleton, there are other ways of obtaining an
  /// "unknown" alias set.
  static const SetLattice<ValueT> &getUnknown() { return unknownSet; }

  bool operator==(const SetLattice<ValueT> &other) const {
    assert(isCanonical() && other.isCanonical());
    return state == other.state && llvm::equal(elements, other.elements);
  }

  LLVM_DUMP_METHOD void print(llvm::raw_ostream &os) const {
    if (isUnknown()) {
      os << unknownSetString;
    } else if (isUndefined()) {
      os << undefinedSetString;
    } else {
      llvm::interleaveComma(elements, os << "{");
      os << "}";
    }
  }

  ChangeResult
  foreachElement(function_ref<ChangeResult(ValueT, State)> callback) const {
    if (state != State::Defined)
      return callback(nullptr, state);

    ChangeResult result = ChangeResult::NoChange;
    for (ValueT element : elements)
      result |= callback(element, state);
    return result;
  }

private:
  explicit SetLattice(State state) : state(state) {}

  ChangeResult updateStateToDefined() {
    assert(state != State::Unknown && "cannot go back from unknown state");
    ChangeResult result = state == State::Undefined ? ChangeResult::Change
                                                    : ChangeResult::NoChange;
    state = State::Defined;
    return result;
  }

  const static SetLattice<ValueT> unknownSet;
  const static SetLattice<ValueT> undefinedSet;

  DenseSet<ValueT> elements;
  State state;
};

template <typename ValueT>
const SetLattice<ValueT> SetLattice<ValueT>::unknownSet =
    SetLattice<ValueT>(SetLattice<ValueT>::State::Unknown);

template <typename ValueT>
const SetLattice<ValueT> SetLattice<ValueT>::undefinedSet =
    SetLattice<ValueT>(SetLattice<ValueT>::State::Undefined);

/// Used when serializing to ensure a consistent order.
bool sortAttributes(Attribute a, Attribute b);
bool sortArraysLexicographic(ArrayAttr a, ArrayAttr b);

//===----------------------------------------------------------------------===//
// SparseSetLattice
//
// An abstract lattice for sparse analyses that wraps a set lattice.
//===----------------------------------------------------------------------===//

template <typename ValueT>
class SparseSetLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;
  SparseSetLattice(Value value, SetLattice<ValueT> &&elements)
      : dataflow::AbstractSparseLattice(value), elements(std::move(elements)) {}

  Attribute serialize(MLIRContext *ctx) const { return serializeSetNaive(ctx); }

  ChangeResult merge(const SetLattice<ValueT> &other) {
    return elements.join(other);
  }

  ChangeResult insert(const DenseSet<ValueT> &newElements) {
    return elements.insert(newElements);
  }

  ChangeResult markUnknown() { return elements.markUnknown(); }

  bool isUnknown() const { return elements.isUnknown(); }

  bool isUndefined() const { return elements.isUndefined(); }

  const DenseSet<ValueT> &getElements() const { return elements.getElements(); }

protected:
  SetLattice<ValueT> elements;

private:
  Attribute serializeSetNaive(MLIRContext *ctx) const {
    if (elements.isUndefined())
      return StringAttr::get(ctx, undefinedSetString);
    if (elements.isUnknown())
      return StringAttr::get(ctx, unknownSetString);
    SmallVector<Attribute> elementsVec;
    for (Attribute element : elements.getElements()) {
      elementsVec.push_back(element);
    }
    llvm::sort(elementsVec, sortAttributes);
    return ArrayAttr::get(ctx, elementsVec);
  }
};

//===----------------------------------------------------------------------===//
// MapOfSetsLattice
//===----------------------------------------------------------------------===//

template <typename KeyT, typename ElementT>
class MapOfSetsLattice : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  Attribute serialize(MLIRContext *ctx) const {
    return serializeMapOfSetsNaive(ctx);
  }

  ChangeResult join(const AbstractDenseLattice &other) {
    const auto &rhs =
        static_cast<const MapOfSetsLattice<KeyT, ElementT> &>(other);
    llvm::SmallDenseSet<DistinctAttr> keys;
    auto lhsRange = llvm::make_first_range(map);
    auto rhsRange = llvm::make_first_range(rhs.map);
    keys.insert(lhsRange.begin(), lhsRange.end());
    keys.insert(rhsRange.begin(), rhsRange.end());

    ChangeResult result = ChangeResult::NoChange;
    for (DistinctAttr key : keys) {
      auto lhsIt = map.find(key);
      auto rhsIt = rhs.map.find(key);
      assert(lhsIt != map.end() || rhsIt != rhs.map.end());

      // If present in both, join.
      if (lhsIt != map.end() && rhsIt != rhs.map.end()) {
        result |= lhsIt->getSecond().join(rhsIt->getSecond());
        continue;
      }

      // Copy from RHS if available only there.
      if (lhsIt == map.end()) {
        map.try_emplace(rhsIt->getFirst(), rhsIt->getSecond());
        result = ChangeResult::Change;
      }

      // Do nothing if available only in LHS.
    }
    return result;
  }

  /// Map all keys to all values.
  ChangeResult insert(const SetLattice<KeyT> &keysToUpdate,
                      const SetLattice<ElementT> &values) {
    if (keysToUpdate.isUnknown())
      return markAllUnknown();

    if (keysToUpdate.isUndefined())
      return ChangeResult::NoChange;

    return keysToUpdate.foreachElement(
        [&](DistinctAttr key, typename SetLattice<KeyT>::State state) {
          assert(state == SetLattice<KeyT>::State::Defined &&
                 "unknown must have been handled above");
          return joinPotentiallyMissing(key, values);
        });
  }

  ChangeResult markAllUnknown() {
    ChangeResult result = ChangeResult::NoChange;
    for (auto &it : map)
      result |= it.getSecond().join(SetLattice<ElementT>::getUnknown());
    return result;
  }

  const SetLattice<ElementT> &lookup(KeyT key) const {
    auto it = map.find(key);
    if (it == map.end())
      return SetLattice<ElementT>::getUndefined();
    return it->getSecond();
  }

protected:
  ChangeResult joinPotentiallyMissing(KeyT key,
                                      const SetLattice<ElementT> &value) {
    // Don't store explicitly undefined values in the mapping, keys absent from
    // the mapping are treated as implicitly undefined.
    if (value.isUndefined())
      return ChangeResult::NoChange;

    bool inserted;
    decltype(map.begin()) iterator;
    std::tie(iterator, inserted) = map.try_emplace(key, value);
    if (!inserted)
      return iterator->second.join(value);
    return ChangeResult::Change;
  }

  /// Maps a key to a set of values. When a key is not present in this map, it
  /// is considered to map to an uninitialized set.
  DenseMap<KeyT, SetLattice<ElementT>> map;

private:
  Attribute serializeMapOfSetsNaive(MLIRContext *ctx) const {
    SmallVector<Attribute> pointsToArray;

    for (const auto &[srcClass, destClasses] : map) {
      SmallVector<Attribute> pair = {srcClass};
      SmallVector<Attribute> aliasClasses;
      if (destClasses.isUnknown()) {
        aliasClasses.push_back(StringAttr::get(ctx, unknownSetString));
      } else if (destClasses.isUndefined()) {
        aliasClasses.push_back(StringAttr::get(ctx, undefinedSetString));
      } else {
        for (const Attribute &destClass : destClasses.getElements()) {
          aliasClasses.push_back(destClass);
        }
        llvm::sort(aliasClasses, sortAttributes);
      }
      pair.push_back(ArrayAttr::get(ctx, aliasClasses));
      pointsToArray.push_back(ArrayAttr::get(ctx, pair));
    }
    llvm::sort(pointsToArray, [&](Attribute a, Attribute b) {
      return sortArraysLexicographic(cast<ArrayAttr>(a), cast<ArrayAttr>(b));
    });
    return ArrayAttr::get(ctx, pointsToArray);
  }
};

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_ANALYSIS_DATAFLOW_LATTICE_H
