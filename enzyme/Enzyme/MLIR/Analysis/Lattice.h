#ifndef ENZYME_MLIR_ANALYSIS_DATAFLOW_LATTICE_H
#define ENZYME_MLIR_ANALYSIS_DATAFLOW_LATTICE_H

#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir {
namespace enzyme {

template <typename ValueT> class SetLattice {
public:
  enum class State {
    Undefined, ///< Has not been analyzed yet (lattice bottom).
    Defined,   ///< Has specific elements.
    Unknown    ///< Analyzed and may point to any class (lattice top).
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
  bool isCanonical() const;

  /// Returns an instance of SetLattice known not to have no elements.
  /// This is different from "undefined" and "unknown". The instance is *not* a
  /// classical singleton.
  static const SetLattice<ValueT> &getEmpty() {
    static const SetLattice<ValueT> empty(State::Defined);
    return empty;
  }

  /// Returns an instance of SetLattice in "undefined" state, i.e. without a
  /// set of elements. This is different from empty alias set, which
  /// indicates that the value is known not to alias with any alias class. The
  /// instance is *not* a classical singleton, there are other ways of obtaining
  /// it.
  static const SetLattice<ValueT> &getUndefined() { return undefinedSet; }

  /// Returns an instance of SetLattice for the "unknown" class. The instance
  /// is *not* a classical singleton, there are other ways of obtaining an
  /// "unknown" alias set.
  static const SetLattice<ValueT> &getUnknown() { return unknownSet; }

  bool operator==(const SetLattice<ValueT> &other) const;

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const SetLattice<ValueT> &setLattice);

  void print(llvm::raw_ostream &os) const {
    if (isUnknown()) {
      os << "<unknown>";
    } else if (isUndefined()) {
      os << "<undefined>";
    } else {
      llvm::interleaveComma(elements, os << "{");
      os << "}";
    }
  }

  ChangeResult foreach (
      function_ref<ChangeResult(ValueT, State)> callback) const;

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

} // namespace enzyme
} // namespace mlir

#endif
