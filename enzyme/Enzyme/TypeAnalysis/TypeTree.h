//===- TypeTree.cpp - Declaration of Type Analysis Type Trees   -----------===//
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
// This file contains the declaration of TypeTrees -- a class
// representing all of the underlying types of a particular LLVM value. This
// consists of a map of memory offsets to an underlying ConcreteType. This
// permits TypeTrees to represent distinct underlying types at different
// locations. Presently, TypeTree's have both a fixed depth of memory lookups
// and a maximum offset to ensure that Type Analysis eventually terminates.
// In the future this should be modified to better represent recursive types
// rather than limiting the depth.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_TYPE_ANALYSIS_TYPE_TREE_H
#define ENZYME_TYPE_ANALYSIS_TYPE_TREE_H 1

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../Utils.h"
#include "BaseType.h"
#include "ConcreteType.h"

/// Maximum offset for type trees to keep
extern "C" {
extern llvm::cl::opt<int> MaxTypeOffset;
extern llvm::cl::opt<bool> EnzymeTypeWarning;
extern llvm::cl::opt<unsigned> EnzymeMaxTypeDepth;
}

/// Helper function to print a vector of ints to a string
static inline std::string to_string(const std::vector<int> x) {
  std::string out = "[";
  for (unsigned i = 0; i < x.size(); ++i) {
    if (i != 0)
      out += ",";
    out += std::to_string(x[i]);
  }
  out += "]";
  return out;
}

class TypeTree;

typedef std::shared_ptr<const TypeTree> TypeResult;
typedef std::map<const std::vector<int>, ConcreteType> ConcreteTypeMapType;
typedef std::map<const std::vector<int>, const TypeResult> TypeTreeMapType;

/// Class representing the underlying types of values as
/// sequences of offsets to a ConcreteType
class TypeTree : public std::enable_shared_from_this<TypeTree> {
private:
  // mapping of known indices to type if one exists
  ConcreteTypeMapType mapping;
  std::vector<int> minIndices;

public:
  TypeTree() {}
  TypeTree(ConcreteType dat) {
    if (dat != ConcreteType(BaseType::Unknown)) {
      mapping.insert(std::pair<const std::vector<int>, ConcreteType>({}, dat));
    }
  }

  static TypeTree parse(llvm::StringRef str, llvm::LLVMContext &ctx) {
    using namespace llvm;
    assert(str[0] == '{');
    str = str.substr(1);

    TypeTree Result;
    while (true) {
      while (str[0] == ' ')
        str = str.substr(1);
      if (str[0] == '}')
        break;

      assert(str[0] == '[');
      str = str.substr(1);

      std::vector<int> idxs;
      while (true) {
        while (str[0] == ' ')
          str = str.substr(1);
        if (str[0] == ']') {
          str = str.substr(1);
          break;
        }

        int idx;
        bool failed = str.consumeInteger(10, idx);
        (void)failed;
        assert(!failed);
        idxs.push_back(idx);

        while (str[0] == ' ')
          str = str.substr(1);

        if (str[0] == ',') {
          str = str.substr(1);
        }
      }

      while (str[0] == ' ')
        str = str.substr(1);

      assert(str[0] == ':');
      str = str.substr(1);

      while (str[0] == ' ')
        str = str.substr(1);

      auto endval = str.find(',');
      auto endval2 = str.find('}');
      auto endval3 = str.find(' ');

      if (endval2 != StringRef::npos &&
          (endval == StringRef::npos || endval2 < endval))
        endval = endval2;
      if (endval3 != StringRef::npos &&
          (endval == StringRef::npos || endval3 < endval))
        endval = endval3;
      assert(endval != StringRef::npos);

      auto tystr = str.substr(0, endval);
      str = str.substr(endval);

      ConcreteType CT(tystr, ctx);
      Result.mapping.emplace(idxs, CT);
      if (Result.minIndices.size() < idxs.size()) {
        for (size_t i = Result.minIndices.size(), end = idxs.size(); i < end;
             ++i) {
          Result.minIndices.push_back(idxs[i]);
        }
      }
      for (size_t i = 0, end = idxs.size(); i < end; ++i) {
        if (idxs[i] < Result.minIndices[i])
          Result.minIndices[i] = idxs[i];
      }

      while (str[0] == ' ')
        str = str.substr(1);

      if (str[0] == ',') {
        str = str.substr(1);
      }
    }

    return Result;
  }

  /// Utility helper to lookup the mapping
  const ConcreteTypeMapType &getMapping() const { return mapping; }

  /// Lookup the underlying ConcreteType at a given offset sequence
  /// or Unknown if none exists
  ConcreteType operator[](const std::vector<int> Seq) const {
    auto Found0 = mapping.find(Seq);
    if (Found0 != mapping.end())
      return Found0->second;
    size_t Len = Seq.size();
    if (Len == 0)
      return BaseType::Unknown;

    std::vector<std::vector<int>> todo[2];
    todo[0].push_back({});
    int parity = 0;
    for (size_t i = 0, Len = Seq.size(); i < Len - 1; ++i) {
      for (auto prev : todo[parity]) {
        prev.push_back(-1);
        if (mapping.find(prev) != mapping.end())
          todo[1 - parity].push_back(prev);
        if (Seq[i] != -1) {
          prev.back() = Seq[i];
          if (mapping.find(prev) != mapping.end())
            todo[1 - parity].push_back(prev);
        }
      }
      todo[parity].clear();
      parity = 1 - parity;
    }

    size_t i = Len - 1;
    for (auto prev : todo[parity]) {
      prev.push_back(-1);
      auto Found = mapping.find(prev);
      if (Found != mapping.end())
        return Found->second;
      if (Seq[i] != -1) {
        prev.back() = Seq[i];
        Found = mapping.find(prev);
        if (Found != mapping.end())
          return Found->second;
      }
    }
    return BaseType::Unknown;
  }

  // Return true if this type tree is fully known (i.e. there
  // is no more information which could be added).
  bool IsFullyDetermined() const {
    std::vector<int> offsets = {-1};
    while (1) {
      auto found = mapping.find(offsets);
      if (found == mapping.end())
        return false;
      if (found->second != BaseType::Pointer)
        return true;
      offsets.push_back(-1);
    }
  }

  /// Return if changed
  bool insert(const std::vector<int> Seq, ConcreteType CT,
              bool PointerIntSame = false) {
    size_t SeqSize = Seq.size();
    if (SeqSize > EnzymeMaxTypeDepth) {
      if (EnzymeTypeWarning) {
        if (CustomErrorHandler) {
          CustomErrorHandler("TypeAnalysisDepthLimit", nullptr,
                             ErrorType::TypeDepthExceeded, this, nullptr,
                             nullptr);
        } else
          llvm::errs() << "not handling more than " << EnzymeMaxTypeDepth
                       << " pointer lookups deep dt:" << str()
                       << " adding v: " << to_string(Seq) << ": " << CT.str()
                       << "\n";
      }
      return false;
    }
    if (SeqSize == 0) {
      mapping.insert(std::pair<const std::vector<int>, ConcreteType>(Seq, CT));
      return true;
    }

    // check types at lower pointer offsets are either pointer or
    // anything. Don't insert into an anything
    {
      std::vector<int> tmp(Seq);
      while (tmp.size() > 0) {
        tmp.erase(tmp.end() - 1);
        auto found = mapping.find(tmp);
        if (found != mapping.end()) {
          if (found->second == BaseType::Anything)
            return false;
          if (found->second != BaseType::Pointer) {
            llvm::errs() << "FAILED CT: " << str()
                         << " adding Seq: " << to_string(Seq) << ": "
                         << CT.str() << "\n";
          }
          assert(found->second == BaseType::Pointer);
        }
      }
    }

    bool changed = false;
    // Check if there is an existing match, e.g. [-1, -1, -1] and inserting
    // [-1, 8, -1]
    {
      for (const auto &pair : llvm::make_early_inc_range(mapping)) {
        if (pair.first.size() == SeqSize) {
          // Whether the the inserted val (e.g. [-1, 0] or [0, 0]) is at least
          // as general as the existing map val (e.g. [0, 0]).
          bool newMoreGeneralThanOld = true;
          // Whether the the existing val (e.g. [-1, 0] or [0, 0]) is at least
          // as general as the inserted map val (e.g. [0, 0]).
          bool oldMoreGeneralThanNew = true;
          for (unsigned i = 0; i < SeqSize; i++) {
            if (pair.first[i] == Seq[i])
              continue;
            if (Seq[i] == -1) {
              oldMoreGeneralThanNew = false;
            } else if (pair.first[i] == -1) {
              newMoreGeneralThanOld = false;
            } else {
              oldMoreGeneralThanNew = false;
              newMoreGeneralThanOld = false;
              break;
            }
          }

          if (oldMoreGeneralThanNew) {
            // Inserting an existing or less general version
            if (CT == pair.second)
              return false;

            // Inserting an existing or less general version (with pointer-int
            // equivalence)
            if (PointerIntSame)
              if ((CT == BaseType::Pointer &&
                   pair.second == BaseType::Integer) ||
                  (CT == BaseType::Integer && pair.second == BaseType::Pointer))
                return false;

            // Inserting into an anything. Since from above we know this is not
            // an anything, the inserted value contains no new information
            if (pair.second == BaseType::Anything)
              return false;

            // Inserting say a [0]:anything into a [-1]:Float
            if (CT == BaseType::Anything)
              continue;

            // Otherwise, inserting a non-equivalent pair into a more general
            // slot. This is invalid.
            llvm::errs() << "inserting into : " << str() << " with "
                         << to_string(Seq) << " of " << CT.str() << "\n";
            llvm_unreachable("illegal insertion");
          } else if (newMoreGeneralThanOld) {
            // This new is strictly more general than the old. If they were
            // equivalent, the case above would have been hit.

            if (CT == BaseType::Anything || CT == pair.second) {
              // previous equivalent values or values overwritten by
              // an anything are removed
              changed = true;
              mapping.erase(pair.first);
              continue;
            }

            // Inserting an existing or less general version (with pointer-int
            // equivalence)
            if (PointerIntSame)
              if ((CT == BaseType::Pointer &&
                   pair.second == BaseType::Integer) ||
                  (CT == BaseType::Integer &&
                   pair.second == BaseType::Pointer)) {
                changed = true;
                mapping.erase(pair.first);
                continue;
              }

            // Keep lingering anythings if not being overwritten, even if this
            // (e.g. Float) applies to more locations. Therefore it is legal to
            // have [-1]:Float, [8]:Anything
            if (CT != BaseType::Anything && pair.second == BaseType::Anything)
              continue;

            // Otherwise, inserting a more general non-equivalent pair. This is
            // invalid.
            llvm::errs() << "inserting into : " << str() << " with "
                         << to_string(Seq) << " of " << CT.str() << "\n";
            llvm_unreachable("illegal insertion");
          }
        }
      }
    }

    bool possibleDeletion = false;
    size_t minLen =
        (minIndices.size() <= SeqSize) ? minIndices.size() : SeqSize;
    for (size_t i = 0; i < minLen; i++) {
      if (minIndices[i] > Seq[i]) {
        if (minIndices[i] > MaxTypeOffset)
          possibleDeletion = true;
        minIndices[i] = Seq[i];
      }
    }

    if (minIndices.size() < SeqSize) {
      for (size_t i = minIndices.size(), end = SeqSize; i < end; ++i) {
        minIndices.push_back(Seq[i]);
      }
    }

    if (possibleDeletion) {
      for (const auto &pair : llvm::make_early_inc_range(mapping)) {
        size_t i = 0;
        bool mustKeep = false;
        bool considerErase = false;
        for (int val : pair.first) {
          if (val > MaxTypeOffset) {
            if (val == minIndices[i]) {
              mustKeep = true;
              break;
            }
            considerErase = true;
          }
          ++i;
        }
        if (!mustKeep && considerErase) {
          mapping.erase(pair.first);
          changed = true;
        }
      }
    }

    size_t i = 0;
    bool keep = false;
    bool considerErase = false;
    for (auto val : Seq) {
      if (val > MaxTypeOffset) {
        if (val == minIndices[i]) {
          keep = true;
          break;
        }
        considerErase = true;
      }
      i++;
    }
    if (considerErase && !keep)
      return changed;
    mapping.insert(std::pair<const std::vector<int>, ConcreteType>(Seq, CT));
    return true;
  }

  /// How this TypeTree compares with another
  bool operator<(const TypeTree &vd) const { return mapping < vd.mapping; }

  /// Whether this TypeTree contains any information
  bool isKnown() const {
#ifndef NDEBUG
    for (const auto &pair : mapping) {
      // we should assert here as we shouldn't keep any unknown maps for
      // efficiency
      assert(pair.second.isKnown());
    }
#endif
    return mapping.size() != 0;
  }

  /// Whether this TypeTree knows any non-pointer information
  bool isKnownPastPointer() const {
    for (auto &pair : mapping) {
      // we should assert here as we shouldn't keep any unknown maps for
      // efficiency
      assert(pair.second.isKnown());
      if (pair.first.size() == 0) {
        assert(pair.second == BaseType::Pointer ||
               pair.second == BaseType::Anything);
        continue;
      }
      return true;
    }
    return false;
  }

  /// Select only the Integer ConcreteTypes
  TypeTree JustInt() const {
    TypeTree vd;
    for (auto &pair : mapping) {
      if (pair.second == BaseType::Integer) {
        vd.insert(pair.first, pair.second);
      }
    }

    return vd;
  }

  /// Prepend an offset to all mappings
  TypeTree Only(int Off, llvm::Instruction *orig) const {
    TypeTree Result;
    Result.minIndices.reserve(1 + minIndices.size());
    Result.minIndices.push_back(Off);
    for (auto midx : minIndices)
      Result.minIndices.push_back(midx);

    if (Result.minIndices.size() > EnzymeMaxTypeDepth) {
      Result.minIndices.pop_back();
      if (EnzymeTypeWarning) {
        if (CustomErrorHandler) {
          CustomErrorHandler("TypeAnalysisDepthLimit", wrap(orig),
                             ErrorType::TypeDepthExceeded, this, nullptr,
                             nullptr);
        } else if (orig) {
          EmitWarning("TypeAnalysisDepthLimit", *orig, *orig,
                      " not handling more than ", EnzymeMaxTypeDepth,
                      " pointer lookups deep dt: ", str(), " only(", Off, ")");
        } else {
          llvm::errs() << "not handling more than " << EnzymeMaxTypeDepth
                       << " pointer lookups deep dt:" << str() << " only("
                       << Off << "): "
                       << "\n";
        }
      }
    }

    for (const auto &pair : mapping) {
      if (pair.first.size() == EnzymeMaxTypeDepth)
        continue;
      std::vector<int> Vec;
      Vec.reserve(pair.first.size() + 1);
      Vec.push_back(Off);
      for (auto Val : pair.first)
        Vec.push_back(Val);
      Result.mapping.insert(
          std::pair<const std::vector<int>, ConcreteType>(Vec, pair.second));
    }
    return Result;
  }

  /// Peel off the outermost index at offset 0
  TypeTree Data0() const {
    TypeTree Result;

    for (const auto &pair : mapping) {
      if (pair.first.size() == 0) {
        llvm::errs() << str() << "\n";
      }
      assert(pair.first.size() != 0);

      if (pair.first[0] == -1) {
        std::vector<int> next(pair.first.begin() + 1, pair.first.end());
        Result.mapping.insert(
            std::pair<const std::vector<int>, ConcreteType>(next, pair.second));
        for (size_t i = 0, Len = next.size(); i < Len; ++i) {
          if (i == Result.minIndices.size())
            Result.minIndices.push_back(next[i]);
          else if (next[i] < Result.minIndices[i])
            Result.minIndices[i] = next[i];
        }
      }
    }
    for (const auto &pair : mapping) {
      if (pair.first[0] == 0) {
        std::vector<int> next(pair.first.begin() + 1, pair.first.end());
        // We do insertion like this to force an error
        // on the orIn operation if there is an incompatible
        // merge. The insert operation does not error.
        Result.orIn(next, pair.second);
      }
    }

    return Result;
  }

  /// Optimized version of Data0()[{}]
  ConcreteType Inner0() const {
    ConcreteType CT = operator[]({-1});
    CT |= operator[]({0});
    return CT;
  }

  /// Remove any mappings in the range [start, end) or [len, inf)
  /// This function has special handling for -1's
  TypeTree Clear(size_t start, size_t end, size_t len) const {
    TypeTree Result;

    // Note that below do insertion with the orIn operator
    // to force an error if there is an incompatible
    // merge. The insert operation does not error.

    for (const auto &pair : mapping) {
      assert(pair.first.size() != 0);

      if (pair.first[0] == -1) {
        // For "all index" calculations, explicitly
        // add mappings for regions in range
        auto next = pair.first;
        for (size_t i = 0; i < start; ++i) {
          next[0] = i;
          Result.orIn(next, pair.second);
        }
        for (size_t i = end; i < len; ++i) {
          next[0] = i;
          Result.orIn(next, pair.second);
        }
      } else if ((size_t)pair.first[0] < start ||
                 ((size_t)pair.first[0] >= end &&
                  (size_t)pair.first[0] < len)) {
        // Otherwise simply check that the given offset is in range

        Result.insert(pair.first, pair.second);
      }
    }

    // TODO canonicalize this
    return Result;
  }

  /// Select all submappings whose first index is in range [0, len) and remove
  /// the first index. This is the inverse of the `Only` operation
  TypeTree Lookup(size_t len, const llvm::DataLayout &dl) const {

    // Map of indices[1:] => ( End => possible Index[0] )
    std::map<std::vector<int>, std::map<ConcreteType, std::set<int>>> staging;

    for (const auto &pair : mapping) {
      assert(pair.first.size() != 0);

      // Pointer is at offset 0 from this object
      if (pair.first[0] != 0 && pair.first[0] != -1)
        continue;

      if (pair.first.size() == 1) {
        assert(pair.second == ConcreteType(BaseType::Pointer) ||
               pair.second == ConcreteType(BaseType::Anything));
        continue;
      }

      if (pair.first[1] == -1) {
      } else {
        if ((size_t)pair.first[1] >= len)
          continue;
      }

      std::vector<int> next(pair.first.begin() + 2, pair.first.end());

      staging[next][pair.second].insert(pair.first[1]);
    }

    TypeTree Result;
    for (auto &pair : staging) {
      auto &pnext = pair.first;
      for (auto &pair2 : pair.second) {
        auto dt = pair2.first;
        const auto &set = pair2.second;

        bool legalCombine = set.count(-1);

        // See if we can canonicalize the outermost index into a -1
        if (!legalCombine) {
          size_t chunk = 1;
          // Implicit pointer
          if (pnext.size() > 0) {
            chunk = dl.getPointerSizeInBits() / 8;
          } else {
            if (auto flt = dt.isFloat()) {
              chunk = dl.getTypeSizeInBits(flt) / 8;
            } else if (dt == BaseType::Pointer) {
              chunk = dl.getPointerSizeInBits() / 8;
            }
          }

          legalCombine = true;
          for (size_t i = 0; i < len; i += chunk) {
            if (!set.count(i)) {
              legalCombine = false;
              break;
            }
          }
        }

        std::vector<int> next;
        next.reserve(pnext.size() + 1);
        next.push_back(-1);
        for (auto v : pnext)
          next.push_back(v);

        if (legalCombine) {
          Result.insert(next, dt, /*intsAreLegalPointerSub*/ true);
        } else {
          for (auto e : set) {
            next[0] = e;
            Result.insert(next, dt);
          }
        }
      }
    }

    return Result;
  }

  /// Given that this tree represents something of at most size len,
  /// canonicalize this, creating -1's where possible
  void CanonicalizeInPlace(size_t len, const llvm::DataLayout &dl) {
    bool canonicalized = true;
    for (const auto &pair : mapping) {
      assert(pair.first.size() != 0);
      if (pair.first[0] != -1) {
        canonicalized = false;
        break;
      }
    }
    if (canonicalized)
      return;

    // Map of indices[1:] => ( End => possible Index[0] )
    std::map<const std::vector<int>, std::map<ConcreteType, std::set<int>>>
        staging;

    for (const auto &pair : mapping) {

      std::vector<int> next(pair.first.begin() + 1, pair.first.end());
      if (pair.first[0] != -1) {
        if ((size_t)pair.first[0] >= len) {
          llvm::errs() << str() << "\n";
          llvm::errs() << " canonicalizing " << len << "\n";
          llvm::report_fatal_error("Canonicalization failed");
        }
      }
      staging[next][pair.second].insert(pair.first[0]);
    }

    // TypeTree mappings which did not get combined
    std::map<const std::vector<int>, ConcreteType> unCombinedToAdd;

    // TypeTree mappings which did get combined into an outer -1
    std::map<const std::vector<int>, ConcreteType> combinedToAdd;

    for (const auto &pair : staging) {
      auto &pnext = pair.first;
      for (const auto &pair2 : pair.second) {
        auto dt = pair2.first;
        const auto &set = pair2.second;

        bool legalCombine = false;

        // See if we can canonicalize the outermost index into a -1
        if (!set.count(-1)) {
          size_t chunk = 1;
          if (pnext.size() > 0) {
            chunk = dl.getPointerSizeInBits() / 8;
          } else {
            if (auto flt = dt.isFloat()) {
              chunk = dl.getTypeSizeInBits(flt) / 8;
            } else if (dt == BaseType::Pointer) {
              chunk = dl.getPointerSizeInBits() / 8;
            }
          }

          legalCombine = true;
          for (size_t i = 0; i < len; i += chunk) {
            if (!set.count(i)) {
              legalCombine = false;
              break;
            }
          }
        }

        std::vector<int> next;
        next.reserve(pnext.size() + 1);
        next.push_back(-1);
        for (auto v : pnext)
          next.push_back(v);

        if (legalCombine) {
          combinedToAdd.emplace(next, dt);
        } else {
          for (auto e : set) {
            next[0] = e;
            unCombinedToAdd.emplace(next, dt);
          }
        }
      }
    }

    // If we combined nothing, just return since there are no
    // changes.
    if (combinedToAdd.size() == 0) {
      return;
    }

    // Non-combined ones do not conflict, since they were already in
    // a TT which we can assume contained no conflicts.
    mapping = std::move(unCombinedToAdd);
    if (minIndices.size() > 0) {
      minIndices[0] = -1;
    }

    // Fusing several terms into a minus one can create a conflict
    // if the prior minus one was already in the map
    // time, or also generated by fusion.
    // E.g. {-1:Anything, [0]:Pointer} on 8 -> create a [-1]:Pointer
    //   which conflicts
    // Alternatively [-1,-1,-1]:Pointer, and generated a [-1,0,-1] fusion
    for (const auto &pair : combinedToAdd) {
      insert(pair.first, pair.second);
    }

    return;
  }

  /// Keep only pointers (or anything's) to a repeated value (represented by -1)
  TypeTree KeepMinusOne(bool &legal) const {
    TypeTree dat;

    for (const auto &pair : mapping) {

      assert(pair.first.size() != 0);

      // Pointer is at offset 0 from this object
      if (pair.first[0] != 0 && pair.first[0] != -1)
        continue;

      if (pair.first.size() == 1) {
        if (pair.second == BaseType::Pointer ||
            pair.second == BaseType::Anything) {
          dat.insert(pair.first, pair.second);
          continue;
        }
        legal = false;
        break;
      }

      if (pair.first[1] == -1) {
        dat.insert(pair.first, pair.second);
      }
    }

    return dat;
  }

  llvm::Type *IsAllFloat(const size_t size, const llvm::DataLayout &dl) const {
    auto m1 = TypeTree::operator[]({-1});
    if (auto FT = m1.isFloat())
      return FT;

    auto m0 = TypeTree::operator[]({0});

    if (auto flt = m0.isFloat()) {
      size_t chunk = dl.getTypeSizeInBits(flt) / 8;
      for (size_t i = chunk; i < size; i += chunk) {
        auto mx = TypeTree::operator[]({(int)i});
        if (auto f2 = mx.isFloat()) {
          if (f2 != flt)
            return nullptr;
        } else
          return nullptr;
      }
      return flt;
    } else {
      return nullptr;
    }
  }

  /// Replace mappings in the range in [offset, offset+maxSize] with those in
  // [addOffset, addOffset + maxSize]. In other words, select all mappings in
  // [offset, offset+maxSize] then add `addOffset`
  TypeTree ShiftIndices(const llvm::DataLayout &dl, const int offset,
                        const int maxSize, size_t addOffset = 0) const {

    // If we have no terms 1+ layer deep return the current result as a shift
    // won't change anything. This also makes the latercode simpler as it
    // can assume at least a first index exists.
    if (minIndices.size() == 0)
      return *this;

    // If we have no size in return, simply return an empty type tree. Again
    // this simplifies later code which can assume that a minus one expantion
    // will always result in an added variable (which would not be the case
    // on a size == 0).
    if (maxSize == 0)
      return TypeTree();

    TypeTree Result;

    // The normal orIn / insert methods do collision checking, which is slow
    //  (and presently O(n)). This is because an expansion of a -1 which could
    //  conflict with a fixed value. Consider calling this
    //  ShiftIndicies(offset=0, maxSize=2, addOffset=0, tt={[-1]:Integer,
    //  [1]:Anything}) the -1 would expand to [0]:Int, [1]:Int, which would need
    //  to be merged with [1]:Anything
    //
    // The only possible values which can cause a conflict are minus -1's.
    // As a result, we start with a fast insertion (aka without check) of
    // non-expanded values, since they just do a literal shift which needs no
    // extra checking, besides bounds checks.
    //
    // Since we're doing things manually, we also need to manually preserve TT
    // invariants. Specifically, TT limits all values to have offsets <
    // MAX_OFFSET, unless it is the smallest offset at that depth. (e.g. so we
    // can still hava  typetree {[123456]:Int}, even if limit is 100).
    //
    // First compute the minimum 0th index to be kept.
    Result.minIndices.resize(minIndices.size(), INT_MAX);

    for (const auto &pair : mapping) {
      if (pair.first.size() == 0) {
        if (pair.second == BaseType::Pointer ||
            pair.second == BaseType::Anything) {
          Result.mapping.emplace(pair.first, pair.second);
          continue;
        }

        llvm::errs() << "could not unmerge " << str() << "\n";
        assert(0 && "ShiftIndices called on a nonpointer/anything");
        llvm_unreachable("ShiftIndices called on a nonpointer/anything");
      }

      int next0 = pair.first[0];

      if (next0 == -1) {
        if (maxSize == -1) {
          // Max size does not clip the next index

          // If we have a follow up offset add, we lose the -1 since we only
          // represent [0, inf) with -1 not the [addOffset, inf) required here
          if (addOffset != 0) {
            next0 = addOffset;
          }

        } else {
          // We're going to insert addOffset + 0...maxSize so the new minIndex
          // is addOffset
          Result.minIndices[0] = addOffset;
          for (size_t i = 1, sz = pair.first.size(); i < sz; i++)
            if (pair.first[i] < Result.minIndices[i])
              Result.minIndices[i] = pair.first[i];
          continue;
        }
      } else {
        // Too small for range
        if (next0 < offset) {
          continue;
        }
        next0 -= offset;

        if (maxSize != -1) {
          if (next0 >= maxSize)
            continue;
        }

        next0 += addOffset;
      }
      if (next0 < Result.minIndices[0])
        Result.minIndices[0] = next0;
      for (size_t i = 1, sz = pair.first.size(); i < sz; i++)
        if (pair.first[i] < Result.minIndices[i])
          Result.minIndices[i] = pair.first[i];
    }

    // Max depth of actual inserted values
    size_t maxInsertedDepth = 0;

    // Insert all
    for (const auto &pair : mapping) {
      if (pair.first.size() == 0)
        continue;

      int next0 = pair.first[0];

      if (next0 == -1) {
        if (maxSize == -1) {
          // Max size does not clip the next index

          // If we have a follow up offset add, we lose the -1 since we only
          // represent [0, inf) with -1 not the [addOffset, inf) required here
          if (addOffset != 0) {
            next0 = addOffset;
          }

        } else {
          // This needs to become 0...maxSize handled separately as it is the
          // only insertion that could have collisions
          continue;
        }
      } else {
        // Too small for range
        if (next0 < offset) {
          continue;
        }
        next0 -= offset;

        if (maxSize != -1) {
          if (next0 >= maxSize)
            continue;
        }

        next0 += addOffset;
      }

      // If after moving this would not merit being kept for being a min index
      // or being within the max type offset, skip it.
      if (next0 > MaxTypeOffset) {
        bool minIndex = next0 == Result.minIndices[0];
        if (!minIndex)
          for (size_t i = 1; i < pair.first.size(); i++) {
            if (pair.first[i] == Result.minIndices[i]) {
              minIndex = true;
              break;
            }
          }
        if (!minIndex)
          continue;
      }

      std::vector<int> next(pair.first);
      next[0] = next0;
      Result.mapping.emplace(next, pair.second);
      if (next.size() > maxInsertedDepth)
        maxInsertedDepth = next.size();
    }

    // Insert and expand the minus one, if needed
    if (maxSize != -1)
      for (const auto &pair : mapping) {
        if (pair.first.size() == 0)
          continue;
        if (pair.first[0] != -1)
          continue;

        size_t chunk = 1;
        std::vector<int> next(pair.first);
        auto op = operator[]({next[0]});
        if (auto flt = op.isFloat()) {
          chunk = dl.getTypeSizeInBits(flt) / 8;
        } else if (op == BaseType::Pointer) {
          chunk = dl.getPointerSizeInBits() / 8;
        }
        auto offincr = (chunk - offset % chunk) % chunk;
        bool inserted = false;
        for (int i = offincr; i < maxSize; i += chunk) {
          next[0] = i + addOffset;
          ConcreteType prev(pair.second);
          // We can use faster checks here, since we know there can be no
          // -1's that we would conflict with, only conflicts from previous
          // fixed value insertions.
          auto found = Result.mapping.find(next);
          if (found != Result.mapping.end()) {
            // orIn returns if changed, update the value in the map if so
            // with the new value.
            if (prev.orIn(found->second, /*pointerIntSame*/ false))
              found->second = prev;
          } else {
            Result.mapping.emplace(next, pair.second);
          }
          inserted = true;
        }
        if (inserted && next.size() > maxInsertedDepth)
          maxInsertedDepth = next.size();
      }

    // Resize minIndices down if we dropped any higher-depth indices for being
    // out of scope.
    Result.minIndices.resize(maxInsertedDepth);
    return Result;
  }

  /// Keep only mappings where the type is not an `Anything`
  TypeTree PurgeAnything() const {
    TypeTree Result;
    Result.minIndices.reserve(minIndices.size());
    for (const auto &pair : mapping) {
      if (pair.second == ConcreteType(BaseType::Anything))
        continue;
      Result.mapping.insert(pair);
      for (size_t i = 0, Len = pair.first.size(); i < Len; ++i) {
        if (i == Result.minIndices.size())
          Result.minIndices.push_back(pair.first[i]);
        else if (pair.first[i] < Result.minIndices[i])
          Result.minIndices[i] = pair.first[i];
      }
    }
    return Result;
  }

  /// Replace -1 with 0
  TypeTree ReplaceMinus() const {
    TypeTree dat;
    for (const auto &pair : mapping) {
      if (pair.second == ConcreteType(BaseType::Anything))
        continue;
      std::vector<int> nex = pair.first;
      for (auto &v : nex)
        if (v == -1)
          v = 0;
      dat.insert(nex, pair.second);
    }
    return dat;
  }

  /// Replace all integer subtypes with anything
  void ReplaceIntWithAnything() {
    for (auto &pair : mapping) {
      if (pair.second == BaseType::Integer) {
        pair.second = BaseType::Anything;
      }
    }
  }

  /// Keep only mappings where the type is an `Anything`
  TypeTree JustAnything() const {
    TypeTree dat;
    for (const auto &pair : mapping) {
      if (pair.second != ConcreteType(BaseType::Anything))
        continue;
      dat.insert(pair.first, pair.second);
    }
    return dat;
  }

  /// Chceck equality of two TypeTrees
  bool operator==(const TypeTree &RHS) const { return mapping == RHS.mapping; }

  /// Set this to another TypeTree, returning if this was changed
  bool operator=(const TypeTree &RHS) {
    if (*this == RHS)
      return false;
    minIndices = RHS.minIndices;
    mapping.clear();
    for (const auto &elems : RHS.mapping) {
      mapping.emplace(elems);
    }
    return true;
  }

  bool checkedOrIn(const std::vector<int> &Seq, ConcreteType RHS,
                   bool PointerIntSame, bool &LegalOr) {
    assert(RHS != BaseType::Unknown);
    ConcreteType CT = operator[](Seq);

    bool subchanged = CT.checkedOrIn(RHS, PointerIntSame, LegalOr);
    if (!subchanged)
      return false;
    if (!LegalOr)
      return subchanged;

    auto SeqSize = Seq.size();

    if (SeqSize > 0) {
      // check pointer abilities from before
      for (size_t i = 0; i < SeqSize; ++i) {
        std::vector<int> tmp(Seq.begin(), Seq.end() - 1 - i);
        auto found = mapping.find(tmp);
        if (found != mapping.end()) {
          if (!(found->second == BaseType::Pointer ||
                found->second == BaseType::Anything)) {
            LegalOr = false;
            return false;
          }
        }
      }

      // Check if there is an existing match, e.g. [-1, -1, -1] and inserting
      // [-1, 8, -1]
      {
        for (const auto &pair : llvm::make_early_inc_range(mapping)) {
          if (pair.first.size() == SeqSize) {
            // Whether the the inserted val (e.g. [-1, 0] or [0, 0]) is at least
            // as general as the existing map val (e.g. [0, 0]).
            bool newMoreGeneralThanOld = true;
            // Whether the the existing val (e.g. [-1, 0] or [0, 0]) is at least
            // as general as the inserted map val (e.g. [0, 0]).
            bool oldMoreGeneralThanNew = true;
            for (unsigned i = 0; i < SeqSize; i++) {
              if (pair.first[i] == Seq[i])
                continue;
              if (Seq[i] == -1) {
                oldMoreGeneralThanNew = false;
              } else if (pair.first[i] == -1) {
                newMoreGeneralThanOld = false;
              } else {
                oldMoreGeneralThanNew = false;
                newMoreGeneralThanOld = false;
                break;
              }
            }

            if (oldMoreGeneralThanNew) {
              // Inserting an existing or less general version
              if (CT == pair.second)
                return false;

              // Inserting an existing or less general version (with pointer-int
              // equivalence)
              if (PointerIntSame)
                if ((CT == BaseType::Pointer &&
                     pair.second == BaseType::Integer) ||
                    (CT == BaseType::Integer &&
                     pair.second == BaseType::Pointer))
                  return false;

              // Inserting into an anything. Since from above we know this is
              // not an anything, the inserted value contains no new information
              if (pair.second == BaseType::Anything)
                return false;

              // Inserting say a [0]:anything into a [-1]:Float
              if (CT == BaseType::Anything) {
                // If both at same index, remove old index
                if (newMoreGeneralThanOld)
                  mapping.erase(pair.first);
                continue;
              }

              // Otherwise, inserting a non-equivalent pair into a more general
              // slot. This is invalid.
              LegalOr = false;
              return false;
            } else if (newMoreGeneralThanOld) {
              // This new is strictly more general than the old. If they were
              // equivalent, the case above would have been hit.

              if (CT == BaseType::Anything || CT == pair.second) {
                // previous equivalent values or values overwritten by
                // an anything are removed
                mapping.erase(pair.first);
                continue;
              }

              // Inserting an existing or less general version (with pointer-int
              // equivalence)
              if (PointerIntSame)
                if ((CT == BaseType::Pointer &&
                     pair.second == BaseType::Integer) ||
                    (CT == BaseType::Integer &&
                     pair.second == BaseType::Pointer)) {
                  mapping.erase(pair.first);
                  continue;
                }

              // Keep lingering anythings if not being overwritten, even if this
              // (e.g. Float) applies to more locations. Therefore it is legal
              // to have [-1]:Float, [8]:Anything
              if (CT != BaseType::Anything && pair.second == BaseType::Anything)
                continue;

              // Otherwise, inserting a more general non-equivalent pair. This
              // is invalid.
              LegalOr = false;
              return false;
            }
          }
        }
      }
    }

    return insert(Seq, CT);
  }

  bool orIn(const std::vector<int> &Seq, ConcreteType RHS,
            bool PointerIntSame = false) {
    bool LegalOr = true;
    bool Result = checkedOrIn(Seq, RHS, PointerIntSame, LegalOr);
    assert(LegalOr);
    return Result;
  }

  /// Set this to the logical or of itself and RHS, returning whether this value
  /// changed Setting `PointerIntSame` considers pointers and integers as
  /// equivalent If this is an illegal operation, `LegalOr` will be set to false
  bool checkedOrIn(const TypeTree &RHS, bool PointerIntSame, bool &LegalOr) {
    // Add fast path where nothing could change because all potentially inserted
    // value are already in the map. Save all other ones that may need slow
    // insertion
    std::vector<std::pair<const std::vector<int>, ConcreteType>> todo;

    for (const auto &pair : RHS.mapping) {
      auto found = mapping.find(pair.first);
      if (found == mapping.end()) {
        todo.emplace_back(pair);
        continue;
      }
      bool SubLegalOr = true;
      auto cur = found->second;
      bool SubChanged =
          cur.checkedOrIn(pair.second, PointerIntSame, SubLegalOr);
      if (!SubLegalOr) {
        LegalOr = false;
        return false;
      }
      if (!SubChanged) {
        todo.emplace_back(pair);
        continue;
      }
    }

    if (todo.size() == 0)
      return false;

    // TODO detect recursive merge and simplify
    bool changed = false;
    for (auto &pair : todo) {
      changed |= checkedOrIn(pair.first, pair.second, PointerIntSame, LegalOr);
    }
    return changed;
  }

  /// Set this to the logical or of itself and RHS, returning whether this value
  /// changed Setting `PointerIntSame` considers pointers and integers as
  /// equivalent This function will error if doing an illegal Operation
  bool orIn(const TypeTree &RHS, bool PointerIntSame) {
    bool Legal = true;
    bool Result = checkedOrIn(RHS, PointerIntSame, Legal);
    if (!Legal) {
      llvm::errs() << "Illegal orIn: " << str() << " right: " << RHS.str()
                   << " PointerIntSame=" << PointerIntSame << "\n";
      assert(0 && "Performed illegal ConcreteType::orIn");
      llvm_unreachable("Performed illegal ConcreteType::orIn");
    }
    return Result;
  }

  /// Set this to the logical or of itself and RHS, returning whether this value
  /// changed Setting `PointerIntSame` considers pointers and integers as
  /// equivalent This function will error if doing an illegal Operation
  bool orIn(const std::vector<int> Seq, ConcreteType CT, bool PointerIntSame) {
    bool Legal = true;
    bool Result = checkedOrIn(Seq, CT, PointerIntSame, Legal);
    if (!Legal) {
      llvm::errs() << "Illegal orIn: " << str() << " right: " << to_string(Seq)
                   << " CT: " << CT.str()
                   << " PointerIntSame=" << PointerIntSame << "\n";
      assert(0 && "Performed illegal ConcreteType::orIn");
      llvm_unreachable("Performed illegal ConcreteType::orIn");
    }
    return Result;
  }

  /// Set this to the logical or of itself and RHS, returning whether this value
  /// changed This assumes that pointers and integers are distinct This function
  /// will error if doing an illegal Operation
  bool operator|=(const TypeTree &RHS) {
    return orIn(RHS, /*PointerIntSame*/ false);
  }

  /// Set this to the logical and of itself and RHS, returning whether this
  /// value changed If this and RHS are incompatible at an index, the result
  /// will be BaseType::Unknown
  bool andIn(const TypeTree &RHS) {
    bool changed = false;

    for (auto &pair : llvm::make_early_inc_range(mapping)) {
      ConcreteType other = BaseType::Unknown;
      auto fd = RHS.mapping.find(pair.first);
      if (fd != RHS.mapping.end()) {
        other = fd->second;
      }
      changed = (pair.second &= other);
      if (pair.second == BaseType::Unknown) {
        mapping.erase(pair.first);
      }
    }

    return changed;
  }

  /// Set this to the logical and of itself and RHS, returning whether this
  /// value changed If this and RHS are incompatible at an index, the result
  /// will be BaseType::Unknown
  bool operator&=(const TypeTree &RHS) { return andIn(RHS); }

  /// Set this to the logical `binop` of itself and RHS, using the Binop Op,
  /// returning true if this was changed.
  /// This function will error on an invalid type combination
  bool binopIn(bool &Legal, const TypeTree &RHS,
               llvm::BinaryOperator::BinaryOps Op) {
    bool changed = false;

    for (auto &pair : llvm::make_early_inc_range(mapping)) {
      // TODO propagate non-first level operands:
      // Special handling is necessary here because a pointer to an int
      // binop with something should not apply the binop rules to the
      // underlying data but instead a different rule
      if (pair.first.size() > 0) {
        mapping.erase(pair.first);
        continue;
      }

      ConcreteType CT(pair.second);
      ConcreteType RightCT(BaseType::Unknown);

      // Mutual mappings
      auto found = RHS.mapping.find(pair.first);
      if (found != RHS.mapping.end()) {
        RightCT = found->second;
      }
      bool SubLegal = true;
      changed |= CT.binopIn(SubLegal, RightCT, Op);
      if (!SubLegal) {
        Legal = false;
        return changed;
      }
      if (CT == BaseType::Unknown) {
        mapping.erase(pair.first);
      } else {
        pair.second = CT;
      }
    }

    // mapings just on the right
    for (auto &pair : RHS.mapping) {
      // TODO propagate non-first level operands:
      // Special handling is necessary here because a pointer to an int
      // binop with something should not apply the binop rules to the
      // underlying data but instead a different rule
      if (pair.first.size() > 0) {
        continue;
      }

      if (mapping.find(pair.first) == RHS.mapping.end()) {
        ConcreteType CT = BaseType::Unknown;
        bool SubLegal = true;
        changed |= CT.binopIn(SubLegal, pair.second, Op);
        if (!SubLegal) {
          Legal = false;
          return changed;
        }
        if (CT != BaseType::Unknown) {
          mapping.insert(std::make_pair(pair.first, CT));
        }
      }
    }

    return changed;
  }

  /// Returns a string representation of this TypeTree
  std::string str() const {
    std::string out = "{";
    bool first = true;
    for (auto &pair : mapping) {
      if (!first) {
        out += ", ";
      }
      out += "[";
      for (unsigned i = 0; i < pair.first.size(); ++i) {
        if (i != 0)
          out += ",";
        out += std::to_string(pair.first[i]);
      }
      out += "]:" + pair.second.str();
      first = false;
    }
    out += "}";
    return out;
  }

  llvm::MDNode *toMD(llvm::LLVMContext &ctx) {
    llvm::SmallVector<llvm::Metadata *, 1> subMD;
    std::map<int, TypeTree> todo;
    ConcreteType base(BaseType::Unknown);
    for (auto &pair : mapping) {
      if (pair.first.size() == 0) {
        base = pair.second;
        continue;
      }
      auto next(pair.first);
      next.erase(next.begin());
      todo[pair.first[0]].mapping.insert(std::make_pair(next, pair.second));
    }
    subMD.push_back(llvm::MDString::get(ctx, base.str()));
    for (auto pair : todo) {
      subMD.push_back(llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), pair.first)));
      subMD.push_back(pair.second.toMD(ctx));
    }
    return llvm::MDNode::get(ctx, subMD);
  };

  void insertFromMD(llvm::MDNode *md, const std::vector<int> &prev = {}) {
    ConcreteType base(
        llvm::cast<llvm::MDString>(md->getOperand(0))->getString(),
        md->getContext());
    if (base != BaseType::Unknown)
      mapping.insert(std::make_pair(prev, base));
    for (size_t i = 1; i < md->getNumOperands(); i += 2) {
      auto off = llvm::cast<llvm::ConstantInt>(
                     llvm::cast<llvm::ConstantAsMetadata>(md->getOperand(i))
                         ->getValue())
                     ->getSExtValue();
      auto next(prev);
      next.push_back((int)off);
      insertFromMD(llvm::cast<llvm::MDNode>(md->getOperand(i + 1)), next);
    }
  }

  static TypeTree fromMD(llvm::MDNode *md) {
    TypeTree ret;
    std::vector<int> off;
    ret.insertFromMD(md, off);
    return ret;
  }
};

#endif
