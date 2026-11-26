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
#include "TypeIndex.h"

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

template <typename SubTypeT> struct TypeTreeTraits;

template <> struct TypeTreeTraits<llvm::Type *> {
  using Origin = llvm::Instruction *;

  static unsigned maxDepth() { return EnzymeMaxTypeDepth; }

  static int maxOffset() { return MaxTypeOffset; }

  template <typename TreeT>
  static void depthLimitExceeded(const TreeT &Tree, Origin Orig,
                                 const llvm::Twine &Detail) {
    if (!EnzymeTypeWarning)
      return;
    if (CustomErrorHandler) {
      CustomErrorHandler("TypeAnalysisDepthLimit", wrap(Orig),
                         ErrorType::TypeDepthExceeded, &Tree, nullptr, nullptr);
    } else if (Orig) {
      EmitWarning("TypeAnalysisDepthLimit", *Orig, *Orig,
                  " not handling more than ", EnzymeMaxTypeDepth,
                  " pointer lookups deep dt: ", Tree.str(), Detail.str());
    } else {
      llvm::errs() << "not handling more than " << EnzymeMaxTypeDepth
                   << " pointer lookups deep dt:" << Tree.str() << Detail
                   << "\n";
    }
  }
};

template <typename SubTypeT> class GenericTypeTree;

using TypeTree = GenericTypeTree<llvm::Type *>;

typedef std::shared_ptr<const TypeTree> TypeResult;
typedef std::vector<TypeIndex> TypePath;
typedef std::map<const TypePath, ConcreteType> ConcreteTypeMapType;
typedef std::map<const std::vector<int>, const TypeResult> TypeTreeMapType;

/// Class representing the underlying types of values as
/// sequences of offsets to a ConcreteType
///
/// The class is templated over the handle of the floating point type, like
/// GenericConcreteType, so that it is shared between LLVM and MLIR
template <typename SubTypeT>
class GenericTypeTree
    : public std::enable_shared_from_this<GenericTypeTree<SubTypeT>> {
  using Traits = TypeTreeTraits<SubTypeT>;

public:
  using ConcreteType = GenericConcreteType<SubTypeT>;
  using Path = std::vector<TypeIndex>;
  using ConcreteTypeMapType = std::map<const Path, ConcreteType>;

private:
  // mapping of known indices to type if one exists
  ConcreteTypeMapType mapping;
  std::vector<int> minIndices;

public:
  GenericTypeTree() {}
  GenericTypeTree(ConcreteType dat) {
    if (dat != ConcreteType(BaseType::Unknown)) {
      mapping.insert(std::pair<const Path, ConcreteType>({}, dat));
    }
  }

  template <typename ContextT>
  static GenericTypeTree parse(llvm::StringRef str, ContextT &ctx) {
    using namespace llvm;
    assert(str[0] == '{');
    str = str.substr(1);

    GenericTypeTree Result;
    while (true) {
      while (str[0] == ' ')
        str = str.substr(1);
      if (str[0] == '}')
        break;

      assert(str[0] == '[');
      str = str.substr(1);

      Path idxs;
      while (true) {
        while (str[0] == ' ')
          str = str.substr(1);
        if (str[0] == ']') {
          str = str.substr(1);
          break;
        }

        TypeIndex idx;
        bool parsed = TypeIndex::consume(str, idx);
        (void)parsed;
        assert(parsed);
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
      Result.noteMinIndices(idxs);

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
  /// or Unknown if none exists. -1 in `Seq` asks for the type at every
  /// offset of that level.
  ConcreteType operator[](const std::vector<int> &Seq) const {
    ConcreteType Found = lookupSimple(Seq);
    if (Found != BaseType::Unknown || !hasRuns())
      return Found;
    return lookupPath(toTypePath(Seq));
  }

  /// Lookup the ConcreteType of an entry whose offset sets include those of
  /// `Seq`, or Unknown if none exists.
  ConcreteType lookupPath(const Path &Seq) const {
    if (llvm::all_of(Seq,
                     [](const TypeIndex &idx) { return idx.isSimple(); })) {
      std::vector<int> Simple;
      for (const TypeIndex &idx : Seq)
        Simple.push_back(idx.key());
      ConcreteType Found = lookupSimple(Simple);
      if (Found != BaseType::Unknown || !hasRuns())
        return Found;
    } else {
      auto Found0 = mapping.find(Seq);
      if (Found0 != mapping.end())
        return Found0->second;
    }
    for (const auto &pair : mapping) {
      if (pair.first.size() != Seq.size())
        continue;
      bool Includes = true;
      for (size_t i = 0, Len = Seq.size(); i < Len && Includes; ++i)
        Includes = pair.first[i].includes(Seq[i]);
      if (Includes)
        return pair.second;
    }
    return BaseType::Unknown;
  }

  // Return true if this type tree is fully known (i.e. there
  // is no more information which could be added).
  bool IsFullyDetermined() const {
    std::vector<int> offsets = {-1};
    while (1) {
      auto found = mapping.find(toTypePath(offsets));
      if (found == mapping.end())
        return false;
      if (found->second != BaseType::Pointer)
        return true;
      offsets.push_back(-1);
    }
  }

  void remove(const std::vector<int> &Seq) { mapping.erase(toTypePath(Seq)); }
  void removePath(const Path &Seq) { mapping.erase(Seq); }

  /// Return if changed
  bool insert(const std::vector<int> &Seq, ConcreteType CT,
              bool PointerIntSame = false) {
    return insertPath(toTypePath(Seq), CT, PointerIntSame);
  }

  /// Return if changed
  bool insertPath(const Path &Seq, ConcreteType CT,
                  bool PointerIntSame = false) {
    size_t SeqSize = Seq.size();
    if (SeqSize > Traits::maxDepth()) {
      Traits::depthLimitExceeded(
          *this, nullptr, " adding v: " + to_string(Seq) + ": " + CT.str());
      return false;
    }
    if (SeqSize == 0) {
      mapping.insert(std::pair<const Path, ConcreteType>(Seq, CT));
      return true;
    }

    // check types at lower pointer offsets are either pointer or
    // anything. Don't insert into an anything
    {
      Path tmp(Seq);
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
          Generality G = compare(Seq, pair.first);
          if (G.Disjoint)
            continue;

          if (G.OldMoreGeneral) {
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
          } else if (G.NewMoreGeneral) {
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
          } else if (G.RunsOverlap &&
                     !compatibleOverlap(CT, pair.second, PointerIntSame)) {
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
      if (minIndices[i] > Seq[i].key()) {
        if (minIndices[i] > Traits::maxOffset())
          possibleDeletion = true;
        minIndices[i] = Seq[i].key();
      }
    }

    if (minIndices.size() < SeqSize) {
      for (size_t i = minIndices.size(), end = SeqSize; i < end; ++i) {
        minIndices.push_back(Seq[i].key());
      }
    }

    if (possibleDeletion) {
      for (const auto &pair : llvm::make_early_inc_range(mapping)) {
        size_t i = 0;
        bool mustKeep = false;
        bool considerErase = false;
        for (const TypeIndex &idx : pair.first) {
          int val = idx.key();
          if (val > Traits::maxOffset() && idx.isSimple()) {
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

    if (!keepBeyondMaxOffset(Seq, minIndices, mapping))
      return changed;
    mapping.insert(std::pair<const Path, ConcreteType>(Seq, CT));
    return true;
  }

  /// Drop the entries with an open offset set, which hold only within the
  /// function whose argument types introduced them, before this tree is
  /// applied to values of another function.
  GenericTypeTree DropOpen() const {
    GenericTypeTree Result;
    for (const auto &pair : mapping) {
      if (llvm::any_of(pair.first,
                       [](const TypeIndex &idx) { return idx.Open; }))
        continue;
      Result.mapping.insert(pair);
      Result.noteMinIndices(pair.first);
    }
    return Result;
  }

private:
  /// How the offset sets of two paths of the same length relate.
  struct Generality {
    /// Some level has no offset in common.
    bool Disjoint = false;
    /// Every level of the new path includes the old one.
    bool NewMoreGeneral = true;
    /// Every level of the old path includes the new one.
    bool OldMoreGeneral = true;
    /// The paths share offsets, neither includes the other, and one of the
    /// levels that differ is a run of offsets.
    bool RunsOverlap = false;
  };

  static Generality compare(const Path &New, const Path &Old) {
    Generality G;
    bool AnyRun = false;
    for (size_t i = 0, Len = New.size(); i < Len; i++) {
      if (New[i] == Old[i])
        continue;
      bool NewIncludes = New[i].includes(Old[i]);
      bool OldIncludes = Old[i].includes(New[i]);
      if (!NewIncludes && !OldIncludes && !New[i].overlaps(Old[i])) {
        G.Disjoint = true;
        G.NewMoreGeneral = G.OldMoreGeneral = false;
        return G;
      }
      AnyRun |= !New[i].isSimple() || !Old[i].isSimple();
      G.NewMoreGeneral &= NewIncludes;
      G.OldMoreGeneral &= OldIncludes;
    }
    // Two simple paths that cross, like [0,-1] and [-1,0], are not compared,
    // as before runs existed.
    G.RunsOverlap = !G.NewMoreGeneral && !G.OldMoreGeneral && AnyRun;
    return G;
  }

  /// Whether two types may hold at the same offsets of overlapping entries.
  static bool compatibleOverlap(ConcreteType New, ConcreteType Old,
                                bool PointerIntSame) {
    if (New == Old || New == BaseType::Anything || Old == BaseType::Anything)
      return true;
    return PointerIntSame &&
           ((New == BaseType::Pointer && Old == BaseType::Integer) ||
            (New == BaseType::Integer && Old == BaseType::Pointer));
  }

  /// Whether any entry has an offset set other than a point or -1.
  bool hasRuns() const {
    for (const auto &pair : mapping)
      for (const TypeIndex &idx : pair.first)
        if (!idx.isSimple())
          return true;
    return false;
  }

  /// Whether an entry with path `Seq` stays within the limits that make type
  /// analysis terminate: a point beyond the maximum offset only if it is the
  /// smallest offset at its depth, and runs beyond it only up to as many as
  /// the maximum offset (constant-offset pointer cycles shift a run further
  /// on each iteration).
  static bool keepBeyondMaxOffset(const Path &Seq,
                                  const std::vector<int> &minIdx,
                                  const ConcreteTypeMapType &entries) {
    bool considerErase = false;
    bool farRun = false;
    for (size_t i = 0, Len = Seq.size(); i < Len; i++) {
      int val = Seq[i].key();
      if (val <= Traits::maxOffset())
        continue;
      if (i < minIdx.size() && val == minIdx[i])
        return true;
      if (Seq[i].isSimple())
        considerErase = true;
      else
        farRun = true;
    }
    if (considerErase)
      return false;
    if (!farRun)
      return true;
    int farRuns = 0;
    for (const auto &pair : entries)
      for (const TypeIndex &idx : pair.first)
        if (!idx.isSimple() && idx.key() > Traits::maxOffset()) {
          ++farRuns;
          break;
        }
    return farRuns < Traits::maxOffset();
  }

  /// Lower minIndices to the offsets of `Seq`.
  void noteMinIndices(const Path &Seq) {
    for (size_t i = 0, Len = Seq.size(); i < Len; ++i) {
      if (i == minIndices.size())
        minIndices.push_back(Seq[i].key());
      else if (Seq[i].key() < minIndices[i])
        minIndices[i] = Seq[i].key();
    }
  }

  /// The lookup of plain offsets among the entries with plain offsets.
  ConcreteType lookupSimple(const std::vector<int> &Seq) const {
    auto Found0 = mapping.find(toTypePath(Seq));
    if (Found0 != mapping.end())
      return Found0->second;
    size_t Len = Seq.size();
    if (Len == 0)
      return BaseType::Unknown;

    std::vector<Path> todo[2];
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

public:
  /// How this TypeTree compares with another
  bool operator<(const GenericTypeTree &vd) const {
    return mapping < vd.mapping;
  }

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
  GenericTypeTree JustInt() const {
    GenericTypeTree vd;
    for (auto &pair : mapping) {
      if (pair.second == BaseType::Integer) {
        vd.insertPath(pair.first, pair.second);
      }
    }

    return vd;
  }

  /// Prepend an offset to all mappings
  GenericTypeTree Only(int Off, typename Traits::Origin orig) const {
    return OnlyIndex(TypeIndex(Off), orig);
  }

  /// Prepend an offset set to all mappings
  GenericTypeTree OnlyIndex(const TypeIndex &Off,
                            typename Traits::Origin orig) const {
    GenericTypeTree Result;
    Result.minIndices.reserve(1 + minIndices.size());
    Result.minIndices.push_back(Off.key());
    for (auto midx : minIndices)
      Result.minIndices.push_back(midx);

    if (Result.minIndices.size() > Traits::maxDepth()) {
      Result.minIndices.pop_back();
      Traits::depthLimitExceeded(*this, orig, " only(" + Off.str() + ")");
    }

    for (const auto &pair : mapping) {
      if (pair.first.size() == Traits::maxDepth())
        continue;
      Path Vec;
      Vec.reserve(pair.first.size() + 1);
      Vec.push_back(Off);
      for (const auto &Val : pair.first)
        Vec.push_back(Val);
      Result.mapping.insert(
          std::pair<const Path, ConcreteType>(Vec, pair.second));
    }
    return Result;
  }

  /// Peel off the outermost index at offset 0
  GenericTypeTree Data0() const {
    GenericTypeTree Result;

    for (const auto &pair : mapping) {
      if (pair.first.size() == 0) {
        llvm::errs() << str() << "\n";
      }
      assert(pair.first.size() != 0);

      if (pair.first[0].isAll()) {
        Path next(pair.first.begin() + 1, pair.first.end());
        Result.mapping.insert(
            std::pair<const Path, ConcreteType>(next, pair.second));
        Result.noteMinIndices(next);
      }
    }
    for (const auto &pair : mapping) {
      if (!pair.first[0].isAll() && pair.first[0].contains(0)) {
        Path next(pair.first.begin() + 1, pair.first.end());
        // We do insertion like this to force an error
        // on the orIn operation if there is an incompatible
        // merge. The insert operation does not error.
        Result.orInPath(next, pair.second);
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
  GenericTypeTree Clear(size_t start, size_t end, size_t len) const {
    GenericTypeTree Result;

    // Note that below do insertion with the orIn operator
    // to force an error if there is an incompatible
    // merge. The insert operation does not error.

    for (const auto &pair : mapping) {
      assert(pair.first.size() != 0);

      if (pair.first[0].isAll()) {
        // For "all index" calculations, explicitly
        // add mappings for regions in range
        auto next = pair.first;
        for (size_t i = 0; i < start; ++i) {
          next[0] = i;
          Result.orInPath(next, pair.second);
        }
        for (size_t i = end; i < len; ++i) {
          next[0] = i;
          Result.orInPath(next, pair.second);
        }
      } else if (pair.first[0].isPoint()) {
        // Otherwise simply check that the given offset is in range
        size_t off = pair.first[0].Start;
        if (off < start || (off >= end && off < len))
          Result.insertPath(pair.first, pair.second);
      } else {
        llvm::SmallVector<TypeIndex, 2> kept;
        pair.first[0].clip(0, start, 0, kept);
        pair.first[0].clip(end, len, 0, kept);
        auto next = pair.first;
        for (const TypeIndex &idx : kept) {
          next[0] = idx;
          Result.orInPath(next, pair.second);
        }
      }
    }

    // TODO canonicalize this
    return Result;
  }

private:
  /// The size of the values whose start offsets an entry of type `dt` with
  /// `inner` further offsets lists, for combining them into a -1.
  static size_t chunkSize(const Path &inner, ConcreteType dt,
                          const llvm::DataLayout &dl) {
    if (inner.size() > 0)
      return dl.getPointerSizeInBits() / 8;
    if (auto flt = dt.isFloat())
      return dl.getTypeSizeInBits(flt) / 8;
    if (dt == BaseType::Pointer)
      return dl.getPointerSizeInBits() / 8;
    return 1;
  }

  /// Whether the offset sets `set` cover every `chunk`-th offset of [0, len).
  static bool covers(const std::vector<TypeIndex> &set, size_t len,
                     size_t chunk) {
    for (size_t i = 0; i < len; i += chunk)
      if (!llvm::any_of(set,
                        [&](const TypeIndex &idx) { return idx.contains(i); }))
        return false;
    return true;
  }

public:
  /// Select all submappings whose first index is in range [0, len) and remove
  /// the first index. This is the inverse of the `Only` operation
  GenericTypeTree Lookup(size_t len, const llvm::DataLayout &dl) const {

    // Map of indices[1:] => ( End => possible Index[0] )
    std::map<Path, std::map<ConcreteType, std::vector<TypeIndex>>> staging;

    for (const auto &pair : mapping) {
      assert(pair.first.size() != 0);

      // Pointer is at offset 0 from this object
      if (!pair.first[0].contains(0))
        continue;

      if (pair.first.size() == 1) {
        assert(pair.second == ConcreteType(BaseType::Pointer) ||
               pair.second == ConcreteType(BaseType::Anything));
        continue;
      }

      Path next(pair.first.begin() + 2, pair.first.end());
      auto &set = staging[next][pair.second];
      if (pair.first[1].isAll()) {
        set.push_back(pair.first[1]);
      } else {
        llvm::SmallVector<TypeIndex, 2> kept;
        pair.first[1].clip(0, len, 0, kept);
        set.insert(set.end(), kept.begin(), kept.end());
      }
    }

    GenericTypeTree Result;
    for (auto &pair : staging) {
      auto &pnext = pair.first;
      for (auto &pair2 : pair.second) {
        auto dt = pair2.first;
        const auto &set = pair2.second;
        if (set.empty())
          continue;

        // See if we can canonicalize the outermost index into a -1
        bool legalCombine =
            llvm::any_of(set,
                         [](const TypeIndex &idx) { return idx.isAll(); }) ||
            covers(set, len, chunkSize(pnext, dt, dl));

        Path next;
        next.reserve(pnext.size() + 1);
        next.push_back(-1);
        for (auto v : pnext)
          next.push_back(v);

        if (legalCombine) {
          Result.insertPath(next, dt, /*intsAreLegalPointerSub*/ true);
        } else {
          for (const auto &e : set) {
            next[0] = e;
            Result.insertPath(next, dt);
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
      if (!pair.first[0].isAll()) {
        canonicalized = false;
        break;
      }
    }
    if (canonicalized)
      return;

    // Map of indices[1:] => ( End => possible Index[0] )
    std::map<const Path, std::map<ConcreteType, std::vector<TypeIndex>>>
        staging;

    for (const auto &pair : mapping) {

      Path next(pair.first.begin() + 1, pair.first.end());
      const TypeIndex &first = pair.first[0];
      if (first.isPoint()) {
        if ((size_t)first.Start >= len) {
          llvm::errs() << str() << "\n";
          llvm::errs() << " canonicalizing " << len << "\n";
          llvm::report_fatal_error("Canonicalization failed");
        }
        staging[next][pair.second].push_back(first);
      } else if (first.isAll()) {
        staging[next][pair.second].push_back(first);
      } else {
        llvm::SmallVector<TypeIndex, 2> kept;
        first.clip(0, len, 0, kept);
        auto &set = staging[next][pair.second];
        set.insert(set.end(), kept.begin(), kept.end());
      }
    }

    // TypeTree mappings which did not get combined
    std::map<const Path, ConcreteType> unCombinedToAdd;

    // TypeTree mappings which did get combined into an outer -1
    std::map<const Path, ConcreteType> combinedToAdd;

    for (const auto &pair : staging) {
      auto &pnext = pair.first;
      for (const auto &pair2 : pair.second) {
        auto dt = pair2.first;
        const auto &set = pair2.second;

        // See if we can canonicalize the outermost index into a -1
        bool legalCombine = !llvm::any_of(set, [](const TypeIndex &idx) {
          return idx.isAll();
        }) && covers(set, len, chunkSize(pnext, dt, dl));

        Path next;
        next.reserve(pnext.size() + 1);
        next.push_back(-1);
        for (auto v : pnext)
          next.push_back(v);

        if (legalCombine) {
          combinedToAdd.emplace(next, dt);
        } else {
          for (const auto &e : set) {
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
      insertPath(pair.first, pair.second);
    }

    return;
  }

  /// Keep only pointers (or anything's) to a repeated value (represented by -1)
  GenericTypeTree KeepMinusOne(bool &legal) const {
    GenericTypeTree dat;

    for (const auto &pair : mapping) {

      assert(pair.first.size() != 0);

      // Pointer is at offset 0 from this object
      if (!pair.first[0].contains(0))
        continue;

      if (pair.first.size() == 1) {
        if (pair.second == BaseType::Pointer ||
            pair.second == BaseType::Anything) {
          dat.insertPath(pair.first, pair.second);
          continue;
        }
        legal = false;
        break;
      }

      if (pair.first[1].isAll()) {
        dat.insertPath(pair.first, pair.second);
      }
    }

    return dat;
  }

  /// The types of what a pointer `Base + k * Stride` points to, for this
  /// pointer's tree and any `k` for which it stays within the runs of offsets
  /// of the data. A run of bytes from at most `Base` holds the `Stride` bytes
  /// from the new pointer; a run of elements of size `Stride` (with `Base` 0)
  /// holds one element.
  GenericTypeTree AtVariableOffset(int64_t Stride, int64_t Base) const {
    GenericTypeTree Result;
    for (const auto &pair : mapping) {
      if (pair.first.size() < 2 || !pair.first[0].contains(0))
        continue;
      const TypeIndex &Data = pair.first[1];
      if (Data.isSimple())
        continue;
      Path next(pair.first);
      if (Data.Dims.size() == 1 && Data.Dims[0].Stride == 1 &&
          Data.Start <= Base && Data.end() >= Base + Stride)
        next[1] = TypeIndex::run(0, Stride, Data.Open);
      else if (Base == 0 && Data.Dims[0].Stride == Stride &&
               Data.Start < Stride && Data.end() > Stride) {
        TypeIndex Element(Data);
        Element.Dims.erase(Element.Dims.begin());
        if (Element.Dims.empty())
          Element = TypeIndex((int)Data.Start);
        next[1] = Element;
      } else
        continue;
      Result.insertPath(next, pair.second);
    }
    return Result;
  }

  llvm::Type *allFloat(llvm::Value *val, const llvm::DataLayout &dl,
                       bool anythingIsFloat = false) const {
    return allFloat(val ? val->getType() : nullptr, dl, anythingIsFloat);
  }

  llvm::Type *allFloat(llvm::Type *type, const llvm::DataLayout &dl,
                       bool anythingIsFloat = false) const {
    auto dt = operator[]({-1});
    if (dt != BaseType::Anything && dt != BaseType::Unknown)
      return dt.isFloat();
    if (anythingIsFloat && dt == BaseType::Anything)
      return (llvm::Type *)0x123;

    if (!type || type->isTokenTy() || type->isVoidTy())
      return nullptr;

    size_t size = (dl.getTypeSizeInBits(type) + 7) / 8;

    auto m0 = GenericTypeTree::operator[]({0});

    llvm::Type *flt = m0.isFloat();
    if (!flt) {
      if (!(anythingIsFloat && m0 == BaseType::Anything)) {
        return nullptr;
      }
    }
    // If offset 0 is Anything (and anythingIsFloat), flt is still null here;
    // walk bytewise until a concrete float type is found.
    size_t chunk = flt ? dl.getTypeSizeInBits(flt) / 8 : 1;
    for (size_t i = chunk; i < size; i += chunk) {
      auto mx = GenericTypeTree::operator[]({(int)i});
      if (auto f2 = mx.isFloat()) {
        if (f2 != flt) {
          if (anythingIsFloat && !flt) {
            flt = f2;
            chunk = dl.getTypeSizeInBits(flt) / 8;
            continue;
          }
          return nullptr;
        }
      } else if (anythingIsFloat && mx == BaseType::Anything) {
        continue;
      } else
        return nullptr;
    }

    if (!flt && anythingIsFloat) {
      return (llvm::Type *)0x123;
    }
    return flt;
  }
  bool anyFloat(llvm::Value *val, const llvm::DataLayout &dl) const {
    auto dt = operator[]({-1});
    if (dt != BaseType::Anything && dt != BaseType::Unknown)
      return dt.isFloat();

    if (val->getType()->isTokenTy() || val->getType()->isVoidTy())
      return false;

    size_t ObjSize = (dl.getTypeSizeInBits(val->getType()) + 7) / 8;
    for (size_t i = 0; i < ObjSize;) {
      dt = operator[]({(int)i});
      if (dt == BaseType::Integer) {
        i++;
        continue;
      }
      if (dt == BaseType::Pointer) {
        i += dl.getPointerSize(0);
        continue;
      }
      return true;
    }
    return false;
  }
  bool anyPointer(llvm::Value *val, const llvm::DataLayout &dl) const {
    auto dt = operator[]({-1});
    if (dt != BaseType::Anything && dt != BaseType::Unknown)
      return dt == BaseType::Pointer;

    if (val->getType()->isTokenTy() || val->getType()->isVoidTy())
      return false;

    size_t ObjSize = (dl.getTypeSizeInBits(val->getType()) + 7) / 8;
    for (size_t i = 0; i < ObjSize;) {
      dt = operator[]({(int)i});
      if (dt == BaseType::Integer) {
        i++;
        continue;
      }
      if (auto FT = dt.isFloat()) {
        i += (dl.getTypeSizeInBits(FT) + 7) / 8;
        continue;
      }
      return true;
    }
    return false;
  }

  /// Replace mappings in the range in [offset, offset+maxSize] with those in
  // [addOffset, addOffset + maxSize]. In other words, select all mappings in
  // [offset, offset+maxSize] then add `addOffset`
  GenericTypeTree ShiftIndices(const llvm::DataLayout &dl, const int offset,
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
      return GenericTypeTree();

    GenericTypeTree Result;

    // The normal orIn / insert methods do collision checking, which is slow
    // (and presently O(n)). The shifted entries are collected first, then
    // inserted without checks, merging only entries with the same path: the
    // tree they come from has no conflicts, and only the expansion of a -1
    // into offsets can collide with a fixed offset (e.g. shifting
    // {[-1]:Integer, [1]:Anything} to two bytes gives [0]:Int, [1]:Int, which
    // merges with [1]:Anything).
    //
    // Since we're doing things manually, we also need to manually preserve TT
    // invariants. Specifically, TT limits all values to have offsets <
    // MAX_OFFSET, unless it is the smallest offset at that depth. (e.g. so we
    // can still hava  typetree {[123456]:Int}, even if limit is 100).
    const int64_t Hi = maxSize == -1 ? std::numeric_limits<int64_t>::max()
                                     : (int64_t)offset + maxSize;
    const int64_t Delta = (int64_t)addOffset - offset;

    // Shifted entries, and the -1's to expand into offsets.
    std::vector<std::pair<Path, ConcreteType>> shifted;
    std::vector<std::pair<Path, ConcreteType>> expand;

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

      const TypeIndex &first = pair.first[0];
      Path next(pair.first);
      if (first.isAll()) {
        if (maxSize == -1) {
          // Max size does not clip the next index; a -1 represents only
          // [0, inf), so from a nonzero addOffset it becomes a run.
          if (addOffset != 0)
            next[0] = TypeIndex::run(addOffset, TypeIndex::Infinite);
          shifted.emplace_back(next, pair.second);
        } else if ((int64_t)addOffset + maxSize - 1 > Traits::maxOffset()) {
          // Offsets beyond the maximum would be dropped.
          next[0] = TypeIndex::run(addOffset, maxSize);
          shifted.emplace_back(next, pair.second);
        } else {
          // This needs to become 0...maxSize handled separately as it is the
          // only insertion that could have collisions
          expand.emplace_back(next, pair.second);
        }
        continue;
      }

      llvm::SmallVector<TypeIndex, 2> pieces;
      first.clip(offset, Hi, Delta, pieces);
      for (const TypeIndex &idx : pieces) {
        next[0] = idx;
        shifted.emplace_back(next, pair.second);
      }
    }

    // First compute the minimum index at each depth to be kept.
    Result.minIndices.resize(minIndices.size(), INT_MAX);
    for (const auto &pair : shifted)
      for (size_t i = 0, sz = pair.first.size(); i < sz; i++)
        if (pair.first[i].key() < Result.minIndices[i])
          Result.minIndices[i] = pair.first[i].key();
    for (const auto &pair : expand) {
      // We're going to insert addOffset + 0...maxSize so the new minIndex
      // is addOffset
      if ((int)addOffset < Result.minIndices[0])
        Result.minIndices[0] = addOffset;
      for (size_t i = 1, sz = pair.first.size(); i < sz; i++)
        if (pair.first[i].key() < Result.minIndices[i])
          Result.minIndices[i] = pair.first[i].key();
    }

    // Max depth of actual inserted values
    size_t maxInsertedDepth = 0;

    for (const auto &pair : shifted) {
      // If after moving this would not merit being kept for being a min index
      // or being within the max type offset, skip it.
      int next0 = pair.first[0].key();
      if (next0 > Traits::maxOffset()) {
        bool minIndex = next0 == Result.minIndices[0];
        if (!minIndex)
          for (size_t i = 1; i < pair.first.size(); i++) {
            if (pair.first[i].key() == Result.minIndices[i]) {
              minIndex = true;
              break;
            }
          }
        if (!minIndex &&
            !keepBeyondMaxOffset(pair.first, Result.minIndices, Result.mapping))
          continue;
      }

      auto found = Result.mapping.find(pair.first);
      if (found != Result.mapping.end()) {
        ConcreteType prev(pair.second);
        if (prev.orIn(found->second, /*pointerIntSame*/ false))
          found->second = prev;
      } else {
        Result.mapping.emplace(pair.first, pair.second);
      }
      if (pair.first.size() > maxInsertedDepth)
        maxInsertedDepth = pair.first.size();
    }

    // Insert and expand the minus one
    for (const auto &pair : expand) {
      size_t chunk = 1;
      Path next(pair.first);
      auto op = lookupPath(Path{next[0]});
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
  GenericTypeTree PurgeAnything() const {
    GenericTypeTree Result;
    Result.minIndices.reserve(minIndices.size());
    for (const auto &pair : mapping) {
      if (pair.second == ConcreteType(BaseType::Anything))
        continue;
      Result.mapping.insert(pair);
      Result.noteMinIndices(pair.first);
    }
    return Result;
  }

  /// Replace -1 with 0
  GenericTypeTree ReplaceMinus() const {
    GenericTypeTree dat;
    for (const auto &pair : mapping) {
      if (pair.second == ConcreteType(BaseType::Anything))
        continue;
      Path nex = pair.first;
      for (auto &v : nex)
        if (!v.isPoint())
          v = TypeIndex((int)v.Start);
      dat.insertPath(nex, pair.second);
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
  GenericTypeTree JustAnything() const {
    GenericTypeTree dat;
    for (const auto &pair : mapping) {
      if (pair.second != ConcreteType(BaseType::Anything))
        continue;
      dat.insertPath(pair.first, pair.second);
    }
    return dat;
  }

  /// Chceck equality of two TypeTrees
  bool operator==(const GenericTypeTree &RHS) const {
    return mapping == RHS.mapping;
  }

  /// Set this to another TypeTree, returning if this was changed
  bool operator=(const GenericTypeTree &RHS) {
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
    return checkedOrInPath(toTypePath(Seq), RHS, PointerIntSame, LegalOr);
  }

  bool checkedOrInPath(const Path &Seq, ConcreteType RHS, bool PointerIntSame,
                       bool &LegalOr) {
    assert(RHS != BaseType::Unknown);
    ConcreteType CT = lookupPath(Seq);

    bool subchanged = CT.checkedOrIn(RHS, PointerIntSame, LegalOr);
    if (!subchanged)
      return false;
    if (!LegalOr)
      return subchanged;

    auto SeqSize = Seq.size();

    if (SeqSize > 0) {
      // check pointer abilities from before
      for (size_t i = 0; i < SeqSize; ++i) {
        Path tmp(Seq.begin(), Seq.end() - 1 - i);
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
            Generality G = compare(Seq, pair.first);
            if (G.Disjoint)
              continue;
            // Whether the the inserted val (e.g. [-1, 0] or [0, 0]) is at least
            // as general as the existing map val (e.g. [0, 0]).
            bool newMoreGeneralThanOld = G.NewMoreGeneral;
            // Whether the the existing val (e.g. [-1, 0] or [0, 0]) is at least
            // as general as the inserted map val (e.g. [0, 0]).
            bool oldMoreGeneralThanNew = G.OldMoreGeneral;

            if (G.RunsOverlap &&
                !compatibleOverlap(CT, pair.second, PointerIntSame)) {
              LegalOr = false;
              return false;
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

    return insertPath(Seq, CT);
  }

  bool orIn(const std::vector<int> &Seq, ConcreteType RHS,
            bool PointerIntSame = false) {
    return orInPath(toTypePath(Seq), RHS, PointerIntSame);
  }

  bool orInPath(const Path &Seq, ConcreteType RHS,
                bool PointerIntSame = false) {
    bool LegalOr = true;
    bool Result = checkedOrInPath(Seq, RHS, PointerIntSame, LegalOr);
    assert(LegalOr);
    return Result;
  }

  /// Set this to the logical or of itself and RHS, returning whether this value
  /// changed Setting `PointerIntSame` considers pointers and integers as
  /// equivalent If this is an illegal operation, `LegalOr` will be set to false
  bool checkedOrIn(const GenericTypeTree &RHS, bool PointerIntSame,
                   bool &LegalOr) {
    // TODO detect recursive merge and simplify

    bool changed = false;
    for (auto &pair : RHS.mapping) {
      changed |=
          checkedOrInPath(pair.first, pair.second, PointerIntSame, LegalOr);
    }
    return changed;
  }

  /// Set this to the logical or of itself and RHS, returning whether this value
  /// changed Setting `PointerIntSame` considers pointers and integers as
  /// equivalent This function will error if doing an illegal Operation
  bool orIn(const GenericTypeTree &RHS, bool PointerIntSame) {
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
  bool operator|=(const GenericTypeTree &RHS) {
    return orIn(RHS, /*PointerIntSame*/ false);
  }

  /// Set this to the logical and of itself and RHS, returning whether this
  /// value changed If this and RHS are incompatible at an index, the result
  /// will be BaseType::Unknown
  bool andIn(const GenericTypeTree &RHS) {
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
  bool operator&=(const GenericTypeTree &RHS) { return andIn(RHS); }

  /// Set this to the logical `binop` of itself and RHS, using the Binop Op,
  /// returning true if this was changed.
  /// This function will error on an invalid type combination
  bool binopIn(bool &Legal, const GenericTypeTree &RHS,
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
      out += to_string(pair.first) + ":" + pair.second.str();
      first = false;
    }
    out += "}";
    return out;
  }

  llvm::MDNode *toMD(llvm::LLVMContext &ctx) {
    llvm::SmallVector<llvm::Metadata *, 1> subMD;
    std::map<TypeIndex, GenericTypeTree> todo;
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
      subMD.push_back(indexToMD(ctx, pair.first));
      subMD.push_back(pair.second.toMD(ctx));
    }
    return llvm::MDNode::get(ctx, subMD);
  };

  /// A point or -1 is a signed i32 constant, any other offset set a tuple of
  /// i64 constants: the start, 1 if open (0 otherwise), and a stride and count
  /// per dimension (-1 for no end).
  static llvm::Metadata *indexToMD(llvm::LLVMContext &ctx,
                                   const TypeIndex &idx) {
    if (idx.isSimple())
      return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          llvm::IntegerType::get(ctx, 32), idx.key(), /*IsSigned*/ true));
    auto *I64 = llvm::IntegerType::get(ctx, 64);
    auto get = [&](int64_t v) -> llvm::Metadata * {
      return llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(I64, v, /*IsSigned*/ true));
    };
    llvm::SmallVector<llvm::Metadata *, 4> ops = {get(idx.Start),
                                                  get(idx.Open)};
    for (const TypeIndex::Dim &D : idx.Dims) {
      ops.push_back(get(D.Stride));
      ops.push_back(get(D.Count));
    }
    return llvm::MDTuple::get(ctx, ops);
  }

  static TypeIndex indexFromMD(llvm::Metadata *md) {
    auto value = [](llvm::Metadata *op) {
      return llvm::cast<llvm::ConstantInt>(
                 llvm::cast<llvm::ConstantAsMetadata>(op)->getValue())
          ->getSExtValue();
    };
    auto *tuple = llvm::dyn_cast<llvm::MDTuple>(md);
    if (!tuple)
      return TypeIndex((int)value(md));
    TypeIndex idx((int)value(tuple->getOperand(0)));
    idx.Open = value(tuple->getOperand(1)) != 0;
    for (unsigned i = 2; i + 1 < tuple->getNumOperands(); i += 2)
      idx.Dims.push_back(
          {value(tuple->getOperand(i)), value(tuple->getOperand(i + 1))});
    return idx;
  }

  void insertFromMD(llvm::MDNode *md, const Path &prev = {}) {
    ConcreteType base(
        llvm::cast<llvm::MDString>(md->getOperand(0))->getString(),
        md->getContext());
    if (base != BaseType::Unknown)
      mapping.insert(std::make_pair(prev, base));
    for (size_t i = 1; i < md->getNumOperands(); i += 2) {
      auto next(prev);
      next.push_back(indexFromMD(md->getOperand(i)));
      insertFromMD(llvm::cast<llvm::MDNode>(md->getOperand(i + 1)), next);
    }
  }

  static GenericTypeTree fromMD(llvm::MDNode *md) {
    GenericTypeTree ret;
    ret.insertFromMD(md);
    return ret;
  }
};

#endif
