//===- TypeIndex.h - Sets of offsets in Type Analysis type trees ----------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares TypeIndex, the set of byte offsets that one level of a
// type tree path names: a single offset, every offset from zero (written -1),
// or strided runs of offsets with an extent.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_TYPE_ANALYSIS_TYPE_INDEX_H
#define ENZYME_TYPE_ANALYSIS_TYPE_INDEX_H 1

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

/// The byte offsets `Start + j_0 * Dims[0].Stride + ... + j_n * Dims[n].Stride`
/// for `0 <= j_i < Dims[i].Count`, outermost dimension first. Dimensions do
/// not overlap: `Dims[i].Stride >= Dims[i+1].Stride * Dims[i+1].Count`.
///
/// A dimension of stride 1 is a run of bytes, each of which lies in a value of
/// the type of the entry. A point is a set without dimensions. An open set
/// ends at an unknown offset no later than its dimensions say; it holds only
/// within the function whose argument types introduced it.
struct TypeIndex {
  static constexpr int64_t Infinite = -1;

  struct Dim {
    int64_t Stride;
    int64_t Count; // >= 1, or Infinite
    bool operator==(const Dim &RHS) const {
      return Stride == RHS.Stride && Count == RHS.Count;
    }
    bool operator<(const Dim &RHS) const {
      return std::tie(Stride, Count) < std::tie(RHS.Stride, RHS.Count);
    }
  };

  int64_t Start = 0;
  llvm::SmallVector<Dim, 1> Dims;
  bool Open = false;

  /// -1 is every offset from zero, any other value that one offset.
  TypeIndex(int Off = 0) {
    if (Off == -1)
      *this = run(0, Infinite);
    else
      Start = Off;
  }

  /// The run of `Bytes` bytes (or Infinite) from `Start`.
  static TypeIndex run(int64_t Start, int64_t Bytes, bool Open = false) {
    TypeIndex Result;
    Result.Start = Start;
    Result.Open = Open;
    if (Bytes == 1 && !Open)
      return Result;
    assert(Bytes == Infinite || Bytes > 0);
    Result.Dims.push_back({1, Bytes});
    return Result;
  }

  bool isAll() const {
    return Start == 0 && !Open && Dims.size() == 1 && Dims[0].Stride == 1 &&
           Dims[0].Count == Infinite;
  }
  bool isPoint() const { return Dims.empty(); }
  /// Whether this is a point or every offset, the two forms that a plain
  /// integer offset can name.
  bool isSimple() const { return isPoint() || isAll(); }
  bool isFinite() const {
    return llvm::all_of(Dims, [](const Dim &D) { return D.Count != Infinite; });
  }

  /// The value that a plain integer offset would have: -1 for every offset,
  /// otherwise the first offset of the set.
  int key() const { return isAll() ? -1 : (int)Start; }

  /// One past the last offset, or INT64_MAX if the set has no end.
  int64_t end() const {
    int64_t Last = Start;
    for (const Dim &D : Dims) {
      if (D.Count == Infinite)
        return std::numeric_limits<int64_t>::max();
      Last += D.Stride * (D.Count - 1);
    }
    return Last + 1;
  }

  /// The number of offsets, or INT64_MAX if the set has no end.
  int64_t size() const {
    int64_t N = 1;
    for (const Dim &D : Dims) {
      if (D.Count == Infinite ||
          N > std::numeric_limits<int64_t>::max() / D.Count)
        return std::numeric_limits<int64_t>::max();
      N *= D.Count;
    }
    return N;
  }

  bool contains(int64_t Off) const {
    if (Off < Start)
      return false;
    int64_t Rem = Off - Start;
    for (const Dim &D : Dims) {
      int64_t Q = Rem / D.Stride;
      if (D.Count != Infinite && Q >= D.Count)
        return false;
      Rem -= Q * D.Stride;
    }
    return Rem == 0;
  }

  /// Call `F` on every offset of a finite set, in increasing order.
  template <typename FnT> void forEach(FnT &&F) const {
    assert(isFinite());
    forEachFrom(0, Start, F);
  }

  /// Whether every offset of `RHS` is in this set.
  bool includes(const TypeIndex &RHS) const {
    if (*this == RHS)
      return true;
    if (RHS.isPoint())
      return contains(RHS.Start);
    if (!contains(RHS.Start))
      return false;
    if (end() < RHS.end())
      return false;
    // A run of bytes holds everything between its ends.
    if (Dims.size() == 1 && Dims[0].Stride == 1)
      return true;
    if (RHS.isFinite() && RHS.size() <= kEnumerationLimit) {
      bool All = true;
      RHS.forEach([&](int64_t Off) { All &= contains(Off); });
      return All;
    }
    if (Dims.size() == 1 && RHS.Dims.size() == 1)
      return RHS.Dims[0].Stride % Dims[0].Stride == 0;
    return false;
  }

  /// Whether this set and `RHS` may share an offset. Large strided sets of
  /// different shapes are taken to overlap.
  bool overlaps(const TypeIndex &RHS) const {
    if (RHS.isPoint())
      return contains(RHS.Start);
    if (isPoint())
      return RHS.contains(Start);
    if (std::max(Start, RHS.Start) >= std::min(end(), RHS.end()))
      return false;
    const TypeIndex &Small = size() <= RHS.size() ? *this : RHS;
    const TypeIndex &Large = size() <= RHS.size() ? RHS : *this;
    if (Large.Dims.size() == 1 && Large.Dims[0].Stride == 1)
      return true;
    if (Small.isFinite() && Small.size() <= kEnumerationLimit) {
      bool Any = false;
      Small.forEach([&](int64_t Off) { Any |= Large.contains(Off); });
      return Any;
    }
    if (Dims.size() == 1 && RHS.Dims.size() == 1) {
      int64_t G = std::gcd(Dims[0].Stride, RHS.Dims[0].Stride);
      return (RHS.Start - Start) % G == 0;
    }
    return true;
  }

  /// Append to `Out` the offsets of this set in [Lo, Hi), moved by `Delta`.
  /// `Hi` is INT64_MAX for no upper bound.
  void clip(int64_t Lo, int64_t Hi, int64_t Delta,
            llvm::SmallVectorImpl<TypeIndex> &Out) const {
    clipFrom(0, Start, Lo, Hi, Delta, Out);
  }

  bool operator==(const TypeIndex &RHS) const {
    return Start == RHS.Start && Open == RHS.Open && Dims == RHS.Dims;
  }
  bool operator!=(const TypeIndex &RHS) const { return !(*this == RHS); }

  /// Every offset sorts first, as -1 did, then by first offset; a point
  /// before the runs starting at it.
  bool operator<(const TypeIndex &RHS) const {
    bool A = isAll(), B = RHS.isAll();
    if (A != B)
      return A;
    if (Start != RHS.Start)
      return Start < RHS.Start;
    if (Dims.size() != RHS.Dims.size())
      return Dims.size() < RHS.Dims.size();
    if (Dims != RHS.Dims)
      return std::lexicographical_compare(Dims.begin(), Dims.end(),
                                          RHS.Dims.begin(), RHS.Dims.end());
    return Open < RHS.Open;
  }

  /// `Start` followed by `+Stride*Count` for each dimension (`inf` for no
  /// end), and `?` if open: `8`, `-1`, `0+1*8000`, `16+24*100+1*8`, `0+1*?`.
  std::string str() const {
    if (isAll())
      return "-1";
    std::string Out = std::to_string(Start);
    for (const Dim &D : Dims) {
      Out += "+" + std::to_string(D.Stride) + "*";
      if (D.Count != Infinite)
        Out += std::to_string(D.Count);
      else
        Out += Open ? "?" : "inf";
    }
    if (Open && isFinite())
      Out += "?";
    return Out;
  }

  /// Parse the form `str` prints from the front of `S`.
  static bool consume(llvm::StringRef &S, TypeIndex &Result) {
    int64_t First;
    if (S.consumeInteger(10, First))
      return false;
    if (First == -1) {
      Result = TypeIndex(-1);
      return true;
    }
    Result = TypeIndex((int)First);
    while (S.consume_front("+")) {
      Dim D;
      if (S.consumeInteger(10, D.Stride) || D.Stride <= 0 ||
          !S.consume_front("*"))
        return false;
      if (S.consume_front("inf")) {
        D.Count = Infinite;
      } else if (S.consume_front("?")) {
        D.Count = Infinite;
        Result.Open = true;
      } else if (S.consumeInteger(10, D.Count) || D.Count <= 0) {
        return false;
      }
      Result.Dims.push_back(D);
    }
    if (S.consume_front("?"))
      Result.Open = true;
    return true;
  }

private:
  static constexpr int64_t kEnumerationLimit = 4096;

  template <typename FnT>
  void forEachFrom(size_t I, int64_t Base, FnT &F) const {
    if (I == Dims.size()) {
      F(Base);
      return;
    }
    for (int64_t J = 0; J < Dims[I].Count; ++J)
      forEachFrom(I + 1, Base + J * Dims[I].Stride, F);
  }

  /// The extent of one element of dimension I (the bytes its inner
  /// dimensions span).
  int64_t innerExtent(size_t I) const {
    int64_t Ext = 1;
    for (size_t K = I + 1; K < Dims.size(); ++K)
      Ext += Dims[K].Stride * (Dims[K].Count - 1);
    return Ext;
  }

  TypeIndex suffix(size_t I, int64_t Base) const {
    TypeIndex Result;
    Result.Start = Base;
    Result.Open = Open;
    Result.Dims.append(Dims.begin() + I, Dims.end());
    return Result;
  }

  void clipFrom(size_t I, int64_t Base, int64_t Lo, int64_t Hi, int64_t Delta,
                llvm::SmallVectorImpl<TypeIndex> &Out) const {
    TypeIndex Rest = suffix(I, Base);
    int64_t RestEnd = Rest.end();
    if (Base >= Hi || RestEnd <= Lo)
      return;
    if (Base >= Lo && RestEnd <= Hi) {
      Rest.Start += Delta;
      Out.push_back(Rest);
      return;
    }
    if (I == Dims.size()) {
      if (Base >= Lo && Base < Hi)
        Out.push_back(TypeIndex((int)(Base + Delta)));
      return;
    }
    const Dim &D = Dims[I];
    int64_t Ext = innerExtent(I);
    // Elements [J0, J1) intersect the window, [F0, F1) lie inside it.
    int64_t J0 = Lo <= Base ? 0 : (Lo - Base - Ext + D.Stride) / D.Stride;
    if (J0 < 0)
      J0 = 0;
    int64_t F0 = Lo <= Base ? 0 : (Lo - Base + D.Stride - 1) / D.Stride;
    int64_t Limit =
        D.Count == Infinite ? std::numeric_limits<int64_t>::max() : D.Count;
    int64_t J1 = Limit, F1 = Limit;
    if (Hi != std::numeric_limits<int64_t>::max()) {
      J1 = std::min(Limit, (Hi - Base + D.Stride - 1) / D.Stride);
      F1 = Hi - Base - Ext < 0
               ? 0
               : std::min(Limit, (Hi - Base - Ext) / D.Stride + 1);
    }
    if (F1 < F0)
      F1 = F0;
    auto Partial = [&](int64_t J) {
      clipFrom(I + 1, Base + J * D.Stride, Lo, Hi, Delta, Out);
    };
    for (int64_t J = J0; J < std::min(F0, J1); ++J)
      Partial(J);
    if (F0 < F1) {
      TypeIndex Whole = suffix(I, Base + F0 * D.Stride + Delta);
      Whole.Dims[0].Count =
          F1 == std::numeric_limits<int64_t>::max() ? Infinite : F1 - F0;
      if (Whole.Dims[0].Count == 1)
        Whole.Dims.erase(Whole.Dims.begin());
      if (Whole.Dims.empty() && !Whole.Open)
        Out.push_back(TypeIndex((int)Whole.Start));
      else
        Out.push_back(Whole);
    }
    for (int64_t J = std::max(F1, F0); J < J1; ++J)
      Partial(J);
  }
};

/// The path of a type tree entry as plain offsets (-1 for every offset).
static inline std::vector<TypeIndex> toTypePath(const std::vector<int> &Seq) {
  return std::vector<TypeIndex>(Seq.begin(), Seq.end());
}

/// Helper function to print a type tree path to a string
static inline std::string to_string(const std::vector<TypeIndex> &x) {
  std::string out = "[";
  for (unsigned i = 0; i < x.size(); ++i) {
    if (i != 0)
      out += ",";
    out += x[i].str();
  }
  out += "]";
  return out;
}

#endif
