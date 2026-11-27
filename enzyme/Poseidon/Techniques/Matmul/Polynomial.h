//===- Polynomial.h - integer polynomials over atom ids -------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Integer polynomials whose variables are atom ids, with exact arithmetic.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_MATMUL_POLYNOMIAL_H
#define POSEIDON_TECHNIQUES_MATMUL_POLYNOMIAL_H

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace poseidon {

using Monomial =
    std::vector<unsigned>; // sorted atom ids; empty == constant term

struct Polynomial {
  std::map<Monomial, int64_t> t;

  static Polynomial constant(int64_t c) {
    Polynomial p;
    if (c)
      p.t[Monomial{}] = c;
    return p;
  }
  static Polynomial atom(unsigned id) {
    Polynomial p;
    p.t[Monomial{id}] = 1;
    return p;
  }
  bool isZero() const { return t.empty(); }
  bool isConstant(int64_t *c = nullptr) const {
    if (t.empty()) {
      if (c)
        *c = 0;
      return true;
    }
    if (t.size() != 1 || !t.begin()->first.empty())
      return false;
    if (c)
      *c = t.begin()->second;
    return true;
  }
  bool contains(unsigned atom) const {
    for (auto &kv : t)
      if (llvm::is_contained(kv.first, atom))
        return true;
    return false;
  }
  void addTerm(const Monomial &m, int64_t c) {
    if (!c)
      return;
    auto it = t.find(m);
    if (it == t.end()) {
      t[m] = c;
      return;
    }
    it->second += c;
    if (it->second == 0)
      t.erase(it);
  }
  bool operator==(const Polynomial &o) const { return t == o.t; }
};

inline Polynomial operator+(const Polynomial &a, const Polynomial &b) {
  Polynomial r = a;
  for (auto &kv : b.t)
    r.addTerm(kv.first, kv.second);
  return r;
}
inline Polynomial operator-(const Polynomial &a, const Polynomial &b) {
  Polynomial r = a;
  for (auto &kv : b.t)
    r.addTerm(kv.first, -kv.second);
  return r;
}
inline Polynomial operator*(const Polynomial &a, const Polynomial &b) {
  Polynomial r;
  for (auto &x : a.t)
    for (auto &y : b.t) {
      Monomial m = x.first;
      m.insert(m.end(), y.first.begin(), y.first.end());
      llvm::sort(m);
      r.addTerm(m, x.second * y.second);
    }
  return r;
}
/// `a` divided by the nonzero integer `d`; none when a coefficient is not a
/// multiple of `d` (a fractional leading dimension), never a rounded guess.
inline std::optional<Polynomial> divideExactly(const Polynomial &a, int64_t d) {
  if (d == 0)
    return std::nullopt;
  Polynomial out;
  for (auto &kv : a.t) {
    if (kv.second % d)
      return std::nullopt;
    out.addTerm(kv.first, kv.second / d);
  }
  return out;
}

/// `p` as quotient * atom + rest, with `rest` free of the atom.
struct AtomSplit {
  Polynomial quotient, rest;
};

/// Divides one occurrence of `atom` out of every monomial of `p` that has it.
/// None when a monomial has the atom to a power above 1: a quadratic index is
/// not a matrix subscript.
inline std::optional<AtomSplit> divideByAtom(const Polynomial &p,
                                             unsigned atom) {
  AtomSplit s;
  for (auto &kv : p.t) {
    unsigned n = 0;
    for (unsigned a : kv.first)
      n += (a == atom);
    if (n == 0) {
      s.rest.addTerm(kv.first, kv.second);
      continue;
    }
    if (n > 1)
      return std::nullopt;
    Monomial m;
    bool dropped = false;
    for (unsigned a : kv.first) {
      if (a == atom && !dropped) {
        dropped = true;
        continue;
      }
      m.push_back(a);
    }
    s.quotient.addTerm(m, kv.second);
  }
  return s;
}

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_MATMUL_POLYNOMIAL_H
