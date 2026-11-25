//===- Schemes.h - what the GEMM runtimes and the compiler share ----------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the GEMM runtimes share with the compiler, whose pricing and accuracy
// model reproduce them: per scheme, its constants, the arithmetic the model
// simulates, and the values of its entry's Mode argument (GemmDispatch.inc).
// Included by both, in host and device code: keep it free of LLVM.
//
// The Ozaki-II constants and residue arithmetic reimplement those of the
// algorithm of K. Ozaki, Y. Uchino and T. Imamura (arXiv:2504.08009) as
// GEMMul8 v3.1.1 instantiates it (ozaki_rt.cu); no code is taken from GEMMul8
// (https://github.com/RIKEN-RCCS/GEMMul8, RIKEN R-CCS, MIT License). The TCEC
// scaling and the direct modes are Poseidon's own.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_RUNTIME_GEMM_SCHEMES_H
#define POSEIDON_RUNTIME_GEMM_SCHEMES_H

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

#ifndef POSEIDON_HD
#if defined(__CUDACC__) || defined(__HIP__)
#define POSEIDON_HD __host__ __device__
#else
#define POSEIDON_HD
#endif
#endif

//===----------------------------------------------------------------------===//
// Ozaki scheme II
//===----------------------------------------------------------------------===//

constexpr unsigned kOzakiIIMaxModuli = 14;

// INT8 moduli in the order the runtime consumes them; nm moduli use the first
// nm entries. p[0] = 256 is even and needs the 128 -> -128 residue fold.
constexpr int32_t kOzakiIIModuli[kOzakiIIMaxModuli] = {
    256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199};

// Each count needs a measured library_ozaki_rel cost-model row
// (poseidon-calibrate covers 8..14).
constexpr unsigned kOzakiIIModuliCounts[] = {8, 9, 10, 11, 12, 13, 14};

// Smallest modulus count the runtime accepts: the scaling constant below is
// defined from nm = 3 on (GEMMul8's nm = 2 entry does not follow its own
// formula, so nm = 2 is not offered).
constexpr unsigned kOzakiIIMinModuli = 3;

// Up to this many moduli P is exact in one double and the CRT runs in plain
// FP64; above it the CRT constants are split into a double-double.
constexpr unsigned kOzakiIIPIsDouble = 6;

// GEMMul8 v3.1.1's fast-mode scaling constant, indexed by nm:
// log2P(nm) = round-down-to-float(log2(P - 1)/2 - 1/2), P = the product of the
// first nm moduli, so 2^(2*log2P) <= (P - 1)/2. A row a of A (column b of B)
// is scaled by 2^floor(log2P - log2||a||_2), which by Cauchy-Schwarz keeps
// every entry of the integer product inside the symmetric CRT window.
// Computed in 80-digit decimal arithmetic; each literal round-trips exactly.
constexpr float kOzakiIILog2P[kOzakiIIMaxModuli + 1] = {
    0.0f,        0.0f,        0.0f,        11.4886732f, 15.4744444f,
    19.4486275f, 23.4050732f, 27.3555069f, 31.2875996f, 35.2071991f,
    39.1204758f, 43.0209236f, 46.9016991f, 50.762249f,  54.5805626f};

// The INT32 residue products hold |sum_k r_a r_b| <= K * 2^14, which stays
// below 2^31 only for K below this.
constexpr int kOzakiIIMaxK = 131072;

// Residue planes are held at extents rounded up to this: the INT8 GEMM's
// leading dimensions must be multiples of 4, and each plane base stays 16-byte
// aligned.
constexpr int kOzakiIIPadTile = 16;

// Block shapes of the sum-of-squares reduction that sets the shifts; they
// decide which partial sums are rounded, so the shifts' bits depend on them.
// A contiguous vector is folded by kOzakiIINormThreads threads; a strided one
// of length kOzakiIINormSplitK or more in chunks of kOzakiIINormChunk.
constexpr int kOzakiIINormThreads = 256;
constexpr int kOzakiIINormSplitK = 1024;
constexpr int kOzakiIINormChunk = 256;

// The CRT constants of nm moduli: -P as a round-to-nearest double-double
// (pHi, pLo), 1/P rounded to nearest (invP), and the weights
// q_i = (P/p_i)((P/p_i)^-1 mod p_i) split into qhi_i + qlo_i, the qhi_i at one
// granularity 2^G so that sum_i qhi_i r_i is exact for |r_i| <= p_i/2.
struct OzakiIIConstants {
  unsigned nm = 0;
  float log2P = 0.f;
  double qhi[kOzakiIIMaxModuli], qlo[kOzakiIIMaxModuli];
  double pHi = 0.0, pLo = 0.0, invP = 0.0;
};

inline long ozakiIIModInverse(long a, long m) {
  long r0 = ((a % m) + m) % m, r = m, s0 = 1, s = 0;
  while (r) {
    long q = r0 / r, t = r0 - q * r;
    r0 = r;
    r = t;
    t = s0 - q * s;
    s0 = s;
    s = t;
  }
  return ((s0 % m) + m) % m;
}

inline int ozakiIIBitLength(__int128 v) {
  int n = 0;
  for (; v; v >>= 1)
    ++n;
  return n;
}

// 1/P rounded to nearest even, by exact long division (P < 2^112, so the
// remainder stays inside __int128).
inline double ozakiIIReciprocalNearest(__int128 P) {
  __int128 r = 1;
  int e = 0;
  for (; r < P; r <<= 1)
    ++e;
  uint64_t q = 0; // floor(2^(e+53) / P), 54 bits
  for (int i = 0; i < 54; ++i) {
    q <<= 1;
    if (r >= P) {
      r -= P;
      q |= 1;
    }
    r <<= 1;
  }
  bool round = q & 1, sticky = r != 0;
  q >>= 1;
  if (round && (sticky || (q & 1)))
    ++q;
  return std::ldexp((double)q, -(e + 52));
}

inline OzakiIIConstants ozakiIIConstants(unsigned nm) {
  OzakiIIConstants c;
  c.nm = nm;
  __int128 P = 1;
  int rho = 0; // bound on the sum of the residue magnitudes
  for (unsigned i = 0; i < nm; ++i) {
    P *= kOzakiIIModuli[i];
    rho += kOzakiIIModuli[i] / 2;
  }
  double pHi = (double)P;
  c.pHi = -pHi;
  c.pLo = -(double)(P - (__int128)pHi);
  c.invP = ozakiIIReciprocalNearest(P);
  c.log2P = kOzakiIILog2P[nm];
  // The high parts keep 53 - ceil(log2 rho) bits below bitlen(P), so their
  // sum weighted by the residues is exact.
  int G = ozakiIIBitLength(P) - (53 - ozakiIIBitLength(rho - 1));
  for (unsigned i = 0; i < nm; ++i) {
    long p = kOzakiIIModuli[i];
    __int128 Mi = P / p;
    __int128 e = (Mi * ozakiIIModInverse((long)(Mi % p), p)) % P;
    if (nm <= kOzakiIIPIsDouble) {
      c.qhi[i] = (double)e; // exact: e < P < 2^48
      c.qlo[i] = 0.0;
    } else {
      __int128 hi = (e >> G) << G;
      c.qhi[i] = (double)hi;
      c.qlo[i] = (double)(e - hi);
    }
  }
  return c;
}

// The symmetric residue modulo p = kOzakiIIModuli[i] of a 64-bit integer (a
// scaled operand entry) or a 32-bit one (an INT32 product entry). The method
// is GEMMul8's (reduce_mant and mod_small_nowrap in src/oz2/mod/mod_core.hpp):
// the bytes of |a| are dotted with the byte weights 2^(8k) mod p, four per
// 32-bit word (one __dp4a each on CUDA devices), and the sum s is reduced with
// one 32-bit multiply-high by m = floor(2^32/p) + 1. That quotient is
// floor(s/p) whenever s * (p - 2^32 mod p) < 2^32; the largest sum the eight
// bytes of any 64-bit magnitude give meets that bound for every modulus, which
// a static_assert checks, so the residue is exact for every input.

// Modulus i with its byte weights: byte k of wlo is 2^(8k) mod p, byte k of
// whi is 2^(32+8k) mod p (each below p <= 256), and m32 = floor(2^32/p) + 1.
struct OzakiIIModulus {
  uint32_t p, wlo, whi, m32;
};

POSEIDON_HD constexpr OzakiIIModulus ozakiIIModulus(unsigned i) {
  const uint32_t p = (uint32_t)kOzakiIIModuli[i];
  uint32_t wlo = 0, whi = 0;
  for (unsigned k = 0; k < 4; ++k) {
    wlo |= (uint32_t)((1ull << (8 * k)) % p) << (8 * k);
    whi |= (uint32_t)((1ull << (32 + 8 * k)) % p) << (8 * k);
  }
  return OzakiIIModulus{p, wlo, whi, (uint32_t)((1ull << 32) / p + 1)};
}

// Every modulus: the largest byte-weight sum of a 64-bit magnitude, s_max =
// 255 * (sum of the eight weights), satisfies s_max * (p - 2^32 mod p) < 2^32.
constexpr bool ozakiIIResiduesExact() {
  for (unsigned i = 0; i < kOzakiIIMaxModuli; ++i) {
    const OzakiIIModulus m = ozakiIIModulus(i);
    uint64_t sMax = 0;
    for (unsigned k = 0; k < 4; ++k)
      sMax += 255 * (((m.wlo >> (8 * k)) & 255) + ((m.whi >> (8 * k)) & 255));
    if (sMax * (m.p - (1ull << 32) % m.p) >= (1ull << 32))
      return false;
  }
  return true;
}
static_assert(ozakiIIResiduesExact(),
              "the multiply-high reduction is not exact on every byte sum");

// c + sum_k byte_k(a) * byte_k(w), bytes unsigned.
POSEIDON_HD inline uint32_t ozakiIIByteDot(uint32_t a, uint32_t w, uint32_t c) {
#ifdef __CUDA_ARCH__
  return __dp4a(a, w, c);
#else
  for (unsigned k = 0; k < 32; k += 8)
    c += ((a >> k) & 255) * ((w >> k) & 255);
  return c;
#endif
}

// s mod p for a byte-weight sum s of m.
POSEIDON_HD inline uint32_t ozakiIIReduceByteSum(uint32_t s, OzakiIIModulus m) {
  return s - m.p * (uint32_t)(((uint64_t)s * m.m32) >> 32);
}

// The representative of r = |a| mod p for a of sign `neg`: [-(p-1)/2, (p-1)/2]
// for odd p, [-128, 127] for p = 256 (the int8 wraparound of GEMMul8's INT8
// backend). The CRT constants assume these.
POSEIDON_HD inline int ozakiIISymmetric(uint32_t r, bool neg, uint32_t p) {
  const uint32_t t = (neg && r != 0) ? p - r : r;
  return (int)t - (t > (p - 1) / 2 ? (int)p : 0);
}

// The residue of a scaled operand entry, an INT8 plane element.
POSEIDON_HD inline int8_t ozakiIIResidueI8(int64_t a, OzakiIIModulus m) {
  const uint64_t u = a < 0 ? 0 - (uint64_t)a : (uint64_t)a;
  const uint32_t s = ozakiIIByteDot(
      (uint32_t)u, m.wlo, ozakiIIByteDot((uint32_t)(u >> 32), m.whi, 0));
  return (int8_t)ozakiIISymmetric(ozakiIIReduceByteSum(s, m), a < 0, m.p);
}

// The residue of an INT32 product entry.
POSEIDON_HD inline int ozakiIIResidueI32(int32_t a, OzakiIIModulus m) {
  const uint32_t u = a < 0 ? 0 - (uint32_t)a : (uint32_t)a;
  return ozakiIISymmetric(ozakiIIReduceByteSum(ozakiIIByteDot(u, m.wlo, 0), m),
                          a < 0, m.p);
}

//===----------------------------------------------------------------------===//
// TCEC: the power-of-two operand scaling, carried as integer exponents
//===----------------------------------------------------------------------===//

// Exponent n of the scale 2^n that brings an operand whose largest magnitude is
// m to [2^10, 2^11]: inside FP16's range with headroom for the high limb of the
// split, and far above its subnormals. The exponent is taken from m rounded to
// FP32, the operand the split sees, wherever that is finite and nonzero, and
// from m itself beyond FP32's range. An operand that is zero, or holds an Inf
// or NaN, is left unscaled.
POSEIDON_HD inline int tcecScaleExponent(double m) {
  float f = (float)m;
  if (f > 0.f && f <= FLT_MAX)
    return 10 - ilogbf(f);
  if (m > 0. && m <= DBL_MAX)
    return 10 - ilogb(m);
  return 0;
}

// 2^n for n in [-1022, 1023].
POSEIDON_HD inline double tcecPow2(int n) {
  uint64_t bits = (uint64_t)(n + 1023) << 52;
  double d;
  memcpy(&d, &bits, sizeof d);
  return d;
}

// x * 2^n, rounded once, for any n. A factor beyond the normal range is applied
// in parts; toward the subnormals each part leaves the intermediate 2^53 above
// them, so only the last product can round.
POSEIDON_HD inline double tcecScale(double x, int n) {
  for (int i = 0; i < 2 && n > 1023; ++i) {
    x *= tcecPow2(1023);
    n -= 1023;
  }
  for (int i = 0; i < 2 && n < -1022; ++i) {
    x *= tcecPow2(-1022 + 53);
    n += 1022 - 53;
  }
  n = n > 1023 ? 1023 : n < -1022 ? -1022 : n;
  return x * tcecPow2(n);
}

// The last argument of __poseidon_tcec_dgemm: cuMpSGEMM's compute mode. The
// TCEC runtime links cuMpSGEMM (https://github.com/enp1s0/cuMpSGEMM, MIT
// License) and includes none of it: the authors' implementation of the scheme
// of H. Ootomo and R. Yokota, IJHPCA 36(4), 2022
// (doi:10.1177/10943420221090256).
enum PoseidonTcecMode : int {
  kTcecFp16Tcec = 1,
  kTcecTf32Tcec = 2,
  kTcecFp16Tc = 3,
  kTcecTf32Tc = 4,
};

//===----------------------------------------------------------------------===//
// Direct
//===----------------------------------------------------------------------===//

// The last argument of __poseidon_direct_dgemm: the format each FP64 operand
// is rounded to (TF32 through FP32), all with an FP32 accumulator.
// kDirectFp32 is SGEMM rather than a tensor-core product.
enum PoseidonDirectMode : int {
  kDirectFp16 = 1,
  kDirectBf16 = 2,
  kDirectTf32 = 3,
  kDirectFp32 = 4,
};

#endif // POSEIDON_RUNTIME_GEMM_SCHEMES_H
