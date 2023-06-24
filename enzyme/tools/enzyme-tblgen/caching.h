// clang-format off
//

#ifndef ENZYME_TBLGEN_CACHING_H
#define ENZYME_TBLGEN_CACHING_H 1

#include "llvm/Support/raw_ostream.h"

#include "datastructures.h"

void emit_mat_vec_caching(const TGPattern &pattern, size_t i, llvm::raw_ostream &os);

void emit_scalar_caching(const TGPattern &pattern, llvm::raw_ostream &os);

void emit_scalar_cacheTypes(const TGPattern &pattern, llvm::raw_ostream &os);

void emit_vec_copy(const TGPattern &pattern, llvm::raw_ostream &os);

void emit_mat_copy(const TGPattern &pattern, llvm::raw_ostream &os);

void emit_cache_for_reverse(const TGPattern &pattern, llvm::raw_ostream &os);

void emit_caching(const TGPattern &pattern, llvm::raw_ostream &os);

#endif
