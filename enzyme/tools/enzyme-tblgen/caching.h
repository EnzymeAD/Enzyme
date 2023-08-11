// clang-format off
//

#ifndef ENZYME_TBLGEN_CACHING_H
#define ENZYME_TBLGEN_CACHING_H 1

#include "llvm/Support/raw_ostream.h"

#include "datastructures.h"

std::string get_input_mat(const llvm::DagInit *ruleDag);

void emit_mat_vec_caching(const TGPattern &pattern, size_t i, llvm::raw_ostream &os);

void emit_need_cache_info(const TGPattern &pattern, raw_ostream &os);

void emit_scalar_cacheTypes(const TGPattern &pattern, llvm::raw_ostream &os);

void emit_vec_like_copy(const TGPattern &pattern, llvm::raw_ostream &os);

void emit_cache_for_reverse(const TGPattern &pattern, llvm::raw_ostream &os);

void emit_caching(const TGPattern &pattern, llvm::raw_ostream &os);

#endif
