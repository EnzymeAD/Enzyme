//===- checkpoint_schemes.c - Reference schemes for Fortran callers -------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The reference checkpointing schemes of enzyme/checkpoint.h are static
// objects of a C header. Fortran cannot take their address, so these return
// it (see enzyme_ckpt_revolve and its siblings in enzyme.f90).
//
//===----------------------------------------------------------------------===//

#include "enzyme/checkpoint.h"

const void *enzyme_ckpt_revolve_scheme(void) { return &EnzymeCkptRevolve; }
const void *enzyme_ckpt_periodic_scheme(void) { return &EnzymeCkptPeriodic; }
const void *enzyme_ckpt_store_all_scheme(void) { return &EnzymeCkptStoreAll; }
