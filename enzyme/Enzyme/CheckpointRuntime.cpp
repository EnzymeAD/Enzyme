//===- CheckpointRuntime.cpp - The built-in checkpointing schedules -------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The reference schedules of enzyme/checkpoint.h, compiled into Enzyme, so
// that a frontend that JIT-compiles the code it differentiates (Enzyme.jl,
// Reactant) needs no runtime of its own for them: a checkpointed loop without
// a schedule table of its own calls __enzyme_checkpoint_builtin(schedule),
// and Enzyme-MLIR's run-time schedules call __enzyme_ckpt_schedule_*. The
// frontend resolves those names to these definitions, which
// EnzymeCheckpointRuntimeSymbol hands it. Code compiled ahead of time keeps
// including the header, which defines them weakly.
//
//===----------------------------------------------------------------------===//

#define ENZYME_CHECKPOINT_RUNTIME
#include "enzyme/checkpoint.h"

#include <string.h>

extern "C" {

/// The address of the checkpointing runtime's function `name`
/// (__enzyme_checkpoint_builtin, or one of __enzyme_ckpt_schedule_*), or
/// null.
void *EnzymeCheckpointRuntimeSymbol(const char *name) {
  struct Entry {
    const char *name;
    void *fn;
  };
  static const Entry entries[] = {
      {"__enzyme_checkpoint_builtin", (void *)&__enzyme_checkpoint_builtin},
      {"__enzyme_ckpt_schedule_begin", (void *)&__enzyme_ckpt_schedule_begin},
      {"__enzyme_ckpt_schedule_next", (void *)&__enzyme_ckpt_schedule_next},
      {"__enzyme_ckpt_schedule_flag", (void *)&__enzyme_ckpt_schedule_flag},
      {"__enzyme_ckpt_schedule_iteration",
       (void *)&__enzyme_ckpt_schedule_iteration},
      {"__enzyme_ckpt_schedule_start", (void *)&__enzyme_ckpt_schedule_start},
      {"__enzyme_ckpt_schedule_slot", (void *)&__enzyme_ckpt_schedule_slot},
      {"__enzyme_ckpt_schedule_slots", (void *)&__enzyme_ckpt_schedule_slots},
      {"__enzyme_ckpt_schedule_end", (void *)&__enzyme_ckpt_schedule_end},
  };
  for (const Entry &e : entries)
    if (strcmp(e.name, name) == 0)
      return e.fn;
  return nullptr;
}
}
