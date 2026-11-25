//===- Explain.h - what a solve changed, considered and rejected ----------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Optimization remarks (pass name "poseidon") and <func>.explain.json, built
// from the solver's own data after it has chosen and before anything is
// materialized. Output only: nothing here feeds back into a decision.
//
// <func>.explain.json, schema "poseidon-explain/1", keys in snake_case; a key
// whose value is unknown or not finite is left out:
//   schema     "poseidon-explain/1"
//   function   name (demangled), symbol, ir (the IR function solved), site (the
//              call: file, line, col), loc (the definition); for a solved site
//              also canonical_hash (of the form the profile was recorded
//              against) and profile (the profile file read)
//   target     tau and tau_from ("site" or "flag") under an accuracy target,
//              else budget (the -poseidon-cost-budget share); for a solved site
//              also cost_model, selector ("dp", "joint", "error-budget",
//              "pinned"), from_cache (the DP table was read from the cache),
//              sites (the IR functions of a joint solve)
//   scale      (solved sites) accuracy_scale S and baseline_accuracy_cost A0:
//              an accuracy cost a is the modelled relative error (A0 + a) / S;
//              profiled_cost, the cost-model units the profiled run spent, a
//              cost share is a fraction of; kappa, kappa_mark and kappa_factor
//              when the profiling run measured a condition number
//   result     status: "rewritten", "as_written" (solved, nothing applied),
//              "no_solution" (no frontier point meets the target or the
//              budget), "nothing_to_optimize", "no_fp_operation", "unprofiled"
//              (no profile of the site), "never_executed" (the profiling run
//              never ran it), "below_cost_share" (under
//              -poseidon-min-cost-share); reason: why, for every status but
//              "rewritten"; profile: the file looked at, for "unprofiled" and
//              "never_executed"; for a solved site also: applied (the ids
//              applied), steps (each step the solve chose, in order: id,
//              choice, the text of its "Applying solution for" line, and
//              applied, false for a step an earlier one made moot), pin (the
//              -poseidon-pin value that applies them again),
//              selected (the frontier point selected: cost, the share it adds,
//              accuracy_cost, modelled_rel_error), best_achievable_rel_error
//              (for "no_solution" under a target), cost_before, cost_after,
//              modelled_rel_error_as_written, modelled_rel_error
//   subgraphs  (solved sites) each optimization unit: id (S<i> an FP
//              subgraph, M<i> a matrix product), kind ("subgraph", "matmul"),
//              loc, shape, cost, executions, ops (each with opcode, type, loc,
//              executions, sensitivity, gradient, min, max, and for a call of
//              a math function float_call, the C function of its float form),
//              inputs, outputs, original, herbie, left_as_written, and
//              alternatives: each candidate with id, kind ("algebraic",
//              "precision", "expansion", "matmul"), summary (for a product,
//              the realization the solve names it by), precision, outputs,
//              fpcore, c, approx, approx_note, changes (each with from, to
//              and the indices of its ops), cost_delta, modelled_error,
//              modelled_domain_error (of a product: under the "error-budget"
//              selector the applied one is the cheapest whose domain error is
//              at most tau), herbie_error_bits, status ("applied", "pareto",
//              "rejected") and reason
//   pareto     (solved sites) the frontier in cost order: budget (the
//              -poseidon-cost-budget that selects the point; none for a point
//              no budget selects), cost, modelled_rel_error, applied, selected
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SOLVE_EXPLAIN_H
#define POSEIDON_SOLVE_EXPLAIN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Error.h"

namespace poseidon {

struct SiteState;
struct Options;
class Session;
struct SolutionStep;
struct SolveRecord;

/// Whether the solve of `F` is explained: -poseidon-explain names a directory,
/// or a remark of the pass is enabled for `F`.
bool explainWanted(const Options &O, const llvm::Function &F);

/// The explanation of a site the pass leaves as written without solving it:
/// `status` is what kept it from the solve and `reason` says why. F is the
/// site's clone, or with `body` the site's body when no clone was made. `tau`
/// > 0 is the site's accuracy target, from the site when `siteTarget`.
void explainLeftAlone(Session &S, llvm::Function &F, double tau,
                      bool siteTarget, llvm::StringRef status,
                      llvm::StringRef reason, llvm::StringRef profile = "",
                      bool body = false);

/// A remark and the explanation of a site with nothing to optimize.
void explainNothingToOptimize(Session &S, llvm::Function &F, double tau,
                              bool siteTarget);

/// A remark and the explanation of a site no profile has records of: `status`
/// is "unprofiled" or "never_executed", `profile` the file looked at.
void explainUnprofiled(Session &S, llvm::Function &F, double tau,
                       bool siteTarget, llvm::StringRef status,
                       llvm::StringRef cause, llvm::StringRef profile);

/// The missed-optimization remark, once per compile, that no algebraic
/// rewrite is proposed because no Herbie can run.
void remarkNoHerbie(Session &S, llvm::Function &F);

/// Remarks and the explanation of the solve of the site of `st`; an operation
/// of a subgraph the cost model has no row for is an error.
llvm::Error explainSolve(Session &S, llvm::Function &F, SiteState &st,
                         llvm::ArrayRef<SolutionStep> steps,
                         const SolveRecord &rec);

} // namespace poseidon
#endif // POSEIDON_SOLVE_EXPLAIN_H
