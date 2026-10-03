# Design note: cross-site sensitivity as a measured condition number

Status 2026-09-05: design decided, implementation scheduled in phase C3 (profile-gen
decoupling) because it replaces the same probe machinery. Companion: PLAN.md WS-E.

## 1. What exists and why it invites objections

Today's mechanism, all paths under `poseidon-clean/enzyme/Enzyme/`
(`Poseidon/PoseidonProfUtils.cpp:522-588`, `Runtimes/FPProfiler/FPProfilerCUDA.cu:264-292`,
`Poseidon/scripts/poseidon_sens_probe.py`, `Poseidon/Poseidon.cpp:1634, 2670-2683`):

1. After AD, every value logged by `poseidonLogValueCUDA` in every profiled function is
   wrapped as `v * (1 +/- eps)` (sign from a bit hash of `v`), inert unless the runtime
   env `ENZYME_FPPROFILE_PERTURB_FUNC` matches the function name as a substring.
2. A Python driver re-runs the profiling workload once per site with that site armed and
   defines `S = max` over all profiled slots of the eps-normalized relative change of
   (a) execution counts and (b) operand minimum magnitudes, capped at 1e30 and floored at
   1e-12. The docstring calls these "convergence-state observables of an iterative solver".
3. `site_sensitivity.txt` (`<name-substring> <S>`) is read at profile-use and multiplies the
   accuracy costs of the site's MATMUL candidates only.

Objections a reviewer will raise:

- The observable is a proxy (iteration count, smallest operand magnitude) chosen because it
  discriminated on R-ChFSI; it is not the quantity the accuracy target tau refers to, and it
  only exists for programs with a convergence loop.
- S is dimensionally incoherent with the per-instruction gradient weights it multiplies:
  the paper presents S as "how sensitive the application result is to each annotated
  computation", the code measures something else. Values of 1e15 are the cap, not a
  measurement.
- Perturbing every intermediate operation of the site, then multiplying by the accuracy
  model's own within-site error propagation, double counts the site's internal
  amplification.
- Name-substring matching, env-var plumbing, a side file outside the profile, and
  matmul-only application are implementation accidents.

## 2. The quantity that is actually needed

The accuracy model estimates, per site k and candidate c, a sensitivity-weighted relative
error of the site's outputs, `e_k(c)`, using reverse-mode gradients of the site's outputs
with respect to its instructions. A joint solve over sites needs these errors on a common
scale: the effect on ONE application-level quantity of interest `m` (the quantity tau is a
tolerance on). To first order

    delta m / m  =  sum_k  kappa_k * e_k(c_k)

where `kappa_k` is the relative condition number of `m` with respect to site k's outputs
along the direction "every output of k carries an independent relative rounding error".
This is the standard condition-number notion; it composes with the within-site model
without double counting because the perturbation is applied at the site's output interface,
not inside it.

## 3. Design

### 3.1 Declared quantity of interest

The application reports the number tau refers to, once, at the end of the profiling
workload:

    poseidon_metric("max_residual", value);   // host runtime call, any scalar

The profiler runtime writes it to `<profile dir>/metric.txt`. For R-ChFSI the natural
metric is the maximum residual norm after the profiled iterations (or the eigenvalue error
against the planted spectrum); for ozp the RMSD; for GENGA the energy drift. If no metric
is declared, every `kappa_k = 1` (no cross-site weighting) and the joint solve prints a
warning naming the missing call. No proxy fallback.

### 3.2 Perturbation at the site interface

In the profile-gen compile Poseidon (not Enzyme) already knows each site's output stores:
stores whose pointer derives from an output argument of the annotated body. Only those
stored values are wrapped:

    v' = __poseidon_perturb(site_id, v)      // returns v * (1 +/- eps) when armed

`site_id` is an integer Poseidon assigns per marker in module order; the runtime writes the
registry `<profile dir>/sites.txt` (`id  name`). Arming is by id:
`POSEIDON_PROBE_SITE=<id>`, `POSEIDON_PROBE_EPS=<eps>`. The value-hash sign stays (a uniform
factor commutes through linear operations and hides cancellation); that rationale is the one
comment the runtime keeps.

### 3.3 Driver

`tools/profile/poseidon-probe` (replaces `poseidon_sens_probe.py`):

1. Baseline run twice: profiles plus `metric.txt`; the difference between the two baseline
   metrics is the noise floor `n` (atomics and reduction order make the run nondeterministic
   at the last bits).
2. For each site id and for `eps` in a small ladder (default `1e-8, 1e-6, 1e-4`): one run,
   read `m_k(eps)`, and set

       kappa_k(eps) = |m_k(eps) - m_0| / (eps * |m_0|)      (absolute form when m_0 = 0)

   `kappa_k` is reported at the smallest eps whose response exceeds `4 n`; if the response
   is within noise at every eps, `kappa_k = 0` with a `below-noise` mark. A run that ends
   non-finite, hits an iteration cap, or changes the exit path is `kappa_k = inf` with a
   `diverged` mark (the site cannot tolerate that eps; the solver keeps its precision).
   The ladder also yields a linearity check: `kappa_k` roughly constant across eps means a
   derivative was measured; a jump between two eps values is a tolerance threshold and is
   printed as such.
3. Writes `Kappa = <value>` and `KappaMark = ok|below-noise|diverged` into the header of the
   site's own `.fpprofile`. No separate side file, no substring keys.

### 3.4 Consumption

`parseProfileFile` reads `Kappa` into the site header. `collectFPCandidates` scales the
accuracy cost of EVERY candidate of the site (scalar subgraphs and matrix products alike) by
`kappa_k`; `loadSiteSensitivity` and `site_sensitivity.txt` are deleted. The joint DP is
unchanged: its shared budget now bounds `sum_k kappa_k e_k(c_k)`, which is the first-order
change of the declared metric. A single-site program is unaffected (`kappa = 1` or a
constant factor that cancels against tau).

### 3.5 What changes for the paper's numbers

The R-ChFSI picks are expected to persist (the filter's output error is corrected by the
next Rayleigh-Ritz step, the residual/Gram/projection errors are not), but the measured
kappa values will not equal today's S values. The chase gate therefore changes by design:
re-bank the chase G0 after C3 with the new probe, and confirm the four single-rewrite
picks and the 2.43x configuration are reproduced. The paper text ("how sensitive the
application result is to each annotated computation") becomes literally true.

## 4. Other design debt found during the review (WS-E in PLAN.md)

| Item | Today | Clean form |
|---|---|---|
| E2 reduction trip counts | `./.poseidon_redtrip/<fn>.redtrip` written into the CWD at profile-gen COMPILE time, read at profile-use | Poseidon embeds the per-site static data as a string global in the instrumented binary; the runtime writes it into the site's `.fpprofile` header (`RedTrip = ...`). The profile becomes self-contained. |
| E3 slot matching | Instructions are matched between the two compiles by rank on a canonicalized clone; a drift changes slots silently | The profile header carries a hash of the canonicalized function (opcode sequence + types); profile-use recomputes it and aborts on mismatch with the two hashes printed. |
| E4 cost quantization | The DP keys costs by `llround(cost)`; GENGA's per-op costs (~1e-4) collapse, so the artifact feeds a x1e6-scaled CSV copy (`cm_native_x1e6.csv`) | The solver quantizes at a fixed resolution `r = 1e-6` of the CSV's unit and accepts budgets as real numbers in CSV units; `budgets.txt` prints real costs. The derived CSV disappears. Budget values in scripts and golden TSVs change units (cosmetic; re-bank). |
| E5 site scaling | Only matmul candidates are weighted by S | Part of 3.4: every candidate of the site. |
| E6 annotation boilerplate | The application allocates, zeroes and seeds a shadow buffer per pointer argument and passes `enzyme_dup, p, p_sh` pairs (chase_real.cu:495-501) | With Poseidon-owned probes (C3), the output-store probe's reverse rule returns the seed directly and the input-load probe discards its adjoint, so the marker takes plain pointers: `__poseidon_fp_optimize(body, out_ptrs..., in_ptrs...)`. Needs the device custom-gradient test first; fall back to today's convention if Enzyme's activity analysis will not carry a forced-active probe on `enzyme_const` memory. |
| E7 legacy device knobs | `-poseidon-gpu-fp64-ratio`, `-poseidon-gpu-sm-count`, `-poseidon-gpu-blocks-per-sm` still parsed and printed | Removed with the flags header (WS-B item 3); native CSVs carry the device. |

## 5. Revision after the first measurement (2026-09-06, C3b-2)

The design in section 3 was implemented and measured on R-ChFSI (five sites, metric
`max_resid`). Three things were wrong; the corrected rules replace 3.2 to 3.4.

1. **Where the perturbation goes.** Perturbing a site's stored values through the slot
   probes put the perturbation inside every reduction loop (each partial sum scaled by
   1 +/- eps), inflating the reduction sites' kappa by at least sqrt(K). Corrected rule:
   a dedicated OUTPUT probe on the stored value at each argument-derived store, after the
   reduction has finished; slot probes never perturb.
2. **Units.** The DP's accuracy cost of a candidate is `mean_samples sum_i |grad_i| |err_i|`,
   a gradient-weighted ABSOLUTE error in site-output units (for a product stored directly,
   the product's absolute error; for `resid`, whose output is `R = HV - lambda V`, the
   absolute error of `HV` carried into `R` with gradient 1). kappa is defined per RELATIVE
   perturbation of the outputs. The composition `kappa * accCost` therefore misses the
   site's output magnitude: for `resid` the metric is `||R||` itself, so kappa is ~1e3 while
   `|HV| / |R|` is ~1e11, and the solver bought the residual site instead of the
   Rayleigh-Ritz product (0/64 locked at the paper's fastest budget). Corrected rule: the
   site factor is `kappa_k / ybar_k`, where `ybar_k` is the mean absolute value of the
   site's output slots, taken from a new per-slot profile field `SumAbsValue` (the
   runtime already sums the signed value; the absolute sum is one more atomic). The
   factor is applied only when the site carries a `Kappa` line (no declared metric means no
   cross-site comparison; single-site behaviour and the banked tables stay untouched).
   Composition is then `delta m / m ~ sum_k kappa_k * accCost_k / ybar_k`.
3. **Ladder rule with a deterministic workload.** The noise floor was exactly 0, so
   "response above 4n" accepted the smallest eps unconditionally, including non-monotonic
   ladders. Corrected rule: kappa is reported at the smallest eps whose value agrees with
   its next larger neighbour within a factor of 2 (a linear regime); if no two agree,
   the smallest eps is reported with `KappaMark = nonlinear`; a run that diverges at an eps
   above a clean one keeps the clean value with `KappaMark = ok` and an extra
   `KappaDivergesAbove = <eps>` line; diverged at the smallest eps stays `diverged`.

What must be reproduced after the correction (the paper's claims): single-rewrite picks
target only the filter; BF16TCEC filter + Ozaki-II nm=14 Rayleigh-Ritz; the fastest locked
configuration (direct BF16 filter + nm=12 Rayleigh-Ritz) at budget -5700206; the
sensitivity-blind arm fails to converge.

## 6. Second revision (2026-09-06, after C3b-3): calibrate the model against the metric

C3b-3 implemented section 5 and measured again. Output-interface perturbation cannot see
the failure mode that matters: a low-precision `resid` injects an ABSOLUTE error of order
`eps * |HV|` into `R = HV - lambda V`, which keeps `R` above the locking tolerance forever,
while a RELATIVE perturbation of the stored `R` only rescales a residual that still
converges. Measured: hemm and resid both ~1.5e3 and `nonlinear`; the corrected composition
still buys `resid` at the paper's fastest budget (0/64). The old probe perturbed every
operation for exactly this reason.

Final design, replacing sections 3.2, 3.4 and 5:

1. **Reference perturbation = per-operation relative noise.** Every slot probe's augmented
   forward applies `v * (1 +/- eps)` when its site is armed (value-hash sign), as C3a
   placed it. This is the site "computed at precision eps". No output probes, no
   `SumAbsValue`.
2. **kappa_k** = metric response per unit of that noise, from the eps ladder with the
   section-5 linear-regime rule (`nonlinear`, `diverged`, `KappaDivergesAbove` kept).
3. **The model's own prediction for the same noise** is already in the profile: to first
   order the site-output error under per-op relative noise eps is
   `eps * sum_{o in k} SumSens_o` (`SumSens_o = sum over executions of |dy/do * o|`, the
   quantity the profiler logs per slot), in the same accumulated site-output units as the
   DP's accuracy cost `accCost_c = mean_samples sum_i |sumGrad_i| |err_i|`.
4. **Site factor** = `kappa_k / sum_{o in k} SumSens_o`. The predicted relative change of
   the metric from candidate c at site k is `factor_k * accCost_c`: the within-site
   propagation appears once in the numerator (measured) and once in the denominator
   (modelled) and cancels; what remains is the measured downstream condition times the
   candidate's error expressed as an equivalent per-operation relative error. Applied only
   when the site carries `Kappa`; `diverged` = 1e30; a site with `Kappa` but zero SumSens
   aborts (never substitute).
5. Nothing else changes: the joint DP bounds `sum_k factor_k * accCost_k`.

Expected on R-ChFSI: `resid` TCEC costs ~kappa * 1e-7, `rrhv` Ozaki-II nm=12 ~kappa * 1e-12,
so the Rayleigh-Ritz product is lowered before the residual, as in the paper.

## 7. AD experiment (2026-09-06): the condition number is a Jacobian norm, not a gradient

Whole-workload reverse-mode AD of the chase surrogate from the declared metric was built
(`refactor-2026-09-work/ad-experiment/chase_ad.cu`: 15 auto-style quartets, hand adjoints
for Cholesky-QR, `syevd` and `V Z`, all FD-checked) and works: the gradient
`dm/dy_k` matches central differences to 0.1 percent wherever the first-order term is
measurable. It is nevertheless the wrong quantity for cross-site weighting:

- Every declared metric here is an error norm at its convergence floor (max residual, RMSD,
  energy drift). At a minimum the first-order coefficient is annihilated by stationarity
  (`d||R - h V||/dh` at `h = 0` is `-(R.V)/||R||`, of order 1e-16 here), and the response to a
  perturbation is quadratic: `m(eps)^2 = m0^2 + Q eps^2`. Fitting the probe's own runs gives
  `sqrt(Q)/m0` = 1718 / 1756 / 3.4e10 / 1.1e10 / 3.4e10 for hemm / resid / gram / rrhv /
  rrproj, matching the banked kappa within 13 percent.
- So kappa_k = ||J_k delta_k|| / ||R||, the norm of the site's Jacobian applied to the
  perturbation direction, not `|grad m . delta|`. The metric gradient under-weights Gram and
  the projection by 3.6e4 and would reproduce the sensitivity-blind failure.
- Reverse AD can still recover it by seeding with random vectors (Hutchinson, 32 seeds
  reproduce every banked kappa within 1.1x), but at five sites it is not cheaper than the
  16-run ladder and needs six invasive rewrites of the host driver (single-assignment
  buffers, device-resident scalars, no memcpy in the region, one pooled allocation,
  adjoint snapshot hooks, frozen control flow). Forward mode would compute `||J_k delta_k||`
  exactly in one tangent pass per site with no tape and no single-assignment requirement,
  but still requires the whole workload to be differentiable, which a library user cannot
  provide.

Decision: finite differences stay the release design for cross-site weighting, run by the
profiling binary itself (WS-F item 3). The design note's name for the quantity is the
site's Jacobian norm against the declared metric; the ladder's `nonlinear` mark on the
filter is the signature of the quadratic regime, not a measurement failure. Forward-mode
tangents are the recorded path if a fully differentiable workload ever makes it worth it.

## 8. Shadow buffers are keyed by allocation (2026-09-06, C4b-1)

The adjoint of a buffer is the buffer's shadow. C4a's profile-generation runtime keyed the
managed shadows by (site, argument), which gives a buffer that one site writes and another
reads two unrelated adjoints and cuts the chain between them. The benches' hand-wiring
approximated the right thing: ozp allocated one `X_sh` / `X2_sh` / `g2_sh` / `tmp_sh` per
buffer and passed `X2_sh` both as GEMM1's output shadow and as GEMM2's input shadow, so the
second product's reverse pass fed the first one's. Keyed by argument that chain is gone and
the profile's `SumGrad` drops to 0.25x the banked value.

The rule is therefore: one shadow per device allocation, keyed by the allocation base from
`cuMemGetAddressRange`, created zeroed on first sight, shared by every site and every launch
that passes a pointer into that allocation with the offset preserved, and freed at exit. A
base seen with a different extent is a reused address, not the same buffer, so its shadow is
rebuilt.

The seed is applied once per (allocation, site that writes it), not once per allocation. A
profiled launch runs its own reverse pass immediately and the launches run in forward order,
so the first site that writes a buffer consumes the buffer's adjoint and leaves zero behind
for the sites that write it later. R-ChFSI measures this: `dW` is written by the filter and
again by the Rayleigh-Ritz `H V`, and with a single seed per allocation the filter's first
launch takes it and `rrhv` records `SumGrad = 0`, `SumSens = 0`, which is exactly the
degenerate profile the seed exists to prevent (the solve aborts on it, as it should). Seeding
each site's own outputs once keeps the existing rule ("the seed is one on the site's
outputs") and still lets the shared buffer carry the projection's adjoint back to the filter,
which the hand-wired two-shadow version could not.

Measured against the C3b-7 bank (same profiles otherwise): ozp recovers the banked
`SumGrad = 5.364904507e+08` and `SumSens = 5.202223456e+05` exactly; genga kepler is
identical; genga force/unified differ in exactly the fields, and only the fields, that also
move between two runs of one plugin (three `SumValue` and nine `SumGrad` entries of order
1e-12 against a 1.96e+05 total, from the atomic order of the reverse pass over the shared
memory tile); R-ChFSI `hemm` moves by
+0.8% in `SumGrad` and +7.0% in `SumSens` and `rrhv` by -0.8% and -21.6%, because `dW` is now
one adjoint instead of two.

Downstream of the profile nothing moved. With the profile bytes the pre-C4a bank used held
fixed, the solve reproduces that bank's budgets, DP table and every decision line for genga
force / kepler / unified and for ozp at both banked budgets and at the autopick tolerance.
The residual differences the gate reports against the freshly generated profiles are the
profile's own last-ulp spread, which flips a tie in the precision-tuning percentage mix.
R-ChFSI keeps all 16 guided ladder rows and the sensitivity-blind arm: the same picks, the
same locked counts and the same eigenvalue errors as the run the condition-number design was
measured on.
