/*
 * Fixed-point loops whose adjoint is iterated to convergence.
 *
 *     __enzyme_fixed_point(step, enzyme_fp_state, z, bytes, ...
 *                          [enzyme_fp_reduction, r,]
 *                          [enzyme_fp_max_iters, n,]
 *                          [enzyme_fp_control, control,]
 *                          [enzyme_checkpoint_region, ptr, bytes,]...
 *                          args...);
 *
 * is the loop
 *
 *     int64_t i = 0;
 *     while (step(i++, args...));
 *
 * which iterates z = phi(z, x) until step returns false. In reverse mode its
 * iterations are not reversed one by one. Only the state the loop ends in is
 * kept, and the reverse pass iterates the adjoint of one step at that state
 * until the adjoint converges (the two-phase adjoint of Christianson, 1994,
 * Tapenade's $AD FP-LOOP). The memory it needs is one snapshot, whatever the
 * number of iterations.
 *
 * - enzyme_fp_state, ptr, bytes: the state z, read and overwritten by each
 *   iteration (doubles, or floats). The squared 2-norm of its adjoint update
 *   decides when the adjoint iteration stops. Its shadow is zero after the
 *   reverse pass: the converged state does not depend on the initial guess.
 *   At least one is required; a state that is not listed still takes part in
 *   the adjoint iteration, but not in the convergence test.
 * - enzyme_fp_reduction, r (double): stop once the squared norm has fallen
 *   below r times its value after the first adjoint iteration (default
 *   1e-12), or once it grows again after five iterations.
 * - enzyme_fp_max_iters, n (int64_t): at most n adjoint iterations (default
 *   1000; 0 for no limit).
 * - enzyme_fp_control, control: decides instead of the test above, with
 *   Tapenade's adFixedPoint_notReduced protocol:
 *       int control(double *cumul, double *reduction);
 *   is called with *cumul = -1 before the first adjoint iteration, and with
 *   the squared norm after each, and returns whether to iterate again. It may
 *   change *cumul (for example to its sum over MPI ranks).
 * - enzyme_checkpoint_region, ptr, bytes: other memory the step reads that
 *   later code may overwrite, as for __enzyme_checkpoint_while. Globals the
 *   step writes, or reads while other code writes them, are kept without
 *   being listed, and so is the state.
 *
 * The primal state after the reverse pass is the one it started from.
 *
 * In forward mode the iterations are not differentiated one by one either.
 * The loop runs, and the tangent of one step is then iterated at the state it
 * ended in, s <- phi_z s + phi_x xdot, from s = 0 (the tangent of the initial
 * guess has no effect), until the squared 2-norm of the change of the state's
 * tangent falls below the reduction times its value after the first pass, with
 * the same limits and control function. A Newton iteration, whose step has
 * phi_z = 0 at its fixed point, takes two passes. The primal state after it is
 * the one the loop ended in.
 *
 * Instead of a step function, a loop may be marked from inside, as a Fortran
 * !DIR$ ENZYME FIXED_POINT directive lowers to:
 *
 *     while (...) {
 *       __enzyme_set_fixed_point(reduction, max_iters, control,
 *                                state, bytes, ...);
 *       ...
 *     }
 *
 * (a negative reduction or max_iters, or a null control, for the defaults;
 * all of them computed before the loop). Enzyme outlines one iteration as
 * the step. The loop needs a single latch and a single exit, from its header
 * or its latch; the adjoint iterations run its body even where the loop's
 * own test would stop it, as it does at the converged state.
 * Fortran passes every argument by reference, which the markers accept.
 */
#ifndef ENZYME_FIXED_POINT_H
#define ENZYME_FIXED_POINT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int enzyme_fp_state;
extern int enzyme_fp_reduction;
extern int enzyme_fp_max_iters;
extern int enzyme_fp_control;
extern int enzyme_checkpoint_region;

void __enzyme_fixed_point(void *step, ...);
void __enzyme_set_fixed_point(double reduction, int64_t max_iters,
                              void *control, ...);

#ifdef __cplusplus
}
#endif

#endif /* ENZYME_FIXED_POINT_H */
