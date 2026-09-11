/* Enzyme custom reverse-mode rule for the Fortran subroutine `fsquare`.
 *
 * This is the C-translation-unit workaround for the fact that flang has no way
 * to spell a custom rule in Fortran source: flang plugins run after semantics
 * and replace codegen rather than augmenting it, so they cannot introduce a
 * directive that survives into IR, and the registration global below needs a
 * *constant* array of function addresses, which Fortran cannot express
 * (C_FUNLOC is not a constant expression, so it cannot initialize a module
 * variable).  So the registration is written in C and llvm-linked in.
 *
 * fsquare is `bind(C)` and takes its arguments by reference, hence the pointer
 * signature.  This file is an input to custom_rule_from_c.f90, not a test.
 */

extern void fsquare(const double *x, double *y);

/* Counters, so the test can assert each half of the rule ran exactly once. */
int augment_calls = 0;
int gradient_calls = 0;

/* Augmented forward pass: run the primal and zero the shadow of the output.
 * The tape is unused here, so return null. */
void *augment_fsquare(const double *x, const double *dx, double *y,
                      double *dy) {
  augment_calls++;
  *y = *x * *x;
  *dy = 0.0;
  return (void *)0;
}

/* Reverse pass.  Deliberately *not* 2*x: the bogus factor is what lets the test
 * distinguish this rule from the derivative Enzyme would have synthesized. */
void gradient_fsquare(const double *x, double *dx, const double *y,
                      const double *dy, void *tape) {
  gradient_calls++;
  *dx += 13.0 * *dy;
}

/* Enzyme's registration ABI for reverse mode: {primal, augmented, gradient}.
 * The -passes=preserve-nvvm run turns this global into enzyme_augment and
 * enzyme_gradient metadata on fsquare before the enzyme pass looks at it. */
void *__enzyme_register_gradient_fsquare[] = {
    (void *)fsquare,
    (void *)augment_fsquare,
    (void *)gradient_fsquare,
};

int enzyme_augment_calls(void) { return augment_calls; }
int enzyme_gradient_calls(void) { return gradient_calls; }
