void gmm_objective(size_t d, size_t k, size_t n, double const *alphas,
                   double const *means, double const *icf, double const *x,
                   Wishart wishart, double *err) {
  size_t ix, ik;
  const double CONSTANT = -(double)n * d * 0.5 * log(2 * PI);
  size_t icf_sz = d * (d + 1) / 2;

  double *Qdiags = (double *)malloc(d * k * sizeof(double));
  double *sum_qs = (double *)malloc(k * sizeof(double));
  double *xcentered = (double *)malloc(d * sizeof(double));
  double *Qxcentered = (double *)malloc(d * sizeof(double));
  double *main_term = (double *)malloc(k * sizeof(double));

  preprocess_qs(d, k, icf, &sum_qs[0], &Qdiags[0]);

  double slse = 0.;
  for (ix = 0; ix < n; ix++) {
    for (ik = 0; ik < k; ik++) {
      subtract(d, &x[ix * d], &means[ik * d], &xcentered[0]);
      Qtimesx(d, &Qdiags[ik * d], &icf[ik * icf_sz + d], &xcentered[0],
              &Qxcentered[0]);
      // two caches for qxcentered at idx 0 and at arbitrary index
      main_term[ik] = alphas[ik] + sum_qs[ik] - 0.5 * sqnorm(d, &Qxcentered[0]);
    }

    // storing cmp for max of main_term
    // 2 x (0 and arbitrary) storing sub to exp
    // storing sum for use in log
    slse = slse + log_sum_exp(k, &main_term[0]);
  }

  // storing cmp of alphas
  double lse_alphas = log_sum_exp(k, alphas);

  *err = CONSTANT + slse - n * lse_alphas +
         log_wishart_prior(d, k, wishart, &sum_qs[0], &Qdiags[0], icf);

  free(Qdiags);
  free(sum_qs);
  free(xcentered);
  free(Qxcentered);
  free(main_term);
}

// *      tapenade -b -o gmm_tapenade -head "gmm_objective(err)/(alphas means icf)" gmm.c
void dgmm_objective(size_t d, size_t k, size_t n, const double *alphas, double *
        alphasb, const double *means, double *meansb, const double *icf,
        double *icfb, const double *x, Wishart wishart, double *err, double *
        errb) {
    __enzyme_autodiff(
            gmm_objective,
            enzyme_const, d,
            enzyme_const, k,
            enzyme_const, n,
            enzyme_dup, alphas, alphasb,
            enzyme_dup, means, meansb,
            enzyme_dup, icf, icfb,
            enzyme_const, x,
            enzyme_const, wishart,
            enzyme_dupnoneed, err, errb);
}

