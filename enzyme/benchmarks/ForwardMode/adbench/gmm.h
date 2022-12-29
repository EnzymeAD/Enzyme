#pragma once

#include "../mshared/defs.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <vector>
float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

using namespace std;

struct GMMInput {
  int d, k, n;
  std::vector<double> alphas, means, icf, x;
  Wishart wishart;
};

struct GMMOutput {
  double objective;
  std::vector<double> gradient;
};

struct GMMParameters {
  bool replicate_point;
};

template<unsigned vector_width>
void dgmm_objective_vec(int d, int k, int n, 
                    const double *alphas, double *alphasb, int alphas_size,
                    const double *means, double *meansb, int means_size,
                    const double *icf, double *icfb, int icf_size,
                    const double *x, Wishart wishart, 
                    double *err, double *errb);

template<unsigned NBDirsMax>
void gmm_objective_dv(int d, int k, int n,
                      const double *alphas, double* alphasd, int alphas_size,
                      const double *means, double* meansd, int means_size,
                      const double *icf, double* icfd, int icf_size,
                      const double *x, Wishart wishart, 
                      double *err, double* errd);

extern "C" {
void dgmm_objective(int d, int k, int n, const double *alphas, double *alphasb,
                    const double *means, double *meansb, const double *icf,
                    double *icfb, const double *x, Wishart wishart, double *err,
                    double *errb);

void gmm_objective_d(int d, int k, int n, const double *alphas, double *alphasb,
                     const double *means, double *meansb, const double *icf,
                     double *icfb, const double *x, Wishart wishart,
                     double *err, double *errb);

void adept_dgmm_objective(int d, int k, int n, const double *alphas,
                          double *alphasb, const double *means, double *meansb,
                          const double *icf, double *icfb, const double *x,
                          Wishart wishart, double *err, double *errb);
}

void read_gmm_instance(const string &fn, int *d, int *k, int *n,
                       vector<double> &alphas, vector<double> &means,
                       vector<double> &icf, vector<double> &x, Wishart &wishart,
                       bool replicate_point) {
  FILE *fid = fopen(fn.c_str(), "r");

  if (!fid) {
    printf("could not open file: %s\n", fn.c_str());
    exit(1);
  }

  fscanf(fid, "%i %i %i", d, k, n);

  int d_ = *d, k_ = *k, n_ = *n;

  int icf_sz = d_ * (d_ + 1) / 2;
  alphas.resize(k_);
  means.resize(d_ * k_);
  icf.resize(icf_sz * k_);
  x.resize(d_ * n_);

  for (int i = 0; i < k_; i++) {
    fscanf(fid, "%lf", &alphas[i]);
  }

  for (int i = 0; i < k_; i++) {
    for (int j = 0; j < d_; j++) {
      fscanf(fid, "%lf", &means[i * d_ + j]);
    }
  }

  for (int i = 0; i < k_; i++) {
    for (int j = 0; j < icf_sz; j++) {
      fscanf(fid, "%lf", &icf[i * icf_sz + j]);
    }
  }

  if (replicate_point) {
    for (int j = 0; j < d_; j++) {
      fscanf(fid, "%lf", &x[j]);
    }
    for (int i = 0; i < n_; i++) {
      memcpy(&x[i * d_], &x[0], d_ * sizeof(double));
    }
  } else {
    for (int i = 0; i < n_; i++) {
      for (int j = 0; j < d_; j++) {
        fscanf(fid, "%lf", &x[i * d_ + j]);
      }
    }
  }

  fscanf(fid, "%lf %i", &(wishart.gamma), &(wishart.m));

  fclose(fid);
}

typedef void (*deriv_t)(int d, int k, int n, const double *alphas,
                        double *alphasb, const double *means, double *meansb,
                        const double *icf, double *icfb, const double *x,
                        Wishart wishart, double *err, double *errb);

template<unsigned vector_width>
using vec_deriv_t = decltype(dgmm_objective_vec<vector_width>);

template <deriv_t deriv>
void calculate_jacobian(struct GMMInput &input, struct GMMOutput &result) {
  unsigned nparams =
      input.alphas.size() + input.means.size() + input.icf.size();
  std::vector<double> params(nparams);

  double *alphas_gradient_part = params.data();
  double *means_gradient_part = params.data() + input.alphas.size();
  double *icf_gradient_part =
      params.data() + input.alphas.size() + input.means.size();

  double tmp =
      0.0; // stores fictive result
           // (Tapenade doesn't calculate an original function in reverse mode)

  for (unsigned n = 0; n < nparams; ++n) {
    params[n] = 1.0;

    double *dparams = result.gradient.data() + n;

    deriv(input.d, input.k, input.n, input.alphas.data(), alphas_gradient_part,
          input.means.data(), means_gradient_part, input.icf.data(),
          icf_gradient_part, input.x.data(), input.wishart, &tmp, dparams);

    params[n] = 0.0;
  }
}

template<unsigned vector_width, deriv_t scalar_deriv, vec_deriv_t<vector_width> vec_deriv>
void calculate_jacobian_vec(struct GMMInput &input, struct GMMOutput &result) {
  unsigned nparams = input.alphas.size() + input.means.size() + input.icf.size();

  std::vector<double> dparams(nparams * vector_width);

  double *alphas_gradient_part = dparams.data();
  double *means_gradient_part = dparams.data() + input.alphas.size() * vector_width;
  double *icf_gradient_part = dparams.data() + (input.alphas.size() + input.means.size()) * vector_width;
  double *errord = result.gradient.data();
  
  unsigned nscalar_params = nparams % vector_width;
  unsigned nvector_params = nparams - nscalar_params;

  for (unsigned i = 0; i < nvector_params; i+=vector_width) {
    
    for (int n = 0; n < vector_width; ++n) {
      if (i + n < input.alphas.size()) {
        alphas_gradient_part[i + n + n * input.alphas.size()] = 1.0;
      } else if (i + n < input.alphas.size() + input.means.size()) {
        means_gradient_part[(i - input.alphas.size()) + n + n * input.means.size()] = 1.0;
      } else {
        icf_gradient_part[(i - input.alphas.size() - input.means.size()) + n + n * input.icf.size()] = 1.0;
      }
    }

      double tmp = 0.0; // stores fictive result
           // (Tapenade doesn't calculate an original function in reverse mode)

    vec_deriv(input.d, input.k, input.n,
                    input.alphas.data(), alphas_gradient_part, input.alphas.size(),
                    input.means.data(), means_gradient_part, input.means.size(),
                    input.icf.data(), icf_gradient_part, input.icf.size(),
                    input.x.data(),
                    input.wishart,
                    &tmp, errord + i);

    for (int n = 0; n < vector_width; ++n) {
      if (i + n < input.alphas.size()) {
        alphas_gradient_part[i + n + n * input.alphas.size()] = 0.0;
      } else if (i + n < input.alphas.size() + input.means.size()) {
        means_gradient_part[(i - input.alphas.size()) + n + n * input.means.size()] = 0.0;
      } else {
        icf_gradient_part[(i - input.alphas.size() - input.means.size()) + n + n * input.icf.size()] = 0.0;
      }
    }
  }

  for (unsigned i = nvector_params; i < nparams; i+=1) {
      if (i < input.alphas.size()) {
        alphas_gradient_part[i] = 1.0;
      } else if (i < input.alphas.size() + input.means.size()) {
        means_gradient_part[(i - input.alphas.size())] = 1.0;
      } else {
        icf_gradient_part[(i - input.alphas.size() - input.means.size())] = 1.0;
      }

          double tmp = 0.0; // stores fictive result
           // (Tapenade doesn't calculate an original function in reverse mode)

            scalar_deriv(input.d, input.k, input.n,
                    input.alphas.data(), alphas_gradient_part,
                    input.means.data(), means_gradient_part,
                    input.icf.data(), icf_gradient_part,
                    input.x.data(),
                    input.wishart,
                    &tmp, errord + i);

      if (i < input.alphas.size()) {
        alphas_gradient_part[i] = 0.0;
      } else if (i < input.alphas.size() + input.means.size()) {
        means_gradient_part[(i - input.alphas.size())] = 0.0;
      } else {
        icf_gradient_part[(i - input.alphas.size() - input.means.size())] = 0.0;
      }
  }
}

int main(const int argc, const char *argv[]) {
  printf("starting main\n");

  const auto replicate_point = (argc > 9 && string(argv[9]) == "-rep");
  const GMMParameters params = {replicate_point};

  std::vector<std::string> paths; // = { "1k/gmm_d10_K100.txt" };

  getTests(paths, "../../data/gmm/1k", "1k/");
  getTests(paths, "../../data/gmm/2.5k", "2.5k/");
  getTests(paths, "../../data/gmm/10k", "10k/");

  for (auto path : paths) {
      if (path == "10k/gmm_d128_K200.txt" || path == "10k/gmm_d128_K100.txt" ||
          path == "10k/gmm_d128_K25.txt" || path == "10k/gmm_d128_K50.txt" ||
          path == "10k/gmm_d64_K200.txt" || path == "10k/gmm_d64_K100.txt" ||
          path == "10k/gmm_d64_K50.txt" || path == "10k/gmm_d32_K200.txt"
          || path == "1k/gmm_d128_K200.txt")
        continue;
      printf("starting path %s\n", path.c_str());

      {

        struct GMMInput input;
        read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k,
                          &input.n, input.alphas, input.means, input.icf,
                          input.x, input.wishart, params.replicate_point);

        int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

        struct GMMOutput result = {0, std::vector<double>(Jcols)};

        {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian<gmm_objective_d>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = 0; i < 5; i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

    {
      struct GMMInput input;
      read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k, &input.n,
                        input.alphas, input.means, input.icf, input.x,
                        input.wishart, params.replicate_point);

      int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

      struct GMMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<2, gmm_objective_d, gmm_objective_dv<2>>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade vector 2 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = 0; i < 5; i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

      {
      struct GMMInput input;
      read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k, &input.n,
                        input.alphas, input.means, input.icf, input.x,
                        input.wishart, params.replicate_point);

      int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

      struct GMMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<4, gmm_objective_d, gmm_objective_dv<4>>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade vector 4 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = 0; i < 5; i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

      {
      struct GMMInput input;
      read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k, &input.n,
                        input.alphas, input.means, input.icf, input.x,
                        input.wishart, params.replicate_point);

      int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

      struct GMMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<8, gmm_objective_d, gmm_objective_dv<8>>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade vector 8 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = 0; i < 5; i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

      {
      struct GMMInput input;
      read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k, &input.n,
                        input.alphas, input.means, input.icf, input.x,
                        input.wishart, params.replicate_point);

      int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

      struct GMMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<16, gmm_objective_d, gmm_objective_dv<16>>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade vector 16 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = 0; i < 5; i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

      {
      struct GMMInput input;
      read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k, &input.n,
                        input.alphas, input.means, input.icf, input.x,
                        input.wishart, params.replicate_point);

      int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

      struct GMMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<32, gmm_objective_d, gmm_objective_dv<32>>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade vector 32 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = 0; i < 5; i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

    // {

    //   struct GMMInput input;
    //   read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k, &input.n,
    //                     input.alphas, input.means, input.icf, input.x,
    //                     input.wishart, params.replicate_point);

    //   int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

    //   struct GMMOutput result = {0, std::vector<double>(Jcols)};

    //   try {
    //     struct timeval start, end;
    //     gettimeofday(&start, NULL);
    //     calculate_jacobian<adept_dgmm_objective>(input, result);
    //     gettimeofday(&end, NULL);
    //     printf("Adept combined %0.6f\n", tdiff(&start, &end));
    //     for (unsigned i = 0; i < 5; i++) {
    //       printf("%f ", result.gradient[i]);
    //     }
    //     printf("\n");
    //   } catch (std::bad_alloc) {
    //     printf("Adept combined 88888888 ooms\n");
    //   }
    // }

    {

      struct GMMInput input;
      read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k, &input.n,
                        input.alphas, input.means, input.icf, input.x,
                        input.wishart, params.replicate_point);

      int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

      struct GMMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian<dgmm_objective>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme forward mode combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = 0; i < 5; i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

    // Enzyme Forward-Vector
    {
      struct GMMInput input;
      read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k, &input.n,
                        input.alphas, input.means, input.icf, input.x,
                        input.wishart, params.replicate_point);

      int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

      struct GMMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<2, dgmm_objective, dgmm_objective_vec<2>>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme Forward-Vector 2 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = 0; i < 5; i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

    {
      struct GMMInput input;
      read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k, &input.n,
                        input.alphas, input.means, input.icf, input.x,
                        input.wishart, params.replicate_point);

      int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

      struct GMMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<4, dgmm_objective, dgmm_objective_vec<4>>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme Forward-Vector 4 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = 0; i < 5; i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

    {
      struct GMMInput input;
      read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k, &input.n,
                        input.alphas, input.means, input.icf, input.x,
                        input.wishart, params.replicate_point);

      int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

      struct GMMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<8, dgmm_objective, dgmm_objective_vec<8>>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme Forward-Vector 8 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = 0; i < 5; i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

    {
      struct GMMInput input;
      read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k, &input.n,
                        input.alphas, input.means, input.icf, input.x,
                        input.wishart, params.replicate_point);

      int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

      struct GMMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<16, dgmm_objective, dgmm_objective_vec<16>>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme Forward-Vector 16 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = 0; i < 5; i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

    {
      struct GMMInput input;
      read_gmm_instance("../../data/gmm/" + path, &input.d, &input.k, &input.n,
                        input.alphas, input.means, input.icf, input.x,
                        input.wishart, params.replicate_point);

      int Jcols = (input.k * (input.d + 1) * (input.d + 2)) / 2;

      struct GMMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<32, dgmm_objective, dgmm_objective_vec<32>>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme Forward-Vector 32 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = 0; i < 5; i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }
  }
}
