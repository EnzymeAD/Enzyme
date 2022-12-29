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

struct LSTMInput {
  int l;
  int c;
  int b;
  std::vector<double> main_params;
  std::vector<double> extra_params;
  std::vector<double> state;
  std::vector<double> sequence;
};

struct LSTMOutput {
  double objective;
  std::vector<double> gradient;
};

template<unsigned vector_width>
void dlstm_objective_vec(int l, int c, int b,
                     double const *main_params, double *dmain_params, int main_params_size,
                     double const *extra_params, double *dextra_params, int extra_params_size,
                     double *state,
                     double const *sequence, double *loss, double *dloss);

template<unsigned NBDirsMax>
void lstm_objective_dv(int l, int c, int b,
                       const double *main_params, double *main_paramsd, int main_params_size,
                       const double *extra_params, double *extra_paramsd, int extra_params_size,
                       double *state,
                       const double *sequence,
                       double *loss, double *lossd);

extern "C" {
void dlstm_objective(int l, int c, int b, double const *main_params,
                     double *dmain_params, double const *extra_params,
                     double *dextra_params, double *state,
                     double const *sequence, double *loss, double *dloss);

void lstm_objective_d(int l, int c, int b, const double *main_params,
                      double *main_paramsb, const double *extra_params,
                      double *extra_paramsb, double *state,
                      const double *sequence, double *loss, double *lossb);

void adept_dlstm_objective(int l, int c, int b, double const *main_params,
                           double *dmain_params, double const *extra_params,
                           double *dextra_params, double *state,
                           double const *sequence, double *loss, double *dloss);
}

void read_lstm_instance(const string &fn, int *l, int *c, int *b,
                        vector<double> &main_params,
                        vector<double> &extra_params, vector<double> &state,
                        vector<double> &sequence) {
  FILE *fid = fopen(fn.c_str(), "r");

  if (!fid) {
    printf("could not open file: %s\n", fn.c_str());
    exit(1);
  }

  fscanf(fid, "%i %i %i", l, c, b);

  int l_ = *l, c_ = *c, b_ = *b;

  int main_sz = 2 * l_ * 4 * b_;
  int extra_sz = 3 * b_;
  int state_sz = 2 * l_ * b_;
  int seq_sz = c_ * b_;

  main_params.resize(main_sz);
  extra_params.resize(extra_sz);
  state.resize(state_sz);
  sequence.resize(seq_sz);

  for (int i = 0; i < main_sz; i++) {
    fscanf(fid, "%lf", &main_params[i]);
  }

  for (int i = 0; i < extra_sz; i++) {
    fscanf(fid, "%lf", &extra_params[i]);
  }

  for (int i = 0; i < state_sz; i++) {
    fscanf(fid, "%lf", &state[i]);
  }

  for (int i = 0; i < c_ * b_; i++) {
    fscanf(fid, "%lf", &sequence[i]);
  }

  /*char ch;
  fscanf(fid, "%c", &ch);
  fscanf(fid, "%c", &ch);

  for (int i = 0; i < c_; i++) {
      unsigned char ch;
      fscanf(fid, "%c", &ch);
      int cb = ch;
      for (int j = b_ - 1; j >= 0; j--) {
          int p = pow(2, j);
          if (cb >= p) {
              sequence[(i + 1) * b_ - j - 1] = 1;
              cb -= p;
          }
          else {
              sequence[(i + 1) * b_ - j - 1] = 0;
          }
      }
  }*/

  fclose(fid);
}

typedef void (*deriv_t)(int l, int c, int b, double const *main_params,
                        double *dmain_params, double const *extra_params,
                        double *dextra_params, double *state,
                        double const *sequence, double *loss, double *dloss);


template<unsigned vector_width>
using vec_deriv_t = decltype(dlstm_objective_vec<vector_width>);

template <deriv_t deriv>
void calculate_jacobian(struct LSTMInput &input, struct LSTMOutput &result) {
  unsigned nparams = input.main_params.size() + input.extra_params.size();
  
  for (int i = 0; i < 100; i++) {
    std::vector<double> dparams(nparams);
    double *main_params_gradient_part = dparams.data();
    double *extra_params_gradient_part =
        dparams.data() + input.main_params.size();

    for (unsigned n = 0;
         n < input.main_params.size() + input.extra_params.size(); ++n) {
      dparams[n] = 1.0;

      double loss = 0.0; // stores fictive result
                         // (Tapenade doesn't calculate an original function in
                         // reverse mode)

      double *lossb = result.gradient.data() + n;

      deriv(input.l, input.c, input.b, input.main_params.data(),
            main_params_gradient_part, input.extra_params.data(),
            extra_params_gradient_part, input.state.data(),
            input.sequence.data(), &loss, lossb);

      dparams[n] = 0.0;
    }
  }
}

template<unsigned vector_width, deriv_t scalar_deriv, vec_deriv_t<vector_width> vec_deriv>
void calculate_jacobian_vec(struct LSTMInput &input, struct LSTMOutput &result) {
  unsigned nparams = input.main_params.size() + input.extra_params.size();
  
  for (int it = 0; it < 100; it++) {
    std::vector<double> dparams(nparams * vector_width);

    double *main_params_gradient_part = dparams.data();
    double *extra_params_gradient_part = dparams.data() + input.main_params.size() * vector_width;
    double *lossb = result.gradient.data();

    unsigned nscalar_params = nparams % vector_width;
    unsigned nvector_params = nparams - nscalar_params;

  for (unsigned i = 0; i < nvector_params; i+=vector_width) {

    for (int n = 0; n < vector_width; ++n) {
      if (i + n < input.main_params.size()) {
        main_params_gradient_part[i + n + n * input.main_params.size()] = 1.0;
      } else {
        extra_params_gradient_part[(i - input.main_params.size()) + n + n * input.extra_params.size()] = 1.0;
      }
    }

        double loss = 0.0; // stores fictive result
                         // (Tapenade doesn't calculate an original function in
                         // reverse mode)

      vec_deriv(input.l, input.c, input.b,
                          input.main_params.data(), main_params_gradient_part, input.main_params.size(),
                          input.extra_params.data(), extra_params_gradient_part, input.extra_params.size(),
                          input.state.data(),
                          input.sequence.data(), 
                          &loss, lossb + i);

    for (int n = 0; n < vector_width; ++n) {
      if (i + n < input.main_params.size()) {
        main_params_gradient_part[i + n + n * input.main_params.size()] = 0.0;
      } else {
        extra_params_gradient_part[(i - input.main_params.size()) + n + n * input.extra_params.size()] = 0.0;
      }
    }
    }

  for (unsigned i = nvector_params; i < nparams; i+=1) {
      if (i < input.main_params.size()) {
        main_params_gradient_part[i] = 1.0;
      } else {
        extra_params_gradient_part[(i - input.main_params.size())] = 1.0;
      }

      double loss = 0.0; // stores fictive result
                         // (Tapenade doesn't calculate an original function in
                         // reverse mode)

      scalar_deriv(input.l, input.c, input.b, 
                          input.main_params.data(), main_params_gradient_part,
                          input.extra_params.data(), extra_params_gradient_part,
                          input.state.data(),
                          input.sequence.data(), 
                          &loss, lossb + i);

      if (i < input.main_params.size()) {
        main_params_gradient_part[i] = 0.0;
      } else {
        extra_params_gradient_part[(i - input.main_params.size())] = 0.0;
      }
  }
  }
}

int main(const int argc, const char *argv[]) {
  printf("starting main\n");

  std::vector<std::string> paths = {"lstm_l2_c1024.txt", "lstm_l4_c1024.txt",
                                    "lstm_l2_c4096.txt", "lstm_l4_c4096.txt"};

  for (auto path : paths) {
    printf("starting path %s\n", path.c_str());

    {
      struct LSTMInput input;

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      for (unsigned i = 0; i < 5; i++) {
        printf("%f ", input.state[i]);
      }
      printf("\n");

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian<lstm_objective_d>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

    {
      struct LSTMInput input;

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<2, lstm_objective_d, lstm_objective_dv<2>>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade vector 2 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

       {
      struct LSTMInput input;

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<4, lstm_objective_d, lstm_objective_dv<4>>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade vector 4 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
       }

       {
      struct LSTMInput input;

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<8, lstm_objective_d, lstm_objective_dv<8>>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade vector 8 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
       }

       {
      struct LSTMInput input;

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<16, lstm_objective_d, lstm_objective_dv<16>>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade vector 16 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
       }

       {
      struct LSTMInput input;

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<32, lstm_objective_d, lstm_objective_dv<32>>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade vector 32 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
       }

       {
      struct LSTMInput input;

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<64, lstm_objective_d, lstm_objective_dv<64>>(input, result);
        gettimeofday(&end, NULL);
        printf("Tapenade vector 64 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

//    {
//
//      struct LSTMInput input = {};
//
//      // Read instance
//      read_lstm_instance("../../data/lstm" + path, &input.l, &input.c, &input.b,
//                         input.main_params, input.extra_params, input.state,
//                         input.sequence);
//
//      std::vector<double> state = std::vector<double>(input.state.size());
//
//      int Jcols = 8 * input.l * input.b + 3 * input.b;
//      struct LSTMOutput result = {0, std::vector<double>(Jcols)};
//
//      {
//        struct timeval start, end;
//        gettimeofday(&start, NULL);
//        calculate_jacobian<adept_dlstm_objective>(input, result);
//        gettimeofday(&end, NULL);
//        printf("Adept combined %0.6f\n", tdiff(&start, &end));
//        for (unsigned i = result.gradient.size() - 5;
//             i < result.gradient.size(); i++) {
//          printf("%f ", result.gradient[i]);
//        }
//        printf("\n");
//      }
//    }

    // Enzyme Forward-Mode
    {

      struct LSTMInput input = {};

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian<dlstm_objective>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme Forward combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }

    // Enzyme Forward-Vector Mode
    {
      struct LSTMInput input = {};

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<2, dlstm_objective, dlstm_objective_vec<2>>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme Forward-Vector 2 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }
    
    {
      struct LSTMInput input = {};

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<4, dlstm_objective, dlstm_objective_vec<4>>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme Forward-Vector 4 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }
    
    {
      struct LSTMInput input = {};

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<8, dlstm_objective, dlstm_objective_vec<8>>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme Forward-Vector 8 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }
    
    {
      struct LSTMInput input = {};

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<16, dlstm_objective, dlstm_objective_vec<16>>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme Forward-Vector 16 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }
    
    {
      struct LSTMInput input = {};

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<32, dlstm_objective, dlstm_objective_vec<32>>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme Forward-Vector 32 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }
    
    {
      struct LSTMInput input = {};

      // Read instance
      read_lstm_instance("../../data/lstm/" + path, &input.l, &input.c, &input.b,
                         input.main_params, input.extra_params, input.state,
                         input.sequence);

      std::vector<double> state = std::vector<double>(input.state.size());

      int Jcols = 8 * input.l * input.b + 3 * input.b;
      struct LSTMOutput result = {0, std::vector<double>(Jcols)};

      {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        calculate_jacobian_vec<64, dlstm_objective, dlstm_objective_vec<64>>(input, result);
        gettimeofday(&end, NULL);
        printf("Enzyme Forward-Vector 64 combined %0.6f\n", tdiff(&start, &end));
        for (unsigned i = result.gradient.size() - 5;
             i < result.gradient.size(); i++) {
          printf("%f ", result.gradient[i]);
        }
        printf("\n");
      }
    }
    
  }
}
