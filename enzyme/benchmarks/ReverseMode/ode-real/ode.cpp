#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>
#include <adept_source.h>
#include <adept.h>
#include <adept_arrays.h>
using adept::adouble;
using adept::aVector;

template<typename Return, typename... T>
Return __enzyme_autodiff(T...);
extern "C" {
  extern int enzyme_dup;
  extern int enzyme_const;
  extern int enzyme_dupnoneed;
}

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

#include <iostream>
using namespace std;

#define N 32
#define xmin 0.
#define xmax 1.
#define ymin 0.
#define ymax 1.

#include <assert.h>
#define RANGE(min, max, i, N) ((max-min)/(N-1)*i + min)
#define GETnb(x, i, j) (x)[N*i+j]
#define GET(x, i, j) GETnb(x, i, j)
//#define GET(x, i, j) ({ assert(i >=0); assert( j>=0); assert(j<N); assert(j<N); GETnb(x, i, j); })

template <typename T>
T brusselator_f(T x, T y, T t) {
  bool eq1 = ((x-0.3)*(x-0.3) + (y-0.6)*(y-0.6)) <= 0.1*0.1;
  bool eq2 = t >= 1.1;
  if (eq1 && eq2) {
    return T(5);
  } else {
    return T(0);
  }
}

void init_brusselator(double* __restrict u, double* __restrict v) {
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {

      double x = RANGE(xmin, xmax, i, N);
      double y = RANGE(ymin, ymax, j, N);

      GETnb(u, i, j) = 22*(y*(1-y))*sqrt(y*(1-y));
      GETnb(v, i, j) = 27*(x*(1-x))*sqrt(x*(1-x));
    }
  }
}

__attribute__((noinline))
void brusselator_2d_loop_restrict(double* __restrict du, double* __restrict dv, const double* __restrict u, const double* __restrict v, const double* __restrict p, double t) {
  double A = p[0];
  double B = p[1];
  double alpha = p[2];
  double dx = (double)1/(N-1);

  alpha = alpha/(dx*dx);

  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {

      double x = RANGE(xmin, xmax, i, N);
      double y = RANGE(ymin, ymax, j, N);

      unsigned ip1 = (i == N-1) ? i : (i+1);
      unsigned im1 = (i == 0) ? i : (i-1);

      unsigned jp1 = (j == N-1) ? j : (j+1);
      unsigned jm1 = (j == 0) ? j : (j-1);

      double u2v = GET(u, i, j) * GET(u, i, j) * GET(v, i, j);

      GETnb(du, i, j) = alpha*( GET(u, im1, j) + GET(u, ip1, j) + GET(u, i, jp1) + GET(u, i, jm1) - 4 * GET(u, i, j))
                      + B + u2v - (A + 1)*GET(u, i, j) + brusselator_f(x, y, t);

      GETnb(dv, i, j) = alpha*( GET(v, im1, j) + GET(v, ip1, j) + GET(v, i, jp1) + GET(v, i, jm1) - 4 * GET(v, i, j))
                      + A * GET(u, i, j) - u2v;
    }
  }
}

__attribute__((noinline))
void brusselator_2d_loop_norestrict(double* du, double* dv, const double* u, const double* v, const double* p, double t) {
  double A = p[0];
  double B = p[1];
  double alpha = p[2];
  double dx = (double)1/(N-1);

  alpha = alpha/(dx*dx);

  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {

      double x = RANGE(xmin, xmax, i, N);
      double y = RANGE(ymin, ymax, j, N);

      unsigned ip1 = (i == N-1) ? i : (i+1);
      unsigned im1 = (i == 0) ? i : (i-1);

      unsigned jp1 = (j == N-1) ? j : (j+1);
      unsigned jm1 = (j == 0) ? j : (j-1);

      double u2v = GET(u, i, j) * GET(u, i, j) * GET(v, i, j);

      GETnb(du, i, j) = alpha*( GET(u, im1, j) + GET(u, ip1, j) + GET(u, i, jp1) + GET(u, i, jm1) - 4 * GET(u, i, j))
                      + B + u2v - (A + 1)*GET(u, i, j) + brusselator_f(x, y, t);

      GETnb(dv, i, j) = alpha*( GET(v, im1, j) + GET(v, ip1, j) + GET(v, i, jp1) + GET(v, i, jm1) - 4 * GET(v, i, j))
                      + A * GET(u, i, j) - u2v;
    }
  }
}

typedef double state_type[2*N*N];

void lorenz_norestrict( const state_type &x, state_type &dxdt, double t )
{
    // Extract the parameters
  double p[3] = { /*A*/ 3.4, /*B*/ 1, /*alpha*/10. };
  brusselator_2d_loop_norestrict(dxdt, dxdt + N * N, x, x + N * N, p, t);
}

void lorenz_restrict( const state_type &x, state_type &dxdt, double t )
{
    // Extract the parameters
  double p[3] = { /*A*/ 3.4, /*B*/ 1, /*alpha*/10. };
  brusselator_2d_loop_restrict(dxdt, dxdt + N * N, x, x + N * N, p, t);
}

extern "C" void rust_lorenz_safe(const double* x, double* dxdt, double t);
extern "C" void rust_dbrusselator_2d_loop_safe(double* adjoint, const double* x, double* dx, const double* p, double* dp, double t);
extern "C" void rust_lorenz_unsf(const double* x, double* dxdt, double t);
extern "C" void rust_dbrusselator_2d_loop_unsf(double* adjoint, const double* x, double* dx, const double* p, double* dp, double t);

double rustfoobar_unsf(const double *p, const state_type x, const state_type adjoint, double t) {
  double dp[3] = { 0. };

  state_type dx = { 0. };

  state_type dadjoint_inp;// = adjoint
  for (int i = 0; i < N * N; i++) {
    dadjoint_inp[i] = adjoint[i];
  }

  rust_dbrusselator_2d_loop_unsf(dadjoint_inp, x, dx, p, dp, t);
  return dx[0];
}

double rustfoobar_safe(const double *p, const state_type x, const state_type adjoint, double t) {
  double dp[3] = { 0. };

  state_type dx = { 0. };

  state_type dadjoint_inp;// = adjoint
  for (int i = 0; i < N * N; i++) {
    dadjoint_inp[i] = adjoint[i];
  }

  rust_dbrusselator_2d_loop_safe(dadjoint_inp, x, dx, p, dp, t);
  return dx[0];
}

double foobar_restrict(const double* p, const state_type x, const state_type adjoint, double t) {
    double dp[3] = { 0. };

    state_type dx = { 0. };

    state_type dadjoint_inp;// = adjoint
    for (int i = 0; i < N * N; i++) {
      dadjoint_inp[i] = adjoint[i];
    }

    state_type dxdu;

    __enzyme_autodiff<void>(brusselator_2d_loop_restrict,
                            enzyme_dup, dxdu, dadjoint_inp,
                            enzyme_dup, dxdu + N * N, dadjoint_inp + N * N,
 //                           enzyme_dupnoneed, nullptr, dadjoint_inp,
 //                           enzyme_dupnoneed, nullptr, dadjoint_inp + N * N,
                            enzyme_dup, x, dx,
                            enzyme_dup, x + N * N, dx + N * N,
                            enzyme_dup, p, dp,
                            enzyme_const, t);

    return dx[0];
}

double foobar_norestrict(const double* p, const state_type x, const state_type adjoint, double t) {
    double dp[3] = { 0. };

    state_type dx = { 0. };

    state_type dadjoint_inp;// = adjoint
    for (int i = 0; i < N * N; i++) {
      dadjoint_inp[i] = adjoint[i];
    }

    state_type dxdu;

    __enzyme_autodiff<void>(brusselator_2d_loop_norestrict,
                            enzyme_dup, dxdu, dadjoint_inp,
                            enzyme_dup, dxdu + N * N, dadjoint_inp + N * N,
 //                           enzyme_dupnoneed, nullptr, dadjoint_inp,
 //                           enzyme_dupnoneed, nullptr, dadjoint_inp + N * N,
                            enzyme_dup, x, dx,
                            enzyme_dup, x + N * N, dx + N * N,
                            enzyme_dup, p, dp,
                            enzyme_const, t);

    return dx[0];
}

#undef GETnb
#define GETnb(x, i, j) (x)(N*i+j)

void abrusselator_2d_loop(aVector& du, aVector& dv, aVector& u, aVector& v, aVector& p, double t) {
  adouble A = p(0);
  adouble B = p(1);
  adouble alpha = p(2);
  adouble dx = (double)1/(N-1);

  alpha = alpha/(dx*dx);

  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {

      adouble x = RANGE(xmin, xmax, i, N);
      adouble y = RANGE(ymin, ymax, j, N);

      unsigned ip1 = (i == N-1) ? i : (i+1);
      unsigned im1 = (i == 0) ? i : (i-1);

      unsigned jp1 = (j == N-1) ? j : (j+1);
      unsigned jm1 = (j == 0) ? j : (j-1);

      adouble u2v = GET(u, i, j) * GET(u, i, j) * GET(v, i, j);

      GETnb(du, i, j) = alpha*( GET(u, im1, j) + GET(u, ip1, j) + GET(u, i, jp1) + GET(u, i, jm1) - 4 * GET(u, i, j))
                      + B + u2v - (A + 1)*GET(u, i, j) + brusselator_f<adouble>(x, y, t);

      GETnb(dv, i, j) = alpha*( GET(v, im1, j) + GET(v, ip1, j) + GET(v, i, jp1) + GET(v, i, jm1) - 4 * GET(v, i, j))
                      + A * GET(u, i, j) - u2v;
    }
  }
}

double afoobar(const double* p_in, const state_type x, const state_type adjoint, double t) {
    adept::Stack stack;

    aVector p(3);
    for(unsigned i=0; i<3; i++) p(i) = p_in[i];
    aVector ax(N*N);
    aVector ay(N*N);
    for(unsigned i=0; i<N*N; i++) {
      ax(i) = x[i];
      ay(i) = x[i+N*N];
    }

    aVector dxdu(N*N);
    aVector dydu(N*N);

    stack.new_recording();

    abrusselator_2d_loop(dxdu, dydu, ax, ay, p, t);

    for(unsigned i=0; i<N*N; i++) {
      dxdu(i).set_gradient(adjoint[i]);
      dydu(i).set_gradient(adjoint[i+N*N]);
    }
    stack.compute_adjoint();

    return ax(0).get_gradient();
}


//! Tapenade
extern "C" {
  /*        Generated by TAPENADE     (INRIA, Ecuador team)
    Tapenade 3.15 (master) -  8 Jan 2020 10:48
*/
#include <adBuffer.h>

/*
  Differentiation of get in reverse (adjoint) mode (with options i4 dr8 r4):
   gradient     of useful results: *x get
   with respect to varying inputs: *x
   Plus diff mem management of: x:in
*/
void get_b(const double *x, double *xb, unsigned int i, unsigned int j, double getb)
{
    double get;
    xb[N*i + j] = xb[N*i + j] + getb;
}

double get_nodiff(const double *x, unsigned int i, unsigned int j) {
    return x[N*i + j];
}

double brusselator_f_nodiff(double x, double y, double t) {
    if ((x-0.3)*(x-0.3) + (y-0.6)*(y-0.6) <= 0.1*0.1 && t >= 1.1)
        return 5.0;
    else
        return 0.0;
}

#if 1
void brusselator_2d_loop_b(double *du, double *dub, double *dv, double *dvb,
        const double *u, double *ub, const double *v, double *vb, const double *p, double *pb,
        double t) {
    double A = p[0];
    double Ab = 0.0;
    double B = p[1];
    double Bb = 0.0;
    double alpha = p[2];
    double alphab = 0.0;
    double dx = (double)1/(N-1);
    alpha = alpha/(dx*dx);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double x = (xmax-xmin)/(N-1)*i + xmin;
            double y = (ymax-ymin)/(N-1)*j + ymin;
            unsigned int ip1 = (i == N - 1 ? i : i + 1);
            unsigned int im1 = (i == 0 ? i : i - 1);
            unsigned int jp1 = (j == N - 1 ? j : j + 1);
            unsigned int jm1 = (j == 0 ? j : j - 1);
            double u2v = u[N*i+j]*u[N*i+j]*v[N*i+j];
            double result1;
            pushInteger4(jm1);
            pushInteger4(jp1);
            pushInteger4(im1);
            pushInteger4(ip1);
        }
    *ub = 0.0;
    *vb = 0.0;
    alphab = 0.0;
    Ab = 0.0;
    Bb = 0.0;
    for (int i = N-1; i > -1; --i)
        for (int j = N-1; j > -1; --j) {
            double x;
            double y;
            unsigned int ip1;
            unsigned int im1;
            unsigned int jp1;
            unsigned int jm1;
            double u2v;
            double u2vb = 0.0;
            double result1;
            double temp;
            double tempb;
            popInteger4((int*)&ip1);
            popInteger4((int*)&im1);
            popInteger4((int*)&jp1);
            popInteger4((int*)&jm1);
            temp = u[N*i + j];
            alphab = alphab + (v[N*im1+j]+v[N*ip1+j]+v[N*i+jp1]+v[N*i+jm1]
                -4*v[N*i+j])*dvb[N*i+j] + (u[N*im1+j]+u[N*ip1+j]+u[N*i+
                jp1]+u[N*i+jm1]-4*u[N*i+j])*dub[N*i+j];
            tempb = alpha*dvb[N*i+j];
            Ab = Ab + u[N*i+j]*dvb[N*i+j] - u[N*i+j]*dub[N*i+j];
            ub[N*i + j] = ub[N*i + j] + A*dvb[N*i+j] - (A+1)*dub[N*i+j];
            u2vb = dub[N*i + j] - dvb[N*i + j];
            dvb[N*i + j] = 0.0;
            vb[N*im1 + j] = vb[N*im1 + j] + tempb;
            vb[N*ip1 + j] = vb[N*ip1 + j] + tempb;
            vb[N*i + jp1] = vb[N*i + jp1] + tempb;
            vb[N*i + jm1] = vb[N*i + jm1] + tempb;
            vb[N*i + j] = vb[N*i + j] + temp*temp*u2vb - 4*tempb;
            tempb = alpha*dub[N*i+j];
            Bb = Bb + dub[N*i + j];
            dub[N*i + j] = 0.0;
            ub[N*im1 + j] = ub[N*im1 + j] + tempb;
            ub[N*ip1 + j] = ub[N*ip1 + j] + tempb;
            ub[N*i + jp1] = ub[N*i + jp1] + tempb;
            ub[N*i + jm1] = ub[N*i + jm1] + tempb;
            ub[N*i + j] = ub[N*i + j] + 2*temp*v[N*i+j]*u2vb - 4*tempb;
        }
    alphab = alphab/(dx*dx);
    pb[2] = pb[2] + alphab;
    pb[1] = pb[1] + Bb;
    pb[0] = pb[0] + Ab;

}
#else
/*
  Differentiation of brusselator_2d_loop in reverse (adjoint) mode (with options i4 dr8 r4):
   gradient     of useful results: *du *dv
   with respect to varying inputs: *p *u *du *v *dv
   RW status of diff variables: *p:out *u:out *du:in-out *v:out
                *dv:in-out
   Plus diff mem management of: p:in u:in du:in v:in dv:in
*/
void brusselator_2d_loop_b(double *du, double *dub, double *dv, double *dvb,
        const double *u, double *ub, const double *v, double *vb, const double *p, double *pb,
        double t) {
    double A = p[0];
    double Ab = 0.0;
    double B = p[1];
    double Bb = 0.0;
    double alpha = p[2];
    double alphab = 0.0;
    double dx = (double)1/(N-1);
    alpha = alpha/(dx*dx);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double x = (xmax-xmin)/(N-1)*i + xmin;
            double y = (ymax-ymin)/(N-1)*j + ymin;
            unsigned int ip1 = (i == N - 1 ? i : i + 1);
            unsigned int im1 = (i == 0 ? i : i - 1);
            unsigned int jp1 = (j == N - 1 ? j : j + 1);
            unsigned int jm1 = (j == 0 ? j : j - 1);
            double u2v;
            double result1;
            double result2;
            double result3;
            double result4;
            double result5;
            double result6;
            double result7;
            result1 = get_nodiff(u, i, j);
            result2 = get_nodiff(u, i, j);
            result3 = get_nodiff(v, i, j);
            pushReal8(result1);
            result1 = get_nodiff(u, im1, j);
            pushReal8(result2);
            result2 = get_nodiff(u, ip1, j);
            pushReal8(result3);
            result3 = get_nodiff(u, i, jp1);
            result4 = get_nodiff(u, i, jm1);
            result5 = get_nodiff(u, i, j);
            result6 = get_nodiff(u, i, j);
            pushReal8(result1);
            result1 = get_nodiff(v, im1, j);
            pushReal8(result2);
            result2 = get_nodiff(v, ip1, j);
            pushReal8(result3);
            result3 = get_nodiff(v, i, jp1);
            pushReal8(result4);
            result4 = get_nodiff(v, i, jm1);
            pushReal8(result5);
            result5 = get_nodiff(v, i, j);
            pushReal8(result6);
            result6 = get_nodiff(u, i, j);
            pushInteger4(jm1);
            pushInteger4(jp1);
            pushReal8(result6);
            pushReal8(result5);
            pushReal8(result4);
            pushReal8(result3);
            pushReal8(result2);
            pushReal8(result1);
            pushInteger4(im1);
            pushInteger4(ip1);
        }
    *ub = 0.0;
    *vb = 0.0;
    alphab = 0.0;
    Ab = 0.0;
    Bb = 0.0;
    for (int i = N-1; i > -1; --i)
        for (int j = N-1; j > -1; --j) {
            double x;
            double y;
            unsigned int ip1;
            unsigned int im1;
            unsigned int jp1;
            unsigned int jm1;
            double u2v;
            double u2vb;
            double result1;
            double result1b;
            double result2;
            double result2b;
            double result3;
            double result3b;
            double result4;
            double result4b;
            double result5;
            double result5b;
            double result6;
            double result6b;
            double result7;
            double tempb;
            popInteger4((int*)&ip1);
            popInteger4((int*)&im1);
            popReal8(&result1);
            popReal8(&result2);
            popReal8(&result3);
            popReal8(&result4);
            popReal8(&result5);
            popReal8(&result6);
            popInteger4((int*)&jp1);
            popInteger4((int*)&jm1);
            alphab = alphab + (result1+result2+result3+result4-4*result5)*dvb[
                N*i+j];
            tempb = alpha*dvb[N*i+j];
            Ab = Ab + result6*dvb[N*i+j];
            result6b = A*dvb[N*i+j];
            u2vb = dub[N*i + j] - dvb[N*i + j];
            dvb[N*i + j] = 0.0;
            result1b = tempb;
            result2b = tempb;
            result3b = tempb;
            result4b = tempb;
            result5b = -(4*tempb);
            popReal8(&result6);
            get_b(u, ub, i, j, result6b);
            popReal8(&result5);
            get_b(v, vb, i, j, result5b);
            popReal8(&result4);
            get_b(v, vb, i, jm1, result4b);
            popReal8(&result3);
            get_b(v, vb, i, jp1, result3b);
            popReal8(&result2);
            get_b(v, vb, ip1, j, result2b);
            popReal8(&result1);
            get_b(v, vb, im1, j, result1b);
            alphab = alphab + (result1+result2+result3+result4-4*result5)*dub[
                N*i+j];
            tempb = alpha*dub[N*i+j];
            Bb = Bb + dub[N*i + j];
            Ab = Ab - result6*dub[N*i+j];
            result6b = -((A+1)*dub[N*i+j]);
            dub[N*i + j] = 0.0;
            result1b = tempb;
            result2b = tempb;
            result3b = tempb;
            result4b = tempb;
            result5b = -(4*tempb);
            get_b(u, ub, i, j, result6b);
            get_b(u, ub, i, j, result5b);
            get_b(u, ub, i, jm1, result4b);
            popReal8(&result3);
            get_b(u, ub, i, jp1, result3b);
            popReal8(&result2);
            get_b(u, ub, ip1, j, result2b);
            popReal8(&result1);
            get_b(u, ub, im1, j, result1b);
            result1b = result2*result3*u2vb;
            result2b = result1*result3*u2vb;
            result3b = result1*result2*u2vb;
            get_b(v, vb, i, j, result3b);
            get_b(u, ub, i, j, result2b);
            get_b(u, ub, i, j, result1b);
        }
    alphab = alphab/(dx*dx);
    pb[2] = pb[2] + alphab;
    pb[1] = pb[1] + Bb;
    pb[0] = pb[0] + Ab;
}
#endif
}

double tfoobar(const double* p, const state_type x, const state_type adjoint, double t) {
    double dp[3] = { 0. };

    state_type dx = { 0. };

    state_type dadjoint_inp;// = adjoint
    for (int i = 0; i < N * N; i++) {
      dadjoint_inp[i] = adjoint[i];
    }

    state_type dxdu;

    brusselator_2d_loop_b(nullptr, dadjoint_inp,
                          nullptr, dadjoint_inp + N * N,
                          x, dx,
                          x + N * N, dx + N * N,
                          p, dp,
                          t);

    return dx[0];
}

//! Main
int main(int argc, char** argv) {
  const double p[3] = { /*A*/ 3.4, /*B*/ 1, /*alpha*/10. };

  state_type x;
  init_brusselator(x, x + N * N);

  state_type adjoint;
  init_brusselator(adjoint, adjoint + N * N);

  double t = 2.1;

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res;
  for(int i=0; i<10000; i++)
  res = afoobar(p, x, adjoint, t);

  gettimeofday(&end, NULL);
  printf("Adept combined %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res;
  for(int i=0; i<10000; i++)
  res = tfoobar(p, x, adjoint, t);

  gettimeofday(&end, NULL);
  printf("Tapenade combined %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res;
  for(int i=0; i<10000; i++)
  res = foobar_norestrict(p, x, adjoint, t);

  gettimeofday(&end, NULL);
  printf("C++  Enzyme combined mayalias %0.6f res=%f\n", tdiff(&start, &end), res);
  }
  
  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res;
  for(int i=0; i<10000; i++)
  res = foobar_restrict(p, x, adjoint, t);

  gettimeofday(&end, NULL);
  printf("C++  Enzyme combined restrict %0.6f res=%f\n", tdiff(&start, &end), res);
  }
  
  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double res;
    for(int i=0; i<10000; i++)
    res = rustfoobar_safe(p, x, adjoint, t);

    gettimeofday(&end, NULL);
    printf("Rust Enzyme combined safe %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double res;
    for(int i=0; i<10000; i++)
    res = rustfoobar_unsf(p, x, adjoint, t);

    gettimeofday(&end, NULL);
    printf("Rust Enzyme combined unsf %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    state_type x2;

    for(int i=0; i<10000; i++) {
      lorenz_norestrict(x, x2, t);
    }

    gettimeofday(&end, NULL);
    printf("C++  fwd mayalias %0.6f res=%f\n", tdiff(&start, &end), x2[0]);
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    state_type x2;

    for(int i=0; i<10000; i++) {
      lorenz_restrict(x, x2, t);
    }

    gettimeofday(&end, NULL);
    printf("C++  fwd restrict %0.6f res=%f\n", tdiff(&start, &end), x2[0]);
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    state_type x2;

    for(int i=0; i<10000; i++)
    rust_lorenz_safe(x, x2, t);

    gettimeofday(&end, NULL);
    printf("Rust fwd safe %0.6f res=%f\n\n", tdiff(&start, &end), x2[0]);
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    state_type x2;

    for(int i=0; i<10000; i++)
    rust_lorenz_unsf(x, x2, t);

    gettimeofday(&end, NULL);
    printf("Rust fwd unsf %0.6f res=%f\n\n", tdiff(&start, &end), x2[0]);
  }

  //printf("res=%f\n", foobar(1000));
}
