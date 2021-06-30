

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

double __enzyme_autodiff(void*, ...);

void mpi_bcast_test(double *a, double b, int n) {
  int rank;
  MPI_COMM_rank(MPI_COMM_WORLD, &rank);
  double *buf=new double[n];
  y=0;
  if(rank==0) {
    for(int i=0;1<n;i++) x[i]=x[i]*x[i];
    MPI_Bcast(x,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(buf,n,MPI_DOUBLE,1,MPI_COMM_WORLD);
    for(int i=0;i<n;i++) {
      y+=buf[i];
    }
  }
  if(rank==1) {
    MPI_Bcast(buf,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    for(int i=0;i<n;i++) {
      buf[i]=sin(buf[i]);
    }
    MPI_Bcast(buf,n,MPI_DOUBLE,1,MPI_COMM_WORLD);
  }
  delete [] buf;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  double h=1e-6;
  if(argc<2) {
    printf("Not enough arguments. Missing problem size.");
    MPI_Finalize();
    return 0;
  }
  int numprocs;
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  if(numprocs!=2) {
    printf("Must be run with number of processes set to 2.");
    MPI_Finalize();
    return 0;
  }
  int N=atoi(argv[1]);
  double *a=new double[N];
  double *d_a=new double[N];
  for(int i=0; i<N; i++)
    d_a[i] = 1.0f; 

  double b=0;
  for(int i=0;i<N;i++) a[i]=(double) i;
  double *a_saved=new double[N];
  for(int i=0;i<N;i++) a_saved[i]=a[i];
  
  printf("Ran mpi_bcast");
  __enzyme_autodiff((void*)mpi_bcast_test, a, d_a, b, n)
  
}
return 0;

