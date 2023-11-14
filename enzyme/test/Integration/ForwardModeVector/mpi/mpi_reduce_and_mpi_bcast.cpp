#include <stdio.h>
#include <math.h>
#include <mpi.h>

// Takes derivative inputs, returns resulting derivative in return of dtwoNorm:
extern void __enzyme_fwddiff(void*, ...);


extern int enzyme_width;;

/* Vector 2-norm function, set up to permute existing arrays for vector norms.
*/
void twoNorm(double *x, double *norm, int n, int rank, int size){

    norm[0] = 0;
    for (int i=0; i<n; i++){
        norm[0] += x[i]*x[i];
    }

    //
    // Send partial sums to process 0, then compute the square root, then zeros out the partial sums on other processes
    //

    double tmp;
    // Using MPI_Reduce to reduce the partial sums into process 0, and take the square root
    MPI_Reduce(norm, &tmp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    norm[0] = tmp;
    norm[0] = sqrt(norm[0]);

    // Let's broadcast this norm from process 0 to all the other processes:
    // Note that this will make d_Norm equal on all processes, instead of zero everywhere but process 0
    MPI_Bcast(norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int dtwoNorm_forward(double *x, double **d_x, double *norm, double **d_norm, int n, int rank, int size){

    // Version with vector mode:
    __enzyme_fwddiff((void*) twoNorm, enzyme_width, 2, x, d_x[0], d_x[1], norm, d_norm[0], d_norm[1], n, rank, size);

    return 0;
}

int main(int argc, char* argv[]){

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 16;

    //
    // Here we allocate the local portion of each array:
    //
    int n_loc    = n / size;
    double* x    = (double*) malloc(n_loc*sizeof(double));
    double* norm = (double*) malloc(sizeof(double));

    for (int i=0; i<n_loc; i++){
        x[i] = (double) rank*n_loc + i + 1;
    }

    //
    // Here we compute the total norm, stored in process 0:
    //
    twoNorm(x, norm, n_loc, rank, size);

    if (rank < size){
        printf("Process %d: norm of the range from 1 to %d is %f\n", rank, n, norm[0]);
    }

    //
    // Here we use Enzyme
    //
    int width = 2;
    double** d_x    = (double**) malloc(width*sizeof(double*));
    double** d_norm = (double**) malloc(width*sizeof(double*));
    // Using calloc here to ensure d_x, d_norm are filled with 0s
    for (int i=0; i<width; i++){
        d_x[i]    = (double*) calloc(n_loc, sizeof(double));
        d_norm[i] = (double*) calloc(1, sizeof(double));
    }

    // Adding a few nonzero entries to the beginning of d_x:
    if (rank == 0){
        d_x[0][0] = 1;
        d_x[0][1] = 1;
        d_x[1][1] = 1;
    }
    dtwoNorm_forward(x, d_x, norm, d_norm, n_loc, rank, size);

    printf("Process %d: first derivative of this norm in x is", rank);
    for (int i=0; i<n_loc; i++){
        printf(" %f,", d_x[0][i]);
    }
    
    printf("\nProcess %d: second derivative of this norm in x is", rank);
    for (int i=0; i<n_loc; i++){
        printf(" %f,", d_x[1][i]);
    }
    printf("\n");
    printf("Process %d: output of dtwoNorms is %f, %f\n", rank, d_norm[0][0], d_norm[1][0]);
    
    free(x);
    free(d_x);
    
    MPI_Finalize();
    return 0;
}