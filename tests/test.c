#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

#if 0

/*
__attribute__((noinline))
double times2(double x) {
    return 2 * sqrt(x);
}

double squarea(double x) {
    return x * times2(x);
}

double min(double x, double y) {
    return (x < y) ? x : y;
}

double relu(double x) {
    return (x > 0) ? x : 0;
}

double squa(double x) {
    return relu(min(x, x*x/2));
}

double squarez(double x) {
    if (x > 0) {
        //return cos(x * sin(x));
        return sqrt(x * sin(x));
    } else {
        return 0;
    }
}

__attribute__((noinline))
double ptr(double* __restrict x) {
    return (*x) * (*x);
}
*/

/*
#define LEN 3
__attribute__((noinline))
double sumsquare(double* __restrict x ) {
    double sum = 0;
    for(int i=0; i < LEN; i++) {
        //sum += x[i];
        sum += x[i] * x[i];
    }
    return sum;
}

double square(double x) {
   double ar[LEN] = {x};
   return sumsquare(ar);
}
*/


/*
__attribute__((noinline))
double times(double x, int y) {
    return x * y;
}

double square(double x) {
    return times(x, 2);
}
*/

/*
__attribute__((noinline))
double sumsquare(double* __restrict x, int n) {
    double sum = 0;
    #pragma clang loop vectorize(disable)
    #pragma clang loop unroll(disable)
    for(int i=0; i < n; i++) {
        //sum += x[i];
        //printf("running iteration %d\n", i);
        sum += x[i] * x[i];
    }
    //printf("returning sum\n");
    return sum;
}

double square(double x) {
   int n = 6;
   double ar[6] = {1, x, x*x, x*x*x, x*x*x*x, x*x*x*x*x};//, x*x*x*x*x*x};
   return sumsquare(ar, n);
}
*/

/*
__attribute__((noinline))
double foo(double* __restrict matrix) {//, double* __restrict vector) {
  printf("begin foo\n");
  double output = 0;//{0};

  for (int idx = 0; idx < 10*10; idx++) {
    //printf("foo idx=%d\n", idx);
    int i = idx%10;
    int j = idx/10;
    output += matrix[j*10 + i]//vector[i];//matrix[j*100+i] + vector[i];
    ;
  }
  //printf("ended foo\n");
  return output;
}

double square(double x) {
  printf("starting square\n");
  double vector[10] = {0};
    //#pragma clang loop vectorize(disable)
    //#pragma clang loop unroll(disable)
  for (int i = 0; i < 10; i++) {
    vector[i] = (1.0*i)/10;
    //printf("vector[%d]\n", i);
  }
  //printf("set vector\n");
  double matrix_weights[10*10] = {0};

    //#pragma clang loop vectorize(disable)
    //#pragma clang loop unroll(disable)
  for (int idx = 0; idx < 10*10; idx++) {
    int i = idx%10;
    int j = idx/10;
    //printf("matrix[%d]\n", idx);
    matrix_weights[j*10+i] = 1.0*(j+i) + 1e-20;
  }

  printf("calling foo\n");
  return foo(matrix_weights);//, vector);
}
*/

/*
__attribute__((noinline))
double foo(double* __restrict matrix, double* __restrict vector, size_t len) {
  double output = 0;//{0};

  #pragma clang loop unroll(disable)
  for (int idx = 0; idx < len*len; idx++) {
    //printf("foo idx=%d\n", idx);
    int i = idx%len;
    int j = idx/len;
    output += matrix[j*len + i] + vector[i];
    ;
  }
  //printf("ended foo\n");
  return output;
}
*/


double square(double x) {
  #define len 100
  double vector[len] = {0};
  for (int i = 0; i < len; i++) {
    vector[i] = (1.0*i)/len
     //* x
     ;
  }
  double matrix_weights[len*len] = {0};

  for (int idx = 0; idx < len*len; idx++) {
    int i = idx%len;
    int j = idx/len;
    matrix_weights[j*len+i] =
    //x *
    1.0*(j+i) + 1e-20;
  }

  printf("calling foo\n");
  return foo(matrix_weights, vector, len);
}


__attribute__((noinline))
double foo(double* __restrict matrix, double* __restrict vector, size_t len) {
  double output = 0;//{0};

  #pragma clang loop unroll(disable)
  for (int i = 0; i < len*len; i++) {
    //printf("foo(%i) precond\n", i);
    //if (vector[i % len] > 0) {
      output += matrix[i];
      //printf("  foo(%i) incond\n", i);
    //}
    //printf("foo(%i) endcond\n", i);
    //else {
    //  output += matrix[i] * matrix[i];
    //}
  }
  //printf("ended foo\n");
  return output;
}

#endif

#if 0
__attribute__((noinline))
double foo(double* __restrict matrix, double* __restrict vector, size_t len) {
  double output = 0;//{0};
  printf("matrix[3]=%f\n", matrix[3]);

  #pragma clang loop unroll(disable)
  for (int i = 0; i < len; i++) {
    //printf("foo idx=%d\n", idx);
  #pragma clang loop unroll(disable)
  for (int j = 0; j < len; j++) {
    //printf("foo idx=(i=%d,j=%d)\n", i, j);
    double tmp = sqrt(matrix[i*len + j]);// + vector[i]);
    output += tmp;
    printf("looking at i=%d j=%d, matrix[i*len+j]=%f, sqrt(matrix[i*len+j])=%f\n", i, j,matrix[i*len + j],tmp);
  }
  }
  //printf("ended foo\n");
  printf("returning output\n");
  return output;
}

double square(double x) {
  #define len 5
  double vector[len] = {0};
  for (int i = 0; i < len; i++) {
    vector[i] = (1.0*i)/len
     + x
     ;
  }
  double matrix_weights[len*len] = {0};

  for (int idx = 0; idx < len*len; idx++) {
    int i = idx%len;
    int j = idx/len;
    matrix_weights[i*len+j] =
    x *
    1.0*(j+i) + 1e-20;
    //printf("looking at i=%d j=%d, matrix[i*len+j]=%f\n", i, j,matrix_weights[i*len + j]);
  }

  //printf("calling foo matrix_weights[3]=%f\n", matrix_weights[3]);
  return foo(matrix_weights, vector, len);
}

int main(int argc, char** argv) {
    double f = atof(argv[1]);

    printf("now executing square\n");
    double res0 = square(f);
    printf("finished executing square\n");
    printf("f(x=%lf) = %lf\n", f, res0);
    printf("now executing builtin autodiff\n");
    double res = __builtin_autodiff(square, f);
    printf("finished executing autodiff\n");
    printf("d/dx f(x=%lf) = %lf\n", f, res);
    //printf("d/dx sqrt(x) | x=%lf  = %lf | eval=%lf\n", f, __builtin_autodiff(ptr, f), ptr(f));
}

#endif

#if 0

static double f(double x) {
  for(int i=1; i<5; i++) {
    x = sin(cos(x));
  }
  return x;
}

__attribute__((noinline))
static double loop(double x, int n) {
  double r = x/x;

  #pragma clang loop unroll(disable)
  for(int i=1; i<n; i++) {
    r *= f(x);
  }
  return sin(cos(r));
}

static double test(double x) {
  return loop(x, 3);
}

__attribute__((noinline))
double logsumexp(double *x, int n) {
  double A = x[0];
  #pragma clang loop unroll(disable)
  #pragma clang loop vectorize(disable)
  for(int i=0; i<n; i++) {
    A = max(A, x[i]);
  }
  double ema[n];
  #pragma clang loop unroll(disable)
  #pragma clang loop vectorize(disable)
  for(int i=0; i<n; i++) {
    ema[i] = exp(x[i] - A);
  }
  double sema = 0;
  #pragma clang loop unroll(disable)
  #pragma clang loop vectorize(disable)
  for(int i=0; i<n; i++)
    sema += ema[i];
  return log(sema) + A;
}


double test2(double x) {
  double rands[100000];
  #pragma clang loop unroll(disable)
  #pragma clang loop vectorize(disable)
  for(int i=0; i<100000; i++) {
    rands[i] = i * x;
  }
  return logsumexp(rands, 100000);
}

/*
int main0(int argc, char** argv) {

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = test(2);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = __builtin_autodiff(test, 2);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f\n", tdiff(&start, &end), res);
  }
}
*/

int main(int argc, char** argv) {

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = test2(2);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = __builtin_autodiff(test2, 2);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f\n", tdiff(&start, &end), res);
  }
}

#endif

void matvec(size_t N, size_t M, double* mat, double* vec, double* out) {
  for(int i=0; i<N; i++) {
    out[i] = 0;
    for(int j=0; j<M; j++) {
        out[i] += mat[i*M+j] * vec[j];
    }
  }
}

int main(int argc, char** argv) {
  #define N 10ull
  #define M 4ull
  double mat[N*M];
  double matp[N*M];
  double vec[M];
  double vecp[M];
  double out[N];
  matvec(N, M, mat, vec, out);
  __builtin_autodiff(matvec, N, M, mat, matp, vec, vecp, out);
}

#if 0
double add(double a, double b) {
  return a + b;
}

int main(int argc, char** argv) {

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = add(2., 3.);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = __builtin_autodiff(add, 2., 3.);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f\n", tdiff(&start, &end), res);
  }
}

#endif

#if 0
static double max(double x, double y) {
    return (x > y) ? x : y;
}

__attribute__((noinline))
static double logsumexp(double *__restrict x, size_t n) {
  double A = x[0];
  #pragma clang loop unroll(disable)
  for(int i=0; i<n; i++) {
    A = max(A, x[i]);
  }
  double ema[n];
  #pragma clang loop unroll(disable)
  for(int i=0; i<n; i++) {
    ema[i] = exp(x[i] - A);
  }
  double sema = 0;
  #pragma clang loop unroll(disable)
  for(int i=0; i<n; i++)
    sema += ema[i];
  return log(sema) + A;
}

int main(int argc, char** argv) {

  size_t size = 100000;
  double* rands = (double*)malloc(sizeof(double)*size);
  double* randsp = (double*)malloc(sizeof(double)*size);

  for(int i=0; i<size; i++) {
    rands[i] = i;
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = logsumexp(rands, size);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = __builtin_autodiff(logsumexp, rands, randsp, size);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f\n", tdiff(&start, &end), res);
  }
}
#endif

#if 0

__attribute__((noinline))
double foo(double* __restrict matrix, double* __restrict vector, size_t len) {
  double output[len];//{0};

  #pragma clang loop unroll(disable)
  for (int i = 0; i < len; i++) {
    //printf("foo idx=%d\n", idx);
    output[i] = 0;
  #pragma clang loop unroll(disable)
  for (int j = 0; j < len; j++) {
    double tmp = matrix[i*len + j] * vector[j];
    output[i] += tmp;
  }
  }

  double sum = 0;
  #pragma clang loop unroll(disable)
  for(int i=0; i<len; i++) {
    sum += output[i];
  }
  return sum;
}

double square(double x) {
  #define len 100
  double vector[len] = {0};
  for (int i = 0; i < len; i++) {
    vector[i] = (1.0*i)/len
     + x
     ;
  }
  double matrix_weights[len*len] = {0};

  for (int idx = 0; idx < len*len; idx++) {
    int i = idx%len;
    int j = idx/len;
    matrix_weights[i*len+j] =
    x *
    1.0*(j+i) + 1e-20;
    //printf("looking at i=%d j=%d, matrix[i*len+j]=%f\n", i, j,matrix_weights[i*len + j]);
  }

  //printf("calling foo matrix_weights[3]=%f\n", matrix_weights[3]);
  return foo(matrix_weights, vector, len);
}

int main(int argc, char** argv) {
    double f = atof(argv[1]);

    printf("now executing square\n");
    double res0 = 0;

 {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  for(int i=0; i<200000; i++)
    res0 += square(f);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res0);
 }
    printf("finished executing square\n");
    printf("f(x=%lf) = %lf\n", f, res0);
    printf("now executing builtin autodiff\n");
    double res = 0;
   {
  struct timeval start, end;
  gettimeofday(&start, NULL);


  for(int i=0; i<200000; i++)
    res += __builtin_autodiff(square, f);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f\n", tdiff(&start, &end), res);
  }
    printf("finished executing autodiff\n");
    printf("d/dx f(x=%lf) = %lf\n", f, res);
    //printf("d/dx sqrt(x) | x=%lf  = %lf | eval=%lf\n", f, __builtin_autodiff(ptr, f), ptr(f));
}

#endif

#if 0 
#define MNIST_LABEL_MAGIC 0x00000801
#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_SIZE MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT
#define MNIST_LABELS 10

typedef struct mnist_label_file_header_t_ {
    uint32_t magic_number;
    uint32_t number_of_labels;
} __attribute__((packed)) mnist_label_file_header_t;

typedef struct mnist_image_file_header_t_ {
    uint32_t magic_number;
    uint32_t number_of_images;
    uint32_t number_of_rows;
    uint32_t number_of_columns;
} __attribute__((packed)) mnist_image_file_header_t;

typedef struct mnist_image_t_ {
    uint8_t pixels[MNIST_IMAGE_SIZE];
} __attribute__((packed)) mnist_image_t;

typedef struct mnist_dataset_t_ {
    mnist_image_t * images;
    uint8_t * labels;
    size_t size;
} mnist_dataset_t;

mnist_dataset_t * mnist_get_dataset(const char * image_path, const char * label_path);
void mnist_free_dataset(mnist_dataset_t * dataset);
int mnist_batch(mnist_dataset_t * dataset, mnist_dataset_t * batch, int batch_size, int batch_number);


/**
 * Convert from the big endian format in the dataset if we're on a little endian
 * machine.
 */
uint32_t map_uint32(uint32_t in)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (
        ((in & 0xFF000000) >> 24) |
        ((in & 0x00FF0000) >>  8) |
        ((in & 0x0000FF00) <<  8) |
        ((in & 0x000000FF) << 24)
    );
#else
    return in;
#endif
}

/**
 * Read labels from file.
 *
 * File format: http://yann.lecun.com/exdb/mnist/
 */
uint8_t * get_labels(const char * path, uint32_t * number_of_labels)
{
    FILE * stream;
    mnist_label_file_header_t header;
    uint8_t * labels;

    stream = fopen(path, "rb");

    if (NULL == stream) {
        fprintf(stderr, "Could not open file: %s\n", path);
        return NULL;
    }

    if (1 != fread(&header, sizeof(mnist_label_file_header_t), 1, stream)) {
        fprintf(stderr, "Could not read label file header from: %s\n", path);
        fclose(stream);
        return NULL;
    }

    header.magic_number = map_uint32(header.magic_number);
    header.number_of_labels = map_uint32(header.number_of_labels);

    if (MNIST_LABEL_MAGIC != header.magic_number) {
        fprintf(stderr, "Invalid header read from label file: %s (%08X not %08X)\n", path, header.magic_number, MNIST_LABEL_MAGIC);
        fclose(stream);
        return NULL;
    }

    *number_of_labels = header.number_of_labels;

    labels = malloc(*number_of_labels * sizeof(uint8_t));

    if (labels == NULL) {
        fprintf(stderr, "Could not allocated memory for %d labels\n", *number_of_labels);
        fclose(stream);
        return NULL;
    }

    if (*number_of_labels != fread(labels, 1, *number_of_labels, stream)) {
        fprintf(stderr, "Could not read %d labels from: %s\n", *number_of_labels, path);
        free(labels);
        fclose(stream);
        return NULL;
    }

    fclose(stream);

    return labels;
}

/**
 * Read images from file.
 *
 * File format: http://yann.lecun.com/exdb/mnist/
 */
mnist_image_t * get_images(const char * path, uint32_t * number_of_images)
{
    FILE * stream;
    mnist_image_file_header_t header;
    mnist_image_t * images;

    stream = fopen(path, "rb");

    if (NULL == stream) {
        fprintf(stderr, "Could not open file: %s\n", path);
        return NULL;
    }

    if (1 != fread(&header, sizeof(mnist_image_file_header_t), 1, stream)) {
        fprintf(stderr, "Could not read image file header from: %s\n", path);
        fclose(stream);
        return NULL;
    }

    header.magic_number = map_uint32(header.magic_number);
    header.number_of_images = map_uint32(header.number_of_images);
    header.number_of_rows = map_uint32(header.number_of_rows);
    header.number_of_columns = map_uint32(header.number_of_columns);

    if (MNIST_IMAGE_MAGIC != header.magic_number) {
        fprintf(stderr, "Invalid header read from image file: %s (%08X not %08X)\n", path, header.magic_number, MNIST_IMAGE_MAGIC);
        fclose(stream);
        return NULL;
    }

    if (MNIST_IMAGE_WIDTH != header.number_of_rows) {
        fprintf(stderr, "Invalid number of image rows in image file %s (%d not %d)\n", path, header.number_of_rows, MNIST_IMAGE_WIDTH);
    }

    if (MNIST_IMAGE_HEIGHT != header.number_of_columns) {
        fprintf(stderr, "Invalid number of image columns in image file %s (%d not %d)\n", path, header.number_of_columns, MNIST_IMAGE_HEIGHT);
    }

    *number_of_images = header.number_of_images;
    images = malloc(*number_of_images * sizeof(mnist_image_t));

    if (images == NULL) {
        fprintf(stderr, "Could not allocated memory for %d images\n", *number_of_images);
        fclose(stream);
        return NULL;
    }

    if (*number_of_images != fread(images, sizeof(mnist_image_t), *number_of_images, stream)) {
        fprintf(stderr, "Could not read %d images from: %s\n", *number_of_images, path);
        free(images);
        fclose(stream);
        return NULL;
    }

    fclose(stream);

    return images;
}

mnist_dataset_t * mnist_get_dataset(const char * image_path, const char * label_path)
{
    mnist_dataset_t * dataset;
    uint32_t number_of_images, number_of_labels;

    dataset = calloc(1, sizeof(mnist_dataset_t));

    if (NULL == dataset) {
        return NULL;
    }

    dataset->images = get_images(image_path, &number_of_images);

    if (NULL == dataset->images) {
        mnist_free_dataset(dataset);
        return NULL;
    }

    dataset->labels = get_labels(label_path, &number_of_labels);

    if (NULL == dataset->labels) {
        mnist_free_dataset(dataset);
        return NULL;
    }

    if (number_of_images != number_of_labels) {
        fprintf(stderr, "Number of images does not match number of labels (%d != %d)\n", number_of_images, number_of_labels);
        mnist_free_dataset(dataset);
        return NULL;
    }

    dataset->size = number_of_images;

    return dataset;
}

/**
 * Free all the memory allocated in a dataset. This should not be used on a
 * batched dataset as the memory is allocated to the parent.
 */
void mnist_free_dataset(mnist_dataset_t * dataset)
{
    free(dataset->images);
    free(dataset->labels);
    free(dataset);
}

/**
 * Fills the batch dataset with a subset of the parent dataset.
 */
int mnist_batch(mnist_dataset_t * dataset, mnist_dataset_t * batch, int size, int number)
{
    int start_offset;

    start_offset = size * number;

    if (start_offset >= dataset->size) {
        return 0;
    }

    batch->images = &dataset->images[start_offset];
    batch->labels = &dataset->labels[start_offset];
    batch->size = size;

    if (start_offset + batch->size > dataset->size) {
        batch->size = dataset->size - start_offset;
    }

    return 1;
}

static double conv_layer_old(size_t IN, size_t OUT, size_t NUM, const double* __restrict W, const double* __restrict b, const mnist_image_t* __restrict input, const uint8_t* __restrict true_output) {
  double* output = (double*)malloc(sizeof(double)*NUM*OUT);//{0};

  
  /*
  #pragma clang loop unroll(disable)
  for(int n=0; n<NUM; n++)
  #pragma clang loop unroll(disable)
  for (int o = 0; o < OUT; o++) {
    output[n*OUT + o] = b[o];
  }
  */

  #pragma clang loop unroll(disable)
  for(int n=0; n<NUM; n++)
  
  #pragma clang loop unroll(disable)
  for (int o = 0; o < OUT; o++) 
  {
    output[n*OUT + o] = b[o];

    #pragma clang loop unroll(disable)
    for (int i = 0; i < IN; i++) {
      output[n*OUT + o] += W[o*IN+i] * (double)(input[n].pixels[i] / 255.);
    }
    //printf("end op n:%d o:%d b[o]:%f %f\n",n, o, b[o], output[n*OUT+o]); 
  }

  
  double sum = 0;
  #pragma clang loop unroll(disable)
  for(int n=0; n<NUM; n++)
  #pragma clang loop unroll(disable)
  for(int o=0; o<OUT; o++) {
    double foo = (o == true_output[n]) ? 1.0 : 0.0;
    //printf("n:%d o:%d foo:%f op:%f\n", n, o, foo, output[n*OUT+o]);
    sum += (output[n*OUT+o] - foo) * (output[n*OUT+o] - foo);
  }

  //sum = output[0];
  free(output);
  return sum / NUM;
  
}

static double conv_layer(size_t IN, size_t OUT, size_t NUM, const double* __restrict W, const double* __restrict b, const mnist_image_t* __restrict input, const uint8_t* __restrict true_output) {
  double* output = (double*)malloc(sizeof(double)*NUM*OUT);//{0};
  double sum = 0;

  #pragma clang loop unroll(disable)
  for(int n=0; n<NUM; n++)
  
  #pragma clang loop unroll(disable)
  for (int o = 0; o < OUT; o++) 
  {
    output[n*OUT + o] = b[o];

    #pragma clang loop unroll(disable)
    for (int i = 0; i < IN; i++) {
      output[n*OUT + o] += W[o*IN+i] * (double)(input[n].pixels[i] / 255.);
    }
    double foo = (o == true_output[n]) ? 1.0 : 0.0;
    sum += (output[n*OUT+o] - foo) * (output[n*OUT+o] - foo);
  }

  free(output);
  return sum / NUM;
  
}
const char * train_images_file = "data/train-images-idx3-ubyte";
const char * train_labels_file = "data/train-labels-idx1-ubyte";
const char * test_images_file = "data/t10k-images-idx3-ubyte";
const char * test_labels_file = "data/t10k-labels-idx1-ubyte";

int main(int argc, char** argv) {
    //double f = atof(argv[1]);

    mnist_dataset_t * train_dataset, * test_dataset;
    mnist_dataset_t batch;
    float loss, accuracy;
    int i, batches;

    // Read the datasets from the files
    train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
    test_dataset = mnist_get_dataset(test_images_file, test_labels_file);

    size_t IN = 784, OUT = 10, NUM = 3;

    double* W  = (double*)malloc(sizeof(double)*IN*OUT);
    double* Wp = (double*)malloc(sizeof(double)*IN*OUT);

    double* B  = (double*)malloc(sizeof(double)*OUT);
    double* Bp = (double*)malloc(sizeof(double)*OUT);

    double* input  = (double*)malloc(sizeof(double)*IN*NUM);
    double* inputp = (double*)malloc(sizeof(double)*IN*NUM);

    double* output  = (double*)malloc(sizeof(double)*OUT*NUM);
    double* outputp = (double*)malloc(sizeof(double)*OUT*NUM);

    for(int i=0; i<IN*NUM; i++) {
      input[i] = 2.;
    }

    for(int i=0; i<IN*OUT; i++) {
      W[i] = (double)rand() / RAND_MAX;
    }

    for(int i=0; i<OUT; i++) {
      B[i] = (double)rand() / RAND_MAX;
    }

    for(int i=0; i<OUT*NUM; i++) {
      output[i] = 3.;
    }

    double rate = -0.0001;
    printf("train dataset size=%d\n", train_dataset->size);
    while(1) {
      memset(Wp, 0, sizeof(double) * IN * OUT);
      memset(Bp, 0, sizeof(double) * OUT);

      size_t size;
      size = train_dataset->size;
      //size = 100;
      double loss =
      
      conv_layer(IN,OUT,size,W,B,train_dataset->images,train_dataset->labels);
      double dloss = 0;
      
      dloss = __builtin_autodiff(conv_layer,IN,OUT,size,W,Wp,B,Bp,train_dataset->images,train_dataset->labels);
      for (int o = 0; o < OUT; o++) {
        B[o] += rate * Bp[o];
        //printf("Bp[%d]=%f\n", o, Bp[o]);
        for (int i = 0; i < IN; i++) {
            W[o*IN+i] += rate * Wp[o*IN + i];
            //printf("Wp[%d*IN+%d]=%f\n", o, i, Wp[o*IN+i]);
        }
      }
      printf("dloss = %f loss=%f\n", dloss, loss);
    }

}
#endif
