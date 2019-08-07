//#define EIGEN_USE_BLAS 0
//#define EIGEN_USE_LAPACKE 0
//#define EIGEN_USE_LAPACKE_STRICT 0
//#define EIGEN_USE_LAPACK 0
//#define EIGEN_USE_LAPACK_STRICT 0
#define EIGEN_UNROLLING_LIMIT 0
#define EIGEN_DONT_VECTORIZE 1
#include <eigen3/Eigen/Dense>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

static inline float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

#define MNIST_LABEL_MAGIC 0x00000801
#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_SIZE MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT
#define MNIST_LABELS 10

using Eigen::MatrixXd;
using Eigen::VectorXd;

#if 1
__attribute__((noinline))
static void matvecclean(const MatrixXd& __restrict W,
    const MatrixXd& __restrict b, MatrixXd& __restrict foo) {

  foo = W * b;
}

__attribute__((noinline))
static void matvec(const MatrixXd& __restrict W,
    const VectorXd& __restrict b, VectorXd& __restrict foo) {
    //printf("foo.rows()=%ld foo.cols()=%ld\n", foo.rows(), foo.cols());
    //printf("b.rows()=%ld b.cols()=%ld\n", b.rows(), b.cols());
    //printf("W.rows()=%ld W.cols()=%ld\n", W.rows(), W.cols());
  foo = W * b;
    //printf("r foo.rows()=%ld foo.cols()=%ld\n", foo.rows(), foo.cols());
    //printf("r b.rows()=%ld b.cols()=%ld\n", b.rows(), b.cols());
    //printf("r W.rows()=%ld W.cols()=%ld\n", W.rows(), W.cols());
    //printf("ran forward\n");
}

/*
__attribute__((noinline))
static void matvec(const MatrixXd& __restrict W,
    const MatrixXd& __restrict b, MatrixXd& __restrict foo) {

  for(int n=0; n<NUM; n++)

  for (int o = 0; o < OUT; o++)
  {
    output[n*OUT + o] = b(o);

    for (int i = 0; i < IN; i++) {
      output[n*OUT + o] += W(o, i) * (double)(input[n].pixels[i] / 255.);
    }
  }

}
*/

int main(int argc, char** argv) {

    size_t IN = 40, OUT = 30, NUM = 50;

    MatrixXd W(IN, OUT);
    MatrixXd Wp(IN, OUT);

    VectorXd B(OUT);
    VectorXd Bp(OUT);

    VectorXd foo(IN);
    VectorXd foop(IN);
    
    W = Eigen::MatrixXd::Constant(IN, OUT, 1.0);
    B = Eigen::VectorXd::Constant(OUT, 2.0);
    foo = Eigen::VectorXd::Constant(IN, 1.0);
    
    Wp = Eigen::MatrixXd::Constant(IN, OUT, 0.0);
    Bp = Eigen::VectorXd::Constant(OUT, 0.0);
    foop = Eigen::VectorXd::Constant(IN, 1.0);
      //memset(Wp, 0, sizeof(double) * IN * OUT);
      //memset(Bp, 0, sizeof(double) * OUT);

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  matvec(W, B, foo);

  gettimeofday(&end, NULL);
  printf("forward %0.6f res=%f\n", tdiff(&start, &end), Bp(0));
  }
  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  __builtin_autodiff(matvec, W,Wp,B,Bp,foo,foop);

  gettimeofday(&end, NULL);
  printf("diff %0.6f res=%f\n", tdiff(&start, &end), Bp(0));
  }

/*
    printf("running regular\n");
    //matvec(W, B, foo);
    printf("end running regular\n");
     __builtin_autodiff(matvec, W,Wp,B,Bp,foo,foop);

     for(int o=0; o<OUT; o++)
        printf("Bp(o=%d)=%f\n", o, Bp(o));

     for(int o=0; o<OUT; o++)
     for(int i=0; i<IN; i++)
        printf("Wp(o=%d, i=%d)=%f\n", i, o, Wp(i, o));
*/
}
#endif

#if 0
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

    labels = (uint8_t*)malloc(*number_of_labels * sizeof(uint8_t));

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
    images = (mnist_image_t*)malloc(*number_of_images * sizeof(mnist_image_t));

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

    dataset = (mnist_dataset_t*)calloc(1, sizeof(mnist_dataset_t));

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

void mnist_free_dataset(mnist_dataset_t * dataset)
{
    free(dataset->images);
    free(dataset->labels);
    free(dataset);
}

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

__attribute__((noinline))
static double conv_layer_manual(size_t IN, size_t OUT, size_t NUM, const MatrixXd& __restrict W,
    const MatrixXd& __restrict b, const mnist_image_t* __restrict input,
    const uint8_t* __restrict true_output) {

  double* output = (double*)malloc(sizeof(double)*NUM*OUT);//{0};

  #pragma clang loop unroll(disable)
  for(int n=0; n<NUM; n++)

  #pragma clang loop unroll(disable)
  for (int o = 0; o < OUT; o++)
  {
    output[n*OUT + o] = b(o);

    #pragma clang loop unroll(disable)
    for (int i = 0; i < IN; i++) {
      output[n*OUT + o] += W(o, i) * (double)(input[n].pixels[i] / 255.);
    }
  }


  double sum = 0;
  #pragma clang loop unroll(disable)
  for(int n=0; n<NUM; n++)
  #pragma clang loop unroll(disable)
  for(int o=0; o<OUT; o++) {
    double foo = (o == true_output[n]) ? 1.0 : 0.0;
    sum += (output[n*OUT+o] - foo) * (output[n*OUT+o] - foo);
  }

  free(output);
  return sum / NUM;

}

__attribute__((noinline))
static double conv_layer(size_t IN, size_t OUT, size_t NUM, const MatrixXd& __restrict W,
    const MatrixXd& __restrict b, const mnist_image_t* __restrict input,
    const uint8_t* __restrict true_out) {

  //double* output = (double*)malloc(sizeof(double)*NUM*OUT);//{0};
  //NUM = 1;
  //OUT = 10;
  IN = 784;
  MatrixXd out(NUM, OUT);

  #pragma clang loop unroll(disable)
  for(int n=0; n<NUM; n++)

  #pragma clang loop unroll(disable)
  for (int o = 0; o < OUT; o++)
  {
    out(n, o) = 0;//b(o);

    #pragma clang loop unroll(disable)
    for (int i = 0; i < IN; i++) {
      out(n, o) += W(o, i) * (double)(input[n].pixels[i] / 255.);
    }
  }


  double sum = 0;
  #pragma clang loop unroll(disable)
  for(int n=0; n<NUM; n++)
  #pragma clang loop unroll(disable)
  for(int o=0; o<OUT; o++) {
    double foo = (o == true_out[n]) ? 1.0 : 0.0;
    sum += (out(n,o) - foo) ;//* (out(n,o) - foo);
  }

  //free(output);
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

    MatrixXd W(OUT, IN);
    MatrixXd Wp(OUT, IN);

    VectorXd B(OUT);
    VectorXd Bp(OUT);

    VectorXd foo(IN);
    VectorXd foop(IN);

    double* input  = (double*)malloc(sizeof(double)*IN*NUM);
    double* inputp = (double*)malloc(sizeof(double)*IN*NUM);

    double* output  = (double*)malloc(sizeof(double)*OUT*NUM);
    double* outputp = (double*)malloc(sizeof(double)*OUT*NUM);

    for(int i=0; i<IN*NUM; i++) {
      input[i] = 2.;
    }

    for(int o=0; o<OUT; o++) {
    for(int i=0; i<IN; i++) {
      W(o, i) = (double)rand() / RAND_MAX;
    }
    }

    for(int i=0; i<OUT; i++) {
      B(i) = (double)rand() / RAND_MAX;
    }

    for(int i=0; i<OUT*NUM; i++) {
      output[i] = 3.;
    }

    double rate = -0.0001;
    printf("train dataset size=%d\n", train_dataset->size);
    while(1) {
      Wp = Eigen::MatrixXd::Constant(OUT, IN, 0.0);
      Bp = Eigen::VectorXd::Constant(OUT, 0.0);
      //memset(Wp, 0, sizeof(double) * IN * OUT);
      //memset(Bp, 0, sizeof(double) * OUT);

      size_t size;
      size = train_dataset->size;
      size = 1;
      double loss = 0;

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

      loss = conv_layer(IN,OUT,size,W,B,train_dataset->images,train_dataset->labels);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), loss);
  }


      double dloss = 0;

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

     dloss = __builtin_autodiff(W,Wp,B,Bp,foo,foop);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f\n", tdiff(&start, &end), dloss);
  }
      //dloss = __builtin_autodiff(conv_layer,IN,OUT,size,W,Wp,B,Bp,train_dataset->images,train_dataset->labels);
      B += rate * Bp;
      W += rate * Wp;
      
      /*
      for (int o = 0; o < OUT; o++) {
        B(o) += rate * Bp(o);
        for (int i = 0; i < IN; i++) {
            W(o, i) += rate * Wp(o, i);
        }
      }
      */
      
      printf("dloss = %f loss=%f\n", dloss, loss);
    }

}
#endif

#if 0
static
__attribute__((noinline))
double add(const MatrixXd& __restrict W) {
  //  return W(0, 0);
  double sum = 0;


  #pragma clang loop unroll(disable)
  for (int r = 0; r < W.rows(); r++) {

  #pragma clang loop unroll(disable)
  for (int c = 0; c < W.cols(); c++) {

      sum += W(r, c);

  }
  }

  return sum;
}

int main() {

    size_t ROW = 10, COL = 10;

    MatrixXd W (ROW, COL);
    MatrixXd Wp(ROW, COL);
    Wp = Eigen::MatrixXd::Constant(ROW, COL, 0.0);

    W(0, 0) = 10;

    printf("total = %f\n", add(W));

    __builtin_autodiff(add,W, Wp);

      //printf("W'(%d, %d)=%f\n", 0, 0, Wp(0, 0));
      
  #pragma clang loop unroll(disable)
  for (int r = 0; r < W.rows(); r++) {

  #pragma clang loop unroll(disable)
  for (int c = 0; c < W.cols(); c++) {

      printf("W'(%d, %d)=%f\n", r, c, Wp(r, c));

  }
  }

}
#endif
