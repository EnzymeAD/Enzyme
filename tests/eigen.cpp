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

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

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

using Eigen::MatrixXd;
using Eigen::VectorXd;

__attribute__((noinline))
static double conv_layer(size_t IN, size_t OUT, size_t NUM, const MatrixXd& __restrict W,
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
      output[n*OUT + o] += W(i, o) * (double)(input[n].pixels[i] / 255.);
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
      //size = 100;
      double loss =

      conv_layer(IN,OUT,size,W,B,train_dataset->images,train_dataset->labels);
      double dloss = 0;

      dloss = __builtin_autodiff((void*)conv_layer,IN,OUT,size,W,Wp,B,Bp,train_dataset->images,train_dataset->labels);
      for (int o = 0; o < OUT; o++) {
        B(o) += rate * Bp(o);
        for (int i = 0; i < IN; i++) {
            W(o, i) += rate * Wp(o, i);
        }
      }
      printf("dloss = %f loss=%f\n", dloss, loss);
    }

}




