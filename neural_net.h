#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "matrix.h"

#define ARRAY_LEN(arr) sizeof((arr))/sizeof((arr)[0])
#define PRINT_NN_WITH_NAME(nn) nn_print((nn), #nn)

typedef struct {
    size_t count;
    Matrix *ws;
    Matrix *bs;
    Matrix *as; //one element bigger than the others for the input Matrix

} Neural_Network;

Neural_Network nn_alloc(size_t *architecture, size_t architecture_count);
void nn_print(Neural_Network nn, const char *name);
void nn_fill_with_random(Neural_Network nn, float min, float max);
void nn_forward(Neural_Network nn);


#endif