#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"
#include "neural_net.h"


//size_t architecture[] = {2, 2, 1};
//size_t count = sizeof(layers)/sizeof(layers[0]);
//Neural_Network nn_alloc(architecture, ARRAY_LEN(architecture));

Neural_Network nn_alloc(size_t *architecture, size_t architecture_count){
    Neural_Network nn;

    assert(architecture_count > 0); //if arch count is less than 1 you dont have inputs
    nn.count = architecture_count -1; //architecture_count also contains inputs

    nn.ws = malloc(sizeof(*nn.ws) * nn.count); //allocates memory for pointers to weigths matrix
    assert(nn.ws != NULL);

    nn.bs = malloc(sizeof(*nn.bs) * nn.count); //allocates memory for pointers to bias matrix
    assert(nn.bs != NULL);

    nn.as = malloc(sizeof(*nn.as) * (nn.count + 1)); //allocates memory for pointers to outputs matrix
    assert(nn.as != NULL);

    nn.as[0] = matrix_initialize(1, architecture[0]); //input data is always at index 0

    for(size_t i = 0; i < nn.count; i++){
        nn.ws[i] = matrix_initialize(nn.as[i].cols, architecture[i + 1]); //nn.as[i].cols is the outputs of the previous layer. architecture[i + 1] is this layers outputs (+1 since nn.as[0] is the input data)
        nn.bs[i] = matrix_initialize(1, architecture[i + 1]); //matrix with same size as the inputs to the layer, which always have 1 row and the same amount of columns as the layer
        nn.as[i+1] = matrix_initialize(1, architecture[i + 1]); //output will have the same arch as the bias
    }

    return nn;
}

void nn_print(Neural_Network nn, const char *name){
    printf("NETWORK: %s\n\n", name);

    char buf[256];

    snprintf(buf, sizeof(buf), "as:%d", 0);
    matrix_print(nn.as[0], buf);


    for(size_t i = 0; i < nn.count; i++){
        snprintf(buf, sizeof(buf), "ws:%zu", i); //writes a formatted string into the given buffer. Good for preventing overflows
        matrix_print(nn.ws[i], buf);

        snprintf(buf, sizeof(buf), "bs:%zu", i);
        matrix_print(nn.bs[i], buf);

        snprintf(buf, sizeof(buf), "as:%zu", i+1);
        matrix_print(nn.as[i + 1], buf);

        printf("\n");
    }
}

void nn_fill_with_random(Neural_Network nn, float min, float max){
    for(size_t i = 0; i < nn.count; i++){
        matrix_fill_with_random(nn.ws[i], min, max);
        matrix_fill_with_random(nn.bs[i], min, max);
    }
}

void nn_forward(Neural_Network nn){
    for(size_t i = 0; i < nn.count; i++){
        matrix_multiplication(nn.as[i + 1], nn.as[i], nn.ws[i]); //multiply the inputs with the weights and forward them to the next layer
        matrix_sum(nn.as[i + 1], nn.as[i + 1], nn.bs[i]);
        matrix_apply_sigmoid(nn.as[i + 1]);
    }

}
