#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "matrix.h"
#include "neural_net.h"

#define input_i 4
#define input_j 3

float XOR_values[input_i][input_j] = {
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,0}
};


int main(){
    srand(11);

    size_t arch[] = {2, 2, 1};
    Neural_Network nn = nn_alloc(arch, ARRAY_LEN(arch));
    Neural_Network g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_fill_with_random(nn, 0, 1);
    //nn_forward(nn);

    Matrix XOR_matrix = matrix_initialize(4,3);
    matrix_add_data(XOR_matrix, XOR_values);
    

    Matrix X = matrix_initialize(XOR_matrix.rows, 2);
    matrix_choose_columns(X, XOR_matrix, 0, 2);
    //PRINT_MATRIX_WITH_NAME(X);

    Matrix Y = matrix_initialize(XOR_matrix.rows, 1);
    matrix_choose_columns(Y, XOR_matrix, 2, 3);
    //PRINT_MATRIX_WITH_NAME(Y);

    float loss = 0;

    loss = nn_loss_function(nn, X, Y);

    //PRINT_NN_WITH_NAME(nn);


    printf("0: Start loss: %f\n", loss);

    float epsilon = 1e-1;
    float rate = 1e-1;
    size_t max_iter = 20000; 

    for(size_t i = 0; i < max_iter; i++){
        nn_finite_difference(nn, g, X, Y ,epsilon);
        nn_learn(nn, g, rate);
    }
    

    loss = nn_loss_function(nn, X, Y);
    printf("%zu: After loss: %f\n", max_iter, loss);

    

    printf("---------------------\n");

    //verification
    for(size_t i = 0; i < X.rows; i++){
        matrix_choose_rows(nn.as[0], X, i, i+1); //choose sample

        //pick out induvidual features
        float x1 = MATRIX_AT(nn.as[0], 0, 0); 
        float x2 = MATRIX_AT(nn.as[0], 0, 1);

        nn_forward(nn); //drive the sample through the network

        printf("%f ^ %f: %f\n", x1, x2, MATRIX_AT(nn.as[nn.count], 0, 0)); //print input with corresponding output
    }   

    printf("---------------------\n");
    
    PRINT_NN_WITH_NAME(nn);
    nn_free(nn);
    nn_free(g);

//LAG SUBMATRTIX FUNKSJON (COL, ROW)

    return 0;    
}