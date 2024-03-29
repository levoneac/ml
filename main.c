#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "matrix.h"
#include "neural_net.h"
#include "read_csv.h"



int main(){
    srand(time(0));
    csv data = csv_read("test_data/apples_full.csv", ";");

    size_t arch[] = {data.n_cols-1, 5, 3, 1}; //8 bias + (7*5 + 5*3 + 3*1) weights = 8 + (35 + 15 + 3) = 61 parameters (biases(not first layer) + products of neurons in each pair of layers)
    Neural_Network nn = nn_alloc(arch, ARRAY_LEN(arch));
    Neural_Network g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_fill_with_random(nn, -1,1);
    //nn_forward(nn);


    
    

    Matrix imported = matrix_initialize(data.n_rows, data.n_cols);
    matrix_add_from_csv_import(imported, data);

    csv_free(data);

    //MAKE THIS A FUNCTION
    float train_size_percent = 0.8;
    size_t train_size = data.n_rows * train_size_percent;
    size_t test_size = data.n_rows - train_size;

    Matrix train = matrix_initialize(train_size, data.n_cols);
    matrix_choose_rows(train, imported, 0, train_size);

    Matrix test = matrix_initialize(test_size, data.n_cols);
    matrix_choose_rows(test, imported, train_size, imported.rows);

    Matrix X_train = matrix_initialize(train_size, data.n_cols-1);
    matrix_choose_columns(X_train, train, 0, data.n_cols-1);

    Matrix X_test = matrix_initialize(test_size, data.n_cols-1);
    matrix_choose_columns(X_test, test, 0, data.n_cols-1);

    Matrix Y_train = matrix_initialize(train_size, 1);
    matrix_choose_columns(Y_train, train, data.n_cols-1, data.n_cols);

    Matrix Y_test = matrix_initialize(test_size, 1);
    matrix_choose_columns(Y_test, test, data.n_cols-1, data.n_cols);

    matrix_free(train);
    matrix_free(test);
    matrix_free(imported);


    float loss = 0;

    loss = nn_loss_function(nn, X_train, Y_train);

    //PRINT_NN_WITH_NAME(nn);


    printf("0: Start loss: %f\n", loss);

    float epsilon = 1e-4; //too big and you jump too much per iteration. too small will take too long to reach minimum but more precise
    float rate = 1e-1;
    size_t max_iter = 25000; //epochs

    for(size_t i = 0; i < max_iter; i++){
        nn_finite_difference(nn, g, X_train, Y_train, epsilon);
        nn_learn(nn, g, rate);
    }
    

    loss = nn_loss_function(nn, X_train, Y_train);
    printf("%zu: After loss: %f\n", max_iter, loss);

    

    printf("---------------------\n");

    float train_result = nn_evaluate_classification(nn, X_train, Y_train);
    float test_result = nn_evaluate_classification(nn, X_test, Y_test);

    printf("train score: %f, test score: %f\n", train_result, test_result);
    PRINT_NN_WITH_NAME(nn);

    nn_free(nn);
    nn_free(g);

    matrix_free(X_train);
    matrix_free(Y_train);
    matrix_free(X_test);
    matrix_free(Y_test);

    return 0;    
}

//a multivariable gradient is a vector pointing in the direction of greatest ascent
