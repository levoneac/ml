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
    return 0;
}


/*
int main(){
    srand(time(0));

    size_t arch[] = {7, 3, 2, 1};
    Neural_Network nn = nn_alloc(arch, ARRAY_LEN(arch));
    Neural_Network g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_fill_with_random(nn, 0, 1);
    //nn_forward(nn);


    
    csv data = csv_read("test_data/apples_full.csv", ";");

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

    Matrix X_train = matrix_initialize(train_size, 7);
    matrix_choose_columns(X_train, train, 0, 7);

    Matrix X_test = matrix_initialize(test_size, 7);
    matrix_choose_columns(X_test, test, 0, 7);

    Matrix Y_train = matrix_initialize(train_size, 1);
    matrix_choose_columns(Y_train, train, 7, 8);

    Matrix Y_test = matrix_initialize(test_size, 1);
    matrix_choose_columns(Y_test, test, 7, 8);

    matrix_free(train);
    matrix_free(test);
    matrix_free(imported);


    float loss = 0;

    loss = nn_loss_function(nn, X_train, Y_train);

    //PRINT_NN_WITH_NAME(nn);


    printf("0: Start loss: %f\n", loss);

    float epsilon = 1e-1;
    float rate = 1e-1;
    size_t max_iter = 20; 

    for(size_t i = 0; i < max_iter; i++){
        nn_finite_difference(nn, g, X_train, Y_train ,epsilon);
        nn_learn(nn, g, rate);
    }
    

    loss = nn_loss_function(nn, X_train, Y_train);
    printf("%zu: After loss: %f\n", max_iter, loss);

    

    printf("---------------------\n");

    float train_result = nn_evaluate_classification(nn, X_train, Y_train);
    float test_result = nn_evaluate_classification(nn, X_test, Y_test);

    printf("train score: %f, test score: %f\n", train_result, test_result);

    nn_free(nn);
    nn_free(g);

    matrix_free(X_train);
    matrix_free(Y_train);
    matrix_free(X_test);
    matrix_free(Y_test);

    return 0;    
}
*/