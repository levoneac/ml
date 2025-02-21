#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "matrix.h"

#define input_i 4
#define input_j 3
float XOR_values[input_i][input_j] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}};

float x_values[1][2] = {
    {1.0f, 2.0f}};

typedef struct
{
    Matrix a0; // x
    Matrix w1, b1, a1;
    Matrix w2, b2, a2;
} Xor;

Xor XOR_alloc(void)
{
    Xor model;

    model.a0 = matrix_initialize(1, 2);
    model.w1 = matrix_initialize(2, 2); // Huskeregel: left er input antall og right blir output antall med Matrix formen: 1*output
    model.b1 = matrix_initialize(1, 2);
    model.a1 = matrix_initialize(1, 2); // alltid 1 row på output
    model.w2 = matrix_initialize(2, 1);
    model.b2 = matrix_initialize(1, 1);
    model.a2 = matrix_initialize(1, 1);

    return model;
}

// Drives inputs through the model
float *forward_xor(Xor model)
{
    // float arr[][2] = {{x1, x2}};
    // matrix_add_data(model.a0, arr);
    // PRINT_MATRIX_WITH_NAME(model.a0);

    matrix_multiplication(model.a1, model.a0, model.w1);
    matrix_sum(model.a1, model.a1, model.b1);
    matrix_apply_sigmoid(model.a1);

    matrix_multiplication(model.a2, model.a1, model.w2);
    matrix_sum(model.a2, model.a2, model.b2);
    matrix_apply_sigmoid(model.a2);
    return model.a2.data;
}

float loss_function(Xor model, Matrix training_input, Matrix training_output)
{
    assert(training_input.rows == training_output.rows); // there has to be the same amount of data in both input and output
    assert(training_output.cols == model.a2.cols);       // output of model has to be same size as expected
    size_t input_rows = training_input.rows;
    size_t output_cols = training_output.cols;

    float loss = 0.f;
    Matrix y = matrix_initialize(1, training_output.cols);
    for (size_t i = 0; i < input_rows; i++)
    {                                                           // for each row in training input
        matrix_choose_rows(model.a0, training_input, i, i + 1); // x
        matrix_choose_rows(y, training_output, i, i + 1);       // y
        // PRINT_MATRIX_WITH_NAME(model.a0);
        // PRINT_MATRIX_WITH_NAME(y);
        forward_xor(model);

        for (size_t j = 0; j < output_cols; j++)
        {
            float difference = MATRIX_AT(model.a2, 0, j) - MATRIX_AT(y, 0, j); // difference between predicted and expected
            loss += difference * difference;
        }
    }
    return loss /= input_rows;
}

void finite_difference(Xor model, Xor gradient, Matrix t_input, Matrix t_output, float eps)
{
    float saved;
    float loss = loss_function(model, t_input, t_output);
    for (size_t i = 0; i < model.w1.rows; i++)
    {
        for (size_t j = 0; j < model.w1.cols; j++)
        {
            saved = MATRIX_AT(model.w1, i, j);
            MATRIX_AT(model.w1, i, j) += eps;
            MATRIX_AT(gradient.w1, i, j) = (loss_function(model, t_input, t_output) - loss) / eps;
            MATRIX_AT(model.w1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < model.b1.rows; i++)
    {
        for (size_t j = 0; j < model.b1.cols; j++)
        {
            saved = MATRIX_AT(model.b1, i, j);
            MATRIX_AT(model.b1, i, j) += eps;
            MATRIX_AT(gradient.b1, i, j) = (loss_function(model, t_input, t_output) - loss) / eps;
            MATRIX_AT(model.b1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < model.w2.rows; i++)
    {
        for (size_t j = 0; j < model.w2.cols; j++)
        {
            saved = MATRIX_AT(model.w2, i, j);
            MATRIX_AT(model.w2, i, j) += eps;
            MATRIX_AT(gradient.w2, i, j) = (loss_function(model, t_input, t_output) - loss) / eps;
            MATRIX_AT(model.w2, i, j) = saved;
        }
    }

    for (size_t i = 0; i < model.b2.rows; i++)
    {
        for (size_t j = 0; j < model.b2.cols; j++)
        {
            saved = MATRIX_AT(model.b2, i, j);
            MATRIX_AT(model.b2, i, j) += eps;
            MATRIX_AT(gradient.b2, i, j) = (loss_function(model, t_input, t_output) - loss) / eps;
            MATRIX_AT(model.b2, i, j) = saved;
        }
    }
}

void model_learn(Xor model, Xor gradient, float rate)
{
    for (size_t i = 0; i < model.w1.rows; i++)
    {
        for (size_t j = 0; j < model.w1.cols; j++)
        {
            MATRIX_AT(model.w1, i, j) -= rate * MATRIX_AT(gradient.w1, i, j);
        }
    }

    for (size_t i = 0; i < model.b1.rows; i++)
    {
        for (size_t j = 0; j < model.b1.cols; j++)
        {
            MATRIX_AT(model.b1, i, j) -= rate * MATRIX_AT(gradient.b1, i, j);
        }
    }

    for (size_t i = 0; i < model.w2.rows; i++)
    {
        for (size_t j = 0; j < model.w2.cols; j++)
        {
            MATRIX_AT(model.w2, i, j) -= rate * MATRIX_AT(gradient.w2, i, j);
        }
    }

    for (size_t i = 0; i < model.b2.rows; i++)
    {
        for (size_t j = 0; j < model.b2.cols; j++)
        {
            MATRIX_AT(model.b2, i, j) -= rate * MATRIX_AT(gradient.b2, i, j);
        }
    }
}

int main()
{
    srand(time(0));

    Xor model = XOR_alloc();
    Xor gradient = XOR_alloc();

    // fill in weights
    matrix_fill_with_random(model.w1, 0, 1);
    matrix_fill_with_random(model.b1, 0, 1);
    matrix_fill_with_random(model.w2, 0, 1);
    matrix_fill_with_random(model.b2, 0, 1);

    // import data into Matrix
    Matrix XOR_matrix = matrix_initialize(4, 3);
    matrix_add_data(XOR_matrix, XOR_values);
    // matrix_fill_with_random(XOR_matrix, 0, 10);

    // split data into features and lables
    Matrix X = matrix_initialize(2, XOR_matrix.rows);
    Matrix Y = matrix_initialize(1, XOR_matrix.rows);

    matrix_choose_columns(X, XOR_matrix, 0, 2);
    matrix_choose_columns(Y, XOR_matrix, 2, 3);

    matrix_inplace_transpose(X, &X.rows, &X.cols);
    matrix_inplace_transpose(Y, &Y.rows, &Y.cols);

    // PRINT_MATRIX_WITH_NAME(X);
    // PRINT_MATRIX_WITH_NAME(Y);

    printf("Cost %f: \n", loss_function(model, X, Y));

    float epsilon = 1e-1;
    float rate = 1e-1;
    for (size_t i = 0; i < 50000; i++)
    {
        finite_difference(model, gradient, X, Y, epsilon);
        model_learn(model, gradient, rate);
        printf("Cost %f: \n", loss_function(model, X, Y));
    }

    printf("\n---------------------------------\n");
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            MATRIX_AT(model.a0, 0, 0) = i;
            MATRIX_AT(model.a0, 0, 1) = j;
            forward_xor(model);
            printf("%zu ^ %zu: %f\n", i, j, *forward_xor(model));
        }
    }

    /*
        PRINT_MATRIX_WITH_NAME(model.w1);
        PRINT_MATRIX_WITH_NAME(model.b1);
        PRINT_MATRIX_WITH_NAME(model.a1);
        PRINT_MATRIX_WITH_NAME(model.w2);
        PRINT_MATRIX_WITH_NAME(model.b2);
        PRINT_MATRIX_WITH_NAME(model.a2);
    */
    matrix_free(model.w1);
    matrix_free(model.b1);
    matrix_free(model.w2);
    matrix_free(model.b2);
    matrix_free(model.a1);
    matrix_free(model.a2);
    matrix_free(model.a0);

    /*
        Matrix Ytest = matrix_initialize(3, XOR_matrix.cols);
        matrix_choose_rows(Ytest, XOR_matrix, 1, 4);
        PRINT_MATRIX_WITH_NAME(Ytest);


        for(size_t i = 0; i < test.cols; i++){
            matrix_choose_columns(col_m, test, i, i+1);
            PRINT_MATRIX_WITH_NAME(col_m);
        }

        for(size_t i = 0; i < test.rows; i++){
            matrix_choose_rows(row_m, test, i, i+1);
            PRINT_MATRIX_WITH_NAME(row_m);
        }

    */

    return 0;
}