#ifndef MATRIX_H
#define MATRIX_H

#include "read_csv.h"


typedef struct {
    size_t rows;
    size_t cols;
    float* data;
} Matrix;

#define MATRIX_AT(mat, i, j) (mat).data[ ((i) * (mat).cols) + (j)]
#define PRINT_MATRIX_WITH_NAME(m) matrix_print(m, #m)

float rand_float(void);
float rand_float_between(float min, float max);
float sigmoidf(float x);

//prints the values of the Matrix
void matrix_print(Matrix m, const char* name);

//deep copies a Matrix
void matrix_copy(Matrix dest, Matrix source);

//copies a row of the given Matrix into the dest
//zero indexed and stop is non inclusive
void matrix_choose_rows(Matrix dest, Matrix source, size_t row_start, size_t row_stop);

//copies a column of the given Matrix into the dest
//zero indexed and stop is non inclusive
void matrix_choose_columns(Matrix dest, Matrix source, size_t col_start, size_t col_stop);

//initializes the Matrix with specified amounts of rows and cols
Matrix matrix_initialize(size_t rows, size_t cols);

//adds predefined data to a Matrix
void matrix_add_data(Matrix m, float[m.rows][m.cols]);

void matrix_add_from_csv_import(Matrix m, csv csv_import);

//fills all the space in the Matrix with a chosen value
void matrix_fill_with_value(Matrix m, float value);

//fills all the space in the Matrix with random floats in a range
void matrix_fill_with_random(Matrix m, float min, float max);

//applies activation function, so that all values in the Matrix is between 0 and 1
void matrix_apply_sigmoid(Matrix m);

//performs Matrix multiplication. Number of columns in A needs the be the same as number of rows in B. Output is A_rows X B_cols
int matrix_multiplication(Matrix dest, Matrix A, Matrix B);

//Sums two matrices. Dimensions need to be equal
int matrix_sum(Matrix dest, Matrix A, Matrix B);

//transposes a Matrix so that columns become rows and vice versa. DImensions are flipped.
int matrix_transpose(Matrix dest, Matrix A);

//transposes a Matrix inplace so that columns become rows and vice versa. DImensions are flipped
void matrix_inplace_transpose(Matrix m, size_t *m_rows, size_t *m_cols);

//frees the memory of the Matrix
void matrix_free(Matrix A);

#endif