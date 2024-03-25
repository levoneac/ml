#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"


#define a_i 3
#define a_j 2


float rand_float(void){   
    return (float) rand() / (float) RAND_MAX;
}

float rand_float_between(float min, float max){
    return min + rand_float() * (max - min);
}

float sigmoidf(float x){
    return 1.0f/(1.0f + expf(-x));
}

void matrix_print(Matrix m, const char* name){
    printf("%s = {%zu x %zu}\n", name, m.rows, m.cols);

    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            printf("[%zu, %zu]: %f, ", i, j, MATRIX_AT(m, i, j)); //[%zu, %zu]:
        }
        printf("\n");
    }
    printf("---------------------\n");
}


Matrix matrix_initialize(size_t rows, size_t cols){
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = calloc(m.rows * m.cols, sizeof(float));
    return m;
}

void matrix_free(Matrix A){
    free(A.data);
}

void matrix_copy(Matrix dest, Matrix source){
    assert(dest.rows == source.rows);
    assert(dest.cols == source.cols);

    for(size_t i = 0; i < dest.rows; i++){
        for(size_t j = 0; j < dest.cols; j++){
            MATRIX_AT(dest, i, j) = MATRIX_AT(source, i, j);
        }
    }
}


void matrix_choose_rows(Matrix dest, Matrix source, size_t row_start, size_t row_stop){
    assert(row_start < source.rows);
    assert(row_stop <= source.rows && row_stop > row_start);
    Matrix temp = {
        .rows = row_stop - row_start,
        .cols = source.cols,
        .data = &MATRIX_AT(source, row_start, 0)
    };
    matrix_copy(dest, temp);
}

void matrix_choose_columns(Matrix dest, Matrix source, size_t col_start, size_t col_stop){
    assert(col_start < source.cols);
    assert(col_stop <= source.cols && col_stop > col_start);
    
    Matrix transposed = matrix_initialize(source.cols, source.rows);
    matrix_transpose(transposed, source);

    Matrix temp2 = {
        .rows = col_stop - col_start, //amount of rows -> if result is 2, then 2 rows are gathered in the matrix_copy starting from col_start
        .cols = transposed.cols, 
        .data = &MATRIX_AT(transposed, col_start, 0) //a reference to the starting row (which was a column before transposing)
    };

    matrix_copy(dest, temp2);
    matrix_free(transposed);
}


void matrix_add_data(Matrix m, float datum[m.rows][m.cols]){//to void or not to void
    //dont use
    /*if(m.rows != a_i || m.cols != a_j){
        return;
    }*/
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MATRIX_AT(m, i, j) = datum[i][j];
        }
    }
}

void matrix_fill_with_value(Matrix m, float value){
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MATRIX_AT(m, i, j) = value;
        }
    }
}

void matrix_fill_with_random(Matrix m, float min, float max){
    /*if(min >= max){
        return m;
    } trengs ikke med denne random funksjonen. om man bytter om min og max, så får man bare det motsatte tallet*/

    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MATRIX_AT(m, i, j) = rand_float_between(min, max);
        }
    }
}

void matrix_apply_sigmoid(Matrix m){
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MATRIX_AT(m, i, j) = sigmoidf(MATRIX_AT(m, i, j));
        }
    }
}

int matrix_multiplication(Matrix dest, Matrix A, Matrix B){
    if(A.cols != B.rows){
        return -1;
    };

    float result = 0.0f;
    for(size_t i = 0; i < A.rows; i++){
        for(size_t j = 0; j < B.cols; j++){
            for(size_t k = 0; k < A.cols; k++){ //k is common
                result += MATRIX_AT(A, i, k) * MATRIX_AT(B, k, j);
            }
            MATRIX_AT(dest, i, j) = result;
            result = 0.0f;
        }
    
    }

    return 1;
}

int matrix_sum(Matrix dest, Matrix A, Matrix B){
    if(A.cols != B.cols || A.rows != B.rows){
        return -1;
    }

    for(size_t i = 0; i < A.rows; i++){
        for(size_t j = 0; j < A.cols; j++){
            MATRIX_AT(dest, i, j) = MATRIX_AT(A, i, j) + MATRIX_AT(B, i, j);
        }
    }
    return 1;
}

int matrix_transpose(Matrix dest, Matrix A){
    assert(dest.cols = A.rows);
    assert(dest.rows = A.cols);

    for(size_t i = 0; i < A.rows; i++){
        for(size_t j = 0; j < A.cols; j++){
            MATRIX_AT(dest, j, i) = MATRIX_AT(A, i, j); //flip
        }
    }
    return 1;
}

void matrix_inplace_transpose(Matrix m, size_t *rows, size_t *cols){
    Matrix temp = matrix_initialize(m.cols, m.rows);
    matrix_transpose(temp, m);

    size_t temp_c = *cols;
    *cols = *rows; //greia her er at referencer ikke oppdaterer det som allerede er passa inn
    m.cols = *rows; //have to find a better way to do this later

    *rows = temp_c;
    m.rows = temp_c;

    matrix_copy(m, temp);
    matrix_free(temp);
}


/*
float a[a_i][a_j] = {
        {10.0f, 20.0f},
        {3.0f, 4.0f},
        {5.0f, 6.0f}
    };

float b[2][2] = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

float c[4] = {
    4.0f, 5.0f,
    6.9f, 9.2f
};

int main(){
    
        Matrix m_a = initialize_matrix(3, 2);
        m_a = fill_matrix_with_random(m_a, -7, -30);
        print_matrix(m_a);



        Matrix m_b = initialize_matrix(3, 2);
        add_data_to_matrix(m_b, a);
        print_matrix(m_b);

    
        Matrix m_c = matrix_multiplication(m_a, m_b);
        print_matrix(m_c);


    
        Matrix m_d = transpose_matrix(m_b);
        print_matrix(m_d);

        Matrix m_filled = initialize_matrix(2, 3);
        fill_matrix_with_value(m_filled, 10);
        print_matrix(m_filled);

        Matrix m_summed = sum_matrices(m_filled, m_d);
        print_matrix(m_summed);

        Matrix m_random = initialize_matrix(10, 10);
        fill_matrix_with_random(m_random, 0, 10);
        print_matrix(m_random);

        Matrix test = { .cols = 2, .rows = 2, .data = c};
        print_matrix(test);

    free_matrix(m_b);
    free_matrix(m_d);
    free_matrix(m_summed);
    free_matrix(m_random);
    free_matrix(m_filled);
    free_matrix(test);
    return 0;
}
*/