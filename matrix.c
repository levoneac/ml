#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    size_t rows;
    size_t cols;
    float* data;
} matrix;

float a[3][2] = {
        {1.0f, 2.0f},
        {3.0f, 4.0f},
        {5.0f, 6.0f}
    };

float b[2][2] = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };


void print_matrix(matrix m){
    printf("%zu x %zu\n", m.rows, m.cols);

    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            printf("[%zu, %zu]: %f\n", i, j, m.data[i*m.cols + j]);
        }
    }
}

matrix initialize_matrix(size_t rows, size_t cols){
    matrix m;
    m.cols = cols;
    m.rows = rows;
    m.data = malloc(m.rows * m.cols * sizeof(float));
    return m;
}

matrix add_data_to_matrix(matrix m, float datum[m.rows][m.cols]){//to void or not to void
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            m.data[i*m.cols + j] = datum[i][j];
        }
    }

    return m;
}

matrix matrix_multiplication(matrix A, matrix B){
    if(A.cols != B.rows){
        return initialize_matrix(0,0);
    };

    matrix new_matrix = initialize_matrix(A.rows, B.cols);
    float result = 0.0f;
    for(size_t i = 0; i < A.rows; i++){
        for(size_t j = 0; j < B.cols; j++){
            for(size_t k = 0; k < A.cols; k++){ //common
                result += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
            new_matrix.data[i*new_matrix.cols + j] = result;
            result = 0.0f;
        }
    
    }

    return new_matrix;
}

matrix transpose_matrix(matrix A){
    matrix new_matrix = initialize_matrix(A.cols, A.rows);

    for(size_t i = 0; i < A.rows; i++){
        for(size_t j = 0; j < A.cols; j++){
            new_matrix.data[j * new_matrix.cols + i] = A.data[i * A.cols + j]; //flip
        }
    }
    return new_matrix;
}

void free_matrix(matrix* m){
    free(m->data);
}


int main(){
    matrix m_a = initialize_matrix(3, 2);
    m_a = add_data_to_matrix(m_a, a);
    print_matrix(m_a);

printf("---------------------\n");

    matrix m_b = initialize_matrix(2, 2);
    m_b = add_data_to_matrix(m_b, b);
    print_matrix(m_b);

printf("---------------------\n");

    matrix m_c = matrix_multiplication(m_a, m_b);
    print_matrix(m_c);

printf("---------------------\n");

    matrix m_d = transpose_matrix(m_c);
    print_matrix(m_d);

printf("---------------------\n");

    free_matrix(&m_a);
    free_matrix(&m_b);
    free_matrix(&m_c);
    free_matrix(&m_d);
    return 0;
}