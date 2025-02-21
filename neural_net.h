#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "matrix.h"

#define ARRAY_LEN(arr) sizeof((arr)) / sizeof((arr)[0])
#define PRINT_NN_WITH_NAME(nn) nn_print((nn), #nn)

typedef struct
{
    size_t count;
    Matrix *ws;
    Matrix *bs;
    Matrix *as;                      // one element bigger than the others for the input Matrix
    float (*activation_func)(float); // enter your favourite activation function
} Neural_Network;

typedef struct
{
    float n_samples;
    float total_positives;
    float total_negatives;
    float correct_predictions;
    float wrong_predictions;
    float true_positives;
    float false_positives;
    float true_negatives;
    float false_negatives;
} Model_Confusion_Data;

typedef struct
{
    float accuracy;
    float true_positive_rate; // sensitivity or recall
    float false_positive_rate;
    float true_negative_rate; // specificity
    float false_negative_rate;
    float precision; // tp/(all positive predictions)
    float f1_score;
} Model_Prediction_Information;

Neural_Network nn_alloc(size_t *architecture, size_t architecture_count, float (*activation_function)(float x));
void nn_free(Neural_Network nn);
void nn_print(Neural_Network nn, const char *name);
void nn_fill_with_random(Neural_Network nn, float min, float max);
void nn_forward(Neural_Network nn);
float nn_loss_function(Neural_Network nn, Matrix training_input, Matrix training_output);
void nn_finite_difference(Neural_Network nn, Neural_Network gradient, Matrix t_input, Matrix t_output, float eps);
void nn_learn(Neural_Network nn, Neural_Network gradient, float learn_rate);
Model_Confusion_Data nn_evaluate_classification(Neural_Network nn, Matrix train, Matrix test); // move these to another file
Model_Prediction_Information nn_prediction_information(Model_Confusion_Data confusion_table);

#endif