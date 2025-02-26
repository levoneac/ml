#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"
#include "neural_net.h"

// size_t architecture[] = {2, 2, 1};
// size_t count = sizeof(layers)/sizeof(layers[0]);
// Neural_Network nn_alloc(architecture, ARRAY_LEN(architecture));

Neural_Network nn_alloc(size_t *architecture, size_t architecture_count, float (*activation_function)(float x))
{
    Neural_Network nn;

    nn.activation_func = activation_function;
    assert(architecture_count > 0);    // if arch count is less than 1 you dont have inputs
    nn.count = architecture_count - 1; // architecture_count also contains inputs

    nn.ws = malloc(sizeof(*nn.ws) * nn.count); // allocates memory for pointers to weigths matrix
    assert(nn.ws != NULL);

    nn.bs = malloc(sizeof(*nn.bs) * nn.count); // allocates memory for pointers to bias matrix
    assert(nn.bs != NULL);

    nn.as = malloc(sizeof(*nn.as) * (nn.count + 1)); // allocates memory for pointers to outputs matrix
    assert(nn.as != NULL);

    nn.as[0] = matrix_initialize(1, architecture[0]); // input data is always at index 0

    for (size_t i = 0; i < nn.count; i++)
    {
        nn.ws[i] = matrix_initialize(nn.as[i].cols, architecture[i + 1]); // nn.as[i].cols is the outputs of the previous layer. architecture[i + 1] is this layers outputs (+1 since nn.as[0] is the input data)
        nn.bs[i] = matrix_initialize(1, architecture[i + 1]);             // matrix with same size as the inputs to the layer, which always have 1 row and the same amount of columns as the layer
        nn.as[i + 1] = matrix_initialize(1, architecture[i + 1]);         // output will have the same arch as the bias
    }

    return nn;
}

void nn_free(Neural_Network nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        matrix_free(nn.ws[i]);
        matrix_free(nn.bs[i]);
        matrix_free(nn.as[i]);
        if (i == nn.count - 1)
        {
            matrix_free(nn.as[i + 1]);
        }
    }
    free(nn.ws);
    free(nn.bs);
    free(nn.as);
}

void nn_print(Neural_Network nn, const char *name)
{
    printf("NETWORK: %s\n\n", name);

    char buf[256];

    // snprintf(buf, sizeof(buf), "as:%d", 0);
    // matrix_print(nn.as[0], buf);

    for (size_t i = 0; i < nn.count; i++)
    {
        snprintf(buf, sizeof(buf), "ws:%zu", i); // writes a formatted string into the given buffer. Good for preventing overflows
        matrix_print(nn.ws[i], buf);

        snprintf(buf, sizeof(buf), "bs:%zu", i);
        matrix_print(nn.bs[i], buf);

        snprintf(buf, sizeof(buf), "as:%zu", i + 1);
        matrix_print(nn.as[i + 1], buf);

        printf("\n");
    }
}

void nn_fill_with_random(Neural_Network nn, float min, float max)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        matrix_fill_with_random(nn.ws[i], min, max);
        matrix_fill_with_random(nn.bs[i], min, max);
    }
}

void nn_forward(Neural_Network nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        matrix_multiplication(nn.as[i + 1], nn.as[i], nn.ws[i]); // multiply the inputs with the weights and forward them to the next layer
        matrix_sum(nn.as[i + 1], nn.as[i + 1], nn.bs[i]);

        if (i == nn.count)
        {
            // matrix_apply_sigmoid(nn.as[i + 1]);
        }
        else
        {
            matrix_apply_activation(nn.as[i + 1], nn.activation_func);
        }
    }
}

float nn_loss_function(Neural_Network nn, Matrix training_input, Matrix training_output)
{
    assert(training_input.rows == training_output.rows);  // there has to be the same amount of data in both input and output
    assert(training_output.cols == nn.as[nn.count].cols); // output of model has to be same size as expected

    size_t input_rows = training_input.rows;
    size_t output_cols = training_output.cols;

    float loss = 0.f;
    float difference = 0.f;
    Matrix y = matrix_initialize(1, training_output.cols); // y matrix
    for (size_t i = 0; i < input_rows; i++)
    {                                                           // iterate over all the input data
        matrix_choose_rows(nn.as[0], training_input, i, i + 1); // insert input into the inputlayer
        matrix_choose_rows(y, training_output, i, i + 1);       // insert the corresponding correct output into the y matrix

        nn_forward(nn); // forward the given input through the current network

        for (size_t j = 0; j < output_cols; j++)
        {                                                                       // calculate the total squared difference between expected and real values
            difference = MATRIX_AT(nn.as[nn.count], 0, j) - MATRIX_AT(y, 0, j); // difference between predicted and expected
            loss += difference * difference;
        }
    }
    matrix_free(y);
    return loss /= input_rows;
}

void nn_finite_difference(Neural_Network nn, Neural_Network gradient, Matrix training_input, Matrix training_output, float eps)
{
    float saved;
    float loss = nn_loss_function(nn, training_input, training_output);

    for (size_t layer = 0; layer < nn.count; layer++)
    {
        for (size_t i = 0; i < nn.ws[layer].rows; i++)
        {
            for (size_t j = 0; j < nn.ws[layer].cols; j++)
            {
                saved = MATRIX_AT(nn.ws[layer], i, j);
                MATRIX_AT(nn.ws[layer], i, j) += eps;
                MATRIX_AT(gradient.ws[layer], i, j) = (nn_loss_function(nn, training_input, training_output) - loss) / eps;
                MATRIX_AT(nn.ws[layer], i, j) = saved;
            }
        }
    }

    for (size_t layer = 0; layer < nn.count; layer++)
    {
        for (size_t i = 0; i < nn.bs[layer].rows; i++)
        {
            for (size_t j = 0; j < nn.bs[layer].cols; j++)
            {
                saved = MATRIX_AT(nn.bs[layer], i, j);
                MATRIX_AT(nn.bs[layer], i, j) += eps;
                MATRIX_AT(gradient.bs[layer], i, j) = (nn_loss_function(nn, training_input, training_output) - loss) / eps;
                MATRIX_AT(nn.bs[layer], i, j) = saved;
            }
        }
    }
}

void nn_learn(Neural_Network nn, Neural_Network gradient, float learn_rate)
{
    for (size_t layer = 0; layer < nn.count; layer++)
    {
        for (size_t i = 0; i < nn.ws[layer].rows; i++)
        {
            for (size_t j = 0; j < nn.ws[layer].cols; j++)
            {
                MATRIX_AT(nn.ws[layer], i, j) -= learn_rate * MATRIX_AT(gradient.ws[layer], i, j);
            }
        }
    }

    for (size_t layer = 0; layer < nn.count; layer++)
    {
        for (size_t i = 0; i < nn.bs[layer].rows; i++)
        {
            for (size_t j = 0; j < nn.bs[layer].cols; j++)
            {
                MATRIX_AT(nn.bs[layer], i, j) -= learn_rate * MATRIX_AT(gradient.bs[layer], i, j);
            }
        }
    }
}

Model_Confusion_Data nn_evaluate_classification(Neural_Network nn, Matrix X, Matrix Y)
{
    Model_Confusion_Data evaluation;
    evaluation.n_samples = X.rows;
    evaluation.total_positives = 0;
    evaluation.total_negatives = 0;
    evaluation.correct_predictions = 0;
    evaluation.wrong_predictions = 0;
    evaluation.true_positives = 0;
    evaluation.false_positives = 0;
    evaluation.true_negatives = 0;
    evaluation.false_negatives = 0;
    Matrix actual_label = matrix_initialize(1, 1);

    for (size_t i = 0; i < X.rows; i++)
    {
        matrix_choose_rows(nn.as[0], X, i, i + 1);
        matrix_choose_rows(actual_label, Y, i, i + 1);

        nn_forward(nn);

        float real = MATRIX_AT(actual_label, 0, 0);
        float predicted = MATRIX_AT(nn.as[nn.count], 0, 0);
        // printf("Predicted: %f, Real: %f\n",predicted, real); //print input with corresponding output

        if (predicted >= 0.5)
        {
            predicted = 1;
        }
        else
        {
            predicted = 0;
        }

        if (predicted == real)
        {
            evaluation.correct_predictions++;
        }
        else
        {
            evaluation.wrong_predictions++;
        }

        if (real == 1 && predicted == 1)
        {
            evaluation.true_positives++;
            evaluation.total_positives++;
        }
        else if (real == 1 && predicted == 0)
        {
            evaluation.false_negatives++;
            evaluation.total_positives++;
        }
        else if (real == 0 && predicted == 1)
        {
            evaluation.false_positives++;
            evaluation.total_negatives++;
        }
        else if (real == 0 && predicted == 0)
        {
            evaluation.true_negatives++;
            evaluation.total_negatives++;
        }
    }

    // printf("Training accuracy = %f%% (%f / %zu)\n", (sum_correct / X.rows), sum_correct, X.rows);
    matrix_free(actual_label);

    return evaluation;
}

Model_Prediction_Information nn_prediction_information(Model_Confusion_Data confusion_table)
{
    Model_Prediction_Information info;
    info.accuracy = confusion_table.correct_predictions / confusion_table.n_samples;
    info.true_positive_rate = confusion_table.true_positives / confusion_table.total_positives;
    info.false_positive_rate = confusion_table.false_positives / confusion_table.total_negatives;
    info.true_negative_rate = confusion_table.true_negatives / confusion_table.total_negatives;
    info.false_negative_rate = confusion_table.false_negatives / confusion_table.total_positives;
    info.precision = confusion_table.true_positives / (confusion_table.true_positives + confusion_table.false_positives);
    info.f1_score = (2 * info.true_positive_rate * info.precision) / (info.true_positive_rate + info.precision);

    return info;
}