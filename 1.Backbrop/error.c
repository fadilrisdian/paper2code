// Neural Network

// 1. Forward Pass
// 2. Compute error at output layer
// 3. Backward Pass: propagate error to hidden layers
// 4. Update Weight and biases

#include <stdio.h>
#include <math.h>

// Define network architecture constants.
#define INPUT_NODES   2
#define HIDDEN_NODES  2
#define OUTPUT_NODES  1

// Sigmoid activation function.
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Function that performs the forward pass.
void forward_pass(
    double input[INPUT_NODES],
    double weight_input_hidden[INPUT_NODES][HIDDEN_NODES],
    double bias_hidden[HIDDEN_NODES],
    double weight_hidden_output[HIDDEN_NODES][OUTPUT_NODES],
    double bias_output[OUTPUT_NODES],
    double output[OUTPUT_NODES]
) {
    double hidden[HIDDEN_NODES];

    // Compute hidden layer activations.
    for (int j = 0; j < HIDDEN_NODES; j++) {
        double activation = bias_hidden[j];
        for (int i = 0; i < INPUT_NODES; i++) {
            activation += input[i] * weight_input_hidden[i][j];
        }
        hidden[j] = sigmoid(activation);
    }

    // Compute output layer activations.
    for (int k = 0; k < OUTPUT_NODES; k++) {
        double activation = bias_output[k];
        for (int j = 0; j < HIDDEN_NODES; j++) {
            activation += hidden[j] * weight_hidden_output[j][k];
        }
        output[k] = sigmoid(activation);
    }
}

int main() {
    // Example input vector.
    double input[INPUT_NODES] = {1.0, 0.5};
    
    // Define the output vector
    double target[OUTPUT_NODES] = {0.8};

    // Example weights and biases for the input-to-hidden layer.
    double weight_input_hidden[INPUT_NODES][HIDDEN_NODES] = {
        {0.5, -0.4},
        {0.3,  0.8}
    };
    double bias_hidden[HIDDEN_NODES] = {0.1, -0.2};

    // Example weights and biases for the hidden-to-output layer.
    double weight_hidden_output[HIDDEN_NODES][OUTPUT_NODES] = {
        {0.7},
        {-1.2}
    };
    double bias_output[OUTPUT_NODES] = {0.05};

    // Array to store the final output.
    double output[OUTPUT_NODES];

    // ---------------------------
    // Step 1: Forward Pass
    // ---------------------------
    forward_pass(input, weight_input_hidden, bias_hidden,
                 weight_hidden_output, bias_output, output);

    // Display the result.
    printf("Forward Pass Output: ");
    for (int i = 0; i < OUTPUT_NODES; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    // ---------------------------
    // Step 2: Compute Error at Output Layer
    // ---------------------------
    // Error is calculated as (target - output)
    
    double output_error[OUTPUT_NODES];
    for (int k = 0; k < OUTPUT_NODES; k++){
        output_error[k] = target[k] - output[k];
        printf("Error at output neuron %d: %f\n", k, output_error[k]);
    }

    return 0;
}

