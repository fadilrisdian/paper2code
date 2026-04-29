#include <stdio.h>
#include <math.h>

#define INPUT_NODES   2
#define HIDDEN_NODES  2
#define OUTPUT_NODES  1

// Sigmoid activation function.
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function.
// Assumes the input x is already the output of the sigmoid.
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Forward pass function: computes the network's hidden and output activations.
void forward_pass(
    double input[INPUT_NODES],
    double weight_input_hidden[INPUT_NODES][HIDDEN_NODES],
    double bias_hidden[HIDDEN_NODES],
    double weight_hidden_output[HIDDEN_NODES][OUTPUT_NODES],
    double bias_output[OUTPUT_NODES],
    double hidden[HIDDEN_NODES],  // Output: hidden layer activations
    double output[OUTPUT_NODES]     // Output: output layer activations
) {
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
    // Define input vector.
    double input[INPUT_NODES] = {1.0, 0.5};

    // Define target output vector.
    double target[OUTPUT_NODES] = {0.8};

    // Define weights and biases for input-to-hidden layer.
    double weight_input_hidden[INPUT_NODES][HIDDEN_NODES] = {
        {0.5, -0.4},
        {0.3,  0.8}
    };
    double bias_hidden[HIDDEN_NODES] = {0.1, -0.2};

    // Define weights and biases for hidden-to-output layer.
    double weight_hidden_output[HIDDEN_NODES][OUTPUT_NODES] = {
        {0.7},
        {-1.2}
    };
    double bias_output[OUTPUT_NODES] = {0.05};

    // Arrays to store the activations.
    double hidden[HIDDEN_NODES];
    double output[OUTPUT_NODES];

    // ---------------------------
    // Step 1: Forward Pass
    // ---------------------------
    forward_pass(input, weight_input_hidden, bias_hidden,
                 weight_hidden_output, bias_output, hidden, output);

    // Display the output from the forward pass.
    printf("Forward Pass Output: ");
    for (int i = 0; i < OUTPUT_NODES; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    // ---------------------------
    // Step 2: Compute Error at Output Layer
    // ---------------------------
    // Error is computed as (target - output)
    double output_error[OUTPUT_NODES];
    for (int k = 0; k < OUTPUT_NODES; k++) {
        output_error[k] = target[k] - output[k];
        printf("Output Error at neuron %d: %f\n", k, output_error[k]);
    }

    // ---------------------------
    // Step 3: Backward Pass - Propagate Error to Hidden Layers
    // ---------------------------
    // First, compute delta for output neurons.
    double delta_output[OUTPUT_NODES];
    for (int k = 0; k < OUTPUT_NODES; k++) {
        // delta = error * derivative of activation function.
        delta_output[k] = output_error[k] * sigmoid_derivative(output[k]);
    }

    // Now, propagate the error to the hidden layer.
    double delta_hidden[HIDDEN_NODES];
    for (int j = 0; j < HIDDEN_NODES; j++) {
        double error = 0.0;
        // Sum the contributions from each output neuron.
        for (int k = 0; k < OUTPUT_NODES; k++) {
            error += delta_output[k] * weight_hidden_output[j][k];
        }
        // Multiply by the derivative of the activation function at the hidden neuron.
        delta_hidden[j] = error * sigmoid_derivative(hidden[j]);
    }

    // Display the computed delta values for hidden neurons.
    for (int j = 0; j < HIDDEN_NODES; j++) {
        printf("Delta at hidden neuron %d: %f\n", j, delta_hidden[j]);
    }

    return 0;
}

