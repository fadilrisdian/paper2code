#include <math.h>
#include <stdio.h>

#define INPUT_NODES 2
#define HIDDEN_NODES 2
#define OUTPUT_NODES 1
#define LEARNING_RATE 0.5

// Sigmoid activation function.
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// Derivative of the sigmoid function.
// (Assumes the input is already the output of the sigmoid.)
double sigmoid_derivative(double x) { return x * (1.0 - x); }

// Forward pass function: computes both hidden and output layer activations.
void forward_pass(
    double input[INPUT_NODES],
    double weight_input_hidden[INPUT_NODES][HIDDEN_NODES],
    double bias_hidden[HIDDEN_NODES],
    double weight_hidden_output[HIDDEN_NODES][OUTPUT_NODES],
    double bias_output[OUTPUT_NODES],
    double hidden[HIDDEN_NODES], // Output: hidden layer activations
    double output[OUTPUT_NODES]  // Output: output layer activations
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
  // Define an example input vector.
  double input[INPUT_NODES] = {1.0, 0.5};

  // Define the target output.
  double target[OUTPUT_NODES] = {0.8};

  // Define weights and biases for the input-to-hidden layer.
  double weight_input_hidden[INPUT_NODES][HIDDEN_NODES] = {{0.5, -0.4},
                                                           {0.3, 0.8}};
  double bias_hidden[HIDDEN_NODES] = {0.1, -0.2};

  // Define weights and biases for the hidden-to-output layer.
  double weight_hidden_output[HIDDEN_NODES][OUTPUT_NODES] = {{0.7}, {-1.2}};
  double bias_output[OUTPUT_NODES] = {0.05};

  // Arrays to store the activations.
  double hidden[HIDDEN_NODES];
  double output[OUTPUT_NODES];

  // ---------------------------
  // Step 1: Forward Pass
  // ---------------------------
  forward_pass(input, weight_input_hidden, bias_hidden, weight_hidden_output,
               bias_output, hidden, output);
  printf("Forward Pass Output: ");
  for (int k = 0; k < OUTPUT_NODES; k++) {
    printf("%f ", output[k]);
  }
  printf("\n");

  // ---------------------------
  // Step 2: Compute Error at Output Layer
  // ---------------------------
  double output_error[OUTPUT_NODES];
  for (int k = 0; k < OUTPUT_NODES; k++) {
    output_error[k] = target[k] - output[k];
    printf("Output Error at neuron %d: %f\n", k, output_error[k]);
  }

  // ---------------------------
  // Step 3: Backward Pass - Propagate Error to Hidden Layers
  // ---------------------------
  double delta_output[OUTPUT_NODES];
  for (int k = 0; k < OUTPUT_NODES; k++) {
    // delta for output neuron = error * derivative of activation.
    delta_output[k] = output_error[k] * sigmoid_derivative(output[k]);
  }

  double delta_hidden[HIDDEN_NODES];
  for (int j = 0; j < HIDDEN_NODES; j++) {
    double error = 0.0;
    // Sum the contributions from all output neurons.
    for (int k = 0; k < OUTPUT_NODES; k++) {
      error += delta_output[k] * weight_hidden_output[j][k];
    }
    delta_hidden[j] = error * sigmoid_derivative(hidden[j]);
    printf("Delta at hidden neuron %d: %f\n", j, delta_hidden[j]);
  }

  // ---------------------------
  // Step 4: Update Weights and Biases
  // ---------------------------
  // Update weights and biases for the hidden-to-output layer.
  for (int j = 0; j < HIDDEN_NODES; j++) {
    for (int k = 0; k < OUTPUT_NODES; k++) {
      weight_hidden_output[j][k] += LEARNING_RATE * delta_output[k] * hidden[j];
    }
  }
  for (int k = 0; k < OUTPUT_NODES; k++) {
    bias_output[k] += LEARNING_RATE * delta_output[k];
  }

  // Update weights and biases for the input-to-hidden layer.
  for (int i = 0; i < INPUT_NODES; i++) {
    for (int j = 0; j < HIDDEN_NODES; j++) {
      weight_input_hidden[i][j] += LEARNING_RATE * delta_hidden[j] * input[i];
    }
  }
  for (int j = 0; j < HIDDEN_NODES; j++) {
    bias_hidden[j] += LEARNING_RATE * delta_hidden[j];
  }

  // Optionally, display the updated weights and biases.
  printf("\nUpdated weights (hidden to output):\n");
  for (int j = 0; j < HIDDEN_NODES; j++) {
    for (int k = 0; k < OUTPUT_NODES; k++) {
      printf("%f ", weight_hidden_output[j][k]);
    }
    printf("\n");
  }

  printf("Updated biases (output):\n");
  for (int k = 0; k < OUTPUT_NODES; k++) {
    printf("%f ", bias_output[k]);
  }
  printf("\n");

  printf("Updated weights (input to hidden):\n");
  for (int i = 0; i < INPUT_NODES; i++) {
    for (int j = 0; j < HIDDEN_NODES; j++) {
      printf("%f ", weight_input_hidden[i][j]);
    }
    printf("\n");
  }

  printf("Updated biases (hidden):\n");
  for (int j = 0; j < HIDDEN_NODES; j++) {
    printf("%f ", bias_hidden[j]);
  }
  printf("\n");

  return 0;
}
