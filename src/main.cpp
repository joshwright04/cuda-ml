#include <iostream>
#include "Tensor.h"
#include "ops.h"
#include <vector>

int main() {

    // Model is trained on predicting XOR
    // Since XOR is not linearly separable, the non-linear layers of the network are put to use

    // Initialize XOR training data
    Tensor X1(1, 2);
    X1(0,0) = 0;
    X1(0,1) = 0;
    int y1 = 0;

    Tensor X2(1, 2);
    X2(0,0) = 0;
    X2(0,1) = 1;
    int y2 = 1;

    Tensor X3(1, 2);
    X3(0,0) = 1;
    X3(0,1) = 0;
    int y3 = 1;

    Tensor X4(1, 2);
    X4(0,0) = 1;
    X4(0,1) = 1;
    int y4 = 0;

    std::vector<Tensor*> samples = {&X1, &X2, &X3, &X4};
    std::vector<int> labels = {y1, y2, y3, y4};

    float lr = 0.05f; // How much the weights are updated during training
    int epochs = 20000; // # times the training process (forward + backward pass) is repeated

    Tensor W1(2, 8); // 2 input features (X1 and X2) and 8 hidden neurons
    Tensor b1(1, 8); // This is b in the linear layer Y = WX + b

    Tensor W2(8, 2);
    Tensor b2(1, 2); 

    W1.randomize(-0.5f, 0.5f); // Randomly initialize weights to small values (better than 0 to break symmetry)
    W2.randomize(-0.5f, 0.5f);

    b1.fill(0.0f); // Biases start out at 0
    b2.fill(0.0f);

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        for (int s = 0; s < samples.size(); s++) {
            Tensor& X = *samples[s];
            int correct_class = labels[s];

            // ---------- Forward Pass ----------
            // Recall that the process is X -> Linear -> ReLU -> Linear -> Softmax -> Loss

            Tensor z1 = linear(X, W1, b1);
            Tensor hidden = relu(z1);

            Tensor logits = linear(hidden, W2, b2);
            Tensor probs = softmax(logits);

            float loss = cross_entropy_loss(probs, correct_class);

            // ---------- Backward Pass ----------
            // Backpropagation:
            // Compute gradients for each layer, starting from the output
            // and moving backward through the network using the chain rule.


            // Gradient of loss w.r.t. output logits
            // (softmax + cross entropy shortcut)
            Tensor dlogits = probs;
            dlogits(0, correct_class) -= 1.0f; // Since labels are one-hot, this is equivalent to probs - true_labels

            // Output layer gradients
            Tensor dW2 = matmul(transpose(hidden), dlogits);
            Tensor db2 = dlogits;

            // Propagate error back to hidden layer
            Tensor dhidden = matmul(dlogits, transpose(W2));

            // Apply ReLU derivative to get gradients w.r.t. z1
            Tensor dz1 = hadamard(dhidden, relu_derivative(z1));

            // First layer gradients
            Tensor dW1 = matmul(transpose(X), dz1);
            Tensor db1 = dz1;

            // ---------- Gradient Descent ----------
            sgd_update(W1, dW1, lr);
            sgd_update(b1, db1, lr);

            sgd_update(W2, dW2, lr);
            sgd_update(b2, db2, lr);

            total_loss += loss;

        }

        if (epoch % 2000 == 0) {
            std::cout << "Epoch: " << epoch << "\n";
            std::cout << "Average Loss: " << total_loss / samples.size() << "\n";
            std::cout << "-------------------\n";
        }

    }

    std::cout << "\nFinal predictions:\n";

    for (int s = 0; s < samples.size(); s++) {
        Tensor& X = *samples[s];

        Tensor z1 = linear(X, W1, b1);
        Tensor hidden = relu(z1);
        Tensor logits = linear(hidden, W2, b2);
        Tensor probs = softmax(logits);

        std::cout << "Input: ";
        X.print();

        std::cout << "Expected class: " << labels[s] << "\n";
        std::cout << "Predicted probabilities:\n";
        probs.print();
        std::cout << "-------------------\n";
    }

    return 0;
}