#include "ops.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

// Matrix multiply
Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("matmul shape mismatch");
    }

    Tensor C(A.rows(), B.cols());

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < B.cols(); j++) {
            float sum = 0.0f;

            for (int k = 0; k < A.cols(); k++) {
                sum += A(i, k) * B(k, j);
            }

            C(i, j) = sum;
        }
    }

    return C;
}

// ReLU activation fcn, introduces non-linearity by zeroing out negative values
Tensor relu(const Tensor& X) {
    Tensor Y(X.rows(), X.cols());

    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < X.cols(); j++) {
            float value = X(i, j);
            Y(i, j) = value > 0.0f ? value : 0.0f;
        }
    }

    return Y;
}

// Element-wise add two Tensors (matrix addition)
Tensor add(const Tensor& A, const Tensor& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("add shape mismatch");
    }

    Tensor C(A.rows(), A.cols());

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            C(i, j) = A(i, j) + B(i, j);
        }
    }

    return C;
}

// Element-wise subtract two Tensors (matrix subtraction)
Tensor subtract(const Tensor& A, const Tensor& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("subtract shape mismatch");
    }

    Tensor C(A.rows(), A.cols());

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            C(i, j) = A(i, j) - B(i, j);
        }
    }

    return C;
}

// Multiply every element of a Tensor by a scalar value
Tensor scalar_multiply(const Tensor& X, float scalar) {
    Tensor Y(X.rows(), X.cols());

    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < X.cols(); j++) {
            Y(i, j) = X(i, j) * scalar;
        }
    }

    return Y;
}

// Flip rows and columns of a Tensor
Tensor transpose(const Tensor& X) {
    Tensor Y(X.cols(), X.rows());

    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < X.cols(); j++) {
            Y(j, i) = X(i, j);
        }
    }

    return Y;
}

// Convert logits into a probability distribution for use in classification
Tensor softmax(const Tensor& X) {
    Tensor Y(X.rows(), X.cols());

    for (int i = 0; i < X.rows(); i++) {
        float max_val = X(i, 0);

        for (int j = 1; j < X.cols(); j++) {
            if (X(i, j) > max_val) {
                max_val = X(i, j);
            }
        }

        float sum = 0.0f;

        for (int j = 0; j < X.cols(); j++) {
            Y(i, j) = std::exp(X(i, j) - max_val);
            sum += Y(i, j);
        }

        for (int j = 0; j < X.cols(); j++) {
            Y(i, j) = Y(i, j) / sum;
        }
    }

    return Y;
}

// Linear transformation: Y = XW + b
Tensor linear(const Tensor& X, const Tensor& W, const Tensor& b) {
    Tensor Y = matmul(X, W);
    return add(Y, b);
}

// Simple loss function: measures how accurate a classification prediction is
float cross_entropy_loss(const Tensor& probs, int correct_class) {
    float p = probs(0, correct_class);

    const float epsilon = 1e-7f;

    return -std::log(p + epsilon);
}


// Element-wise multiply two Tensors (Hadamard product)
Tensor hadamard(const Tensor& A, const Tensor& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("hadamard shape mismatch");
    }

    Tensor C(A.rows(), A.cols());

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            C(i, j) = A(i, j) * B(i, j);
        }
    }

    return C;
}


// Derivative of ReLU (needed for gradient descent backpropagation)
Tensor relu_derivative(const Tensor& X) {
    Tensor Y(X.rows(), X.cols());

    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < X.cols(); j++) {
            Y(i, j) = X(i, j) > 0.0f ? 1.0f : 0.0f;
        }
    }

    return Y;
}

// Update parameters using Stochastic Gradient Descent (SGD)
void sgd_update(Tensor& param, const Tensor& grad, float lr) {
    if (param.rows() != grad.rows() || param.cols() != grad.cols()) {
        throw std::invalid_argument("sgd_update shape mismatch");
    }

    for (int i = 0; i < param.rows(); i++) {
        for (int j = 0; j < param.cols(); j++) {
            param(i, j) -= lr * grad(i, j);
        }
    }
}

