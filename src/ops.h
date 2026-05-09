#pragma once
#include "Tensor.h"

Tensor matmul(const Tensor& A, const Tensor& B);
Tensor relu(const Tensor& X);
Tensor add(const Tensor& A, const Tensor& B);
Tensor transpose(const Tensor& X);
Tensor softmax(const Tensor& X);
Tensor linear(const Tensor& X, const Tensor& W, const Tensor& b);
Tensor subtract(const Tensor& A, const Tensor& B);
Tensor scalar_multiply(const Tensor& X, float scalar);

float cross_entropy_loss(const Tensor& probs, int correct_class);

Tensor hadamard(const Tensor& A, const Tensor& B);
Tensor relu_derivative(const Tensor& X);
void sgd_update(Tensor& param, const Tensor& grad, float lr);