#pragma once

#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

class Tensor {
private:
    int rows_;
    int cols_;
    std::vector<float> data_;

public:
    Tensor(int rows, int cols)
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0f)
    {
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument("Tensor dimensions must be positive");
        }
    }

    int rows() const {
        return rows_;
    }

    int cols() const {
        return cols_;
    }

    float& operator()(int row, int col) {
        return data_[row * cols_ + col];
    }

    const float& operator()(int row, int col) const {
        return data_[row * cols_ + col];
    }

    void fill(float value) {
        for (int i = 0; i < rows_ * cols_; i++) {
            data_[i] = value;
        }
    }

    void randomize(float min = -1.0f, float max = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        for (int i = 0; i < rows_ * cols_; i++) {
            data_[i] = dist(gen);
        }
    }

    void print() const {
        for (int r = 0; r < rows_; r++) {
            for (int c = 0; c < cols_; c++) {
                std::cout << (*this)(r, c) << " ";
            }
            std::cout << "\n";
        }
    }

    float* data() {
        return data_.data();
    }

    const float* data() const {
        return data_.data();
    }
};