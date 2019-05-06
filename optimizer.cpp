//
// Created by Robbie on 5/3/2019.
//

#include "optimizer.h"

using namespace std;

Optimizer::Optimizer() {
    learning_rate = 0.0002;
    b1 = 0.5;
    b2 = 0.999;
}
Optimizer::Optimizer(double learning_rate, double b1, double b2) {
    this->learning_rate = learning_rate;
    this->b1 = b1;
    this->b2 = b2;
}

Matrix<double> Optimizer::update(Matrix<double>* w, Matrix<double>* grad_wrt_w) {
    if (!initialized) {
        initialized = true;

    }
    m = m * b1 + *grad_wrt_w * (1 - b1);
    v = v * b2 + (*grad_wrt_w * *grad_wrt_w) * (1 - b2);

    auto m_hat = m / (1 - b1);
    auto v_hat = v / (1 - b2);

    return *w - ((m_hat * learning_rate) / sqrt(v_hat) + eps);
}