//
// Created by Robbie on 5/7/2019.
//

#include "activation_function.h"

Matrix<double> Sigmoid::function(Matrix<double> *x) {
    Matrix<double> ones(x->get_rows_number(), x->get_columns_number(), 1.0);
    auto exp_X = (*x * -1.0).calculate_exp();
    return ones / (exp_X + 1);
}
Matrix<double> Sigmoid::gradient(Matrix<double> *x) {
    auto sig = this->function(x);
    return sig * ((sig * -1.0) + 1);
}
Matrix<double> TanH::function(Matrix<double> *x) {
    Matrix<double> twos(x->get_rows_number(), x->get_columns_number(), 2.0);
    auto exp_X = (*x * -2.0).calculate_exp();
    return (twos / (exp_X + 1)) - 1;
}
Matrix<double> TanH::gradient(Matrix<double> *x) {
    auto tanh = this->function(x);
    return (tanh * tanh * -1) + 1;
}
//LeakyReLU::LeakyReLU(double alpha) {
//    this->alpha = alpha;
//}
Matrix<double> LeakyReLU::function(Matrix<double> *x) {
    //where x >= 0, leave x, otherwise replace with x*alpha
    return x->calculate_leaky_relu(this->alpha);
}
Matrix<double> LeakyReLU::gradient(Matrix<double> *x) {
    return x->calculate_leaky_relu_gradient(this->alpha);
}
