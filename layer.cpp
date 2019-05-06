//
// Created by Robbie on 5/4/2019.
//

#include "layer.h"

void Layer::set_input_shape(shape input_shape) {
    this->input_shape = input_shape;
}

Dense::Dense(int n_units, shape input_shape) {
    this->input_shape = input_shape;
    this->n_units = n_units;
}
void Dense::initialize(Optimizer *optimizer) {
    double limit = 1.0 / sqrt(input_shape.first);
    this->W.set(this->input_shape.first, this->n_units);
    this->W.randomize_uniform(-1.0*limit, limit);
    this->w0.set(1, this->n_units, 0);
}
int Dense::parameters() {
    return this->W.get_rows_number() * this->W.get_columns_number() + this->w0.get_rows_number() * this->w0.get_columns_number();
}
Matrix<double> Dense::forward_pass(Matrix<double> *X, bool) {
    this->layer_input = *X;
    return X->dot(this->W) + this->w0;
}
Matrix<double> * Dense::backward_pass(Matrix<double> *accum_grad, int index) {
    auto W = this->W;
    if (this->trainable) {
        auto grad_W = this->layer_input.calculate_transpose().dot(*accum_grad);
        auto grad_w0 = accum_grad->calculate_columns_sum(); //dimensions???

        this->W = this->W_opt.update(&this->W, &grad_W);

    }
}
