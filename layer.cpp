//
// Created by Robbie on 5/4/2019.
//

#include "layer.h"

#define trace(input) do { if (1) { cout << input << endl; } } while(0)
#define here() do { cout << "  here" << endl; } while (0)
#define there() do { cout << "  there" << endl; } while (0)

void Layer::set_input_shape(Shape input_shape) {
    this->input_shape = input_shape;
}

Dense::Dense(int n_units, Shape input_shape) {
    this->input_shape = input_shape;
    this->n_units = n_units;
}
void Dense::initialize(Optimizer optimizer) {
    double limit = 1.0 / sqrt(input_shape.first);
    this->W.set(this->input_shape.first, this->n_units);
    this->W.randomize_uniform(-1.0*limit, limit);
    this->w0.set(1, this->n_units, 0);
    this->W_opt = optimizer;
    this->w0_opt = optimizer;
}
string Dense::layer_name() {
    return this->name;
}
int Dense::parameters() {
    return this->W.get_rows_number() * this->W.get_columns_number()
        + this->w0.get_rows_number() * this->w0.get_columns_number();
}
Matrix<double> Dense::forward_pass(Matrix<double> *X, bool training) {
    this->layer_input = *X;
//    trace("    X (" << X->get_rows_number() << "," << X->get_columns_number() << ")");
//    trace("    W (" << W.get_rows_number() << "," << W.get_columns_number() << ")");
//    trace("    w0(" << w0.get_rows_number() << "," << w0.get_columns_number() << ")");
    return X->dot(this->W) + this->w0.get_row(0);

}
void Dense::backward_pass(Matrix<double> *accum_grad, int index) {
    auto W = this->W;
    if (this->trainable) {
        // calc gradients w.r.t. layer weights
        auto grad_W = this->layer_input.calculate_transpose().dot(*accum_grad);
        auto grad_w0(this->w0);
        grad_w0.set_column(0, accum_grad->calculate_columns_sum());
        // update layer weights
        this->W = this->W_opt.update(&this->W, &grad_W);
        this->w0 = this->w0_opt.update(&this->w0, &grad_w0);
    }
    *accum_grad = accum_grad->dot(W.calculate_transpose());
}
Shape Dense::output_shape() {
    return Shape (this->n_units, 1);
}


Activation::Activation(string function_name) {
    this->function_name = function_name;
    this ->trainable = true;
    if (function_name == "sigmoid")
        this->activation_function = new Sigmoid();
    else if (function_name == "tanh")
        this->activation_function = new TanH();
    else if (function_name == "leaky_relu")
        this->activation_function = new LeakyReLU();
}
string Activation::layer_name() {
    return this->name + " (" + this->function_name + ")";
}
int Activation::parameters() {
    return 0;
}
Matrix<double> Activation::forward_pass(Matrix<double> *X, bool training) {
    this->layer_input = *X;
    return this->activation_function->function(X);
}
void Activation::backward_pass(Matrix<double> *accum_grad, int index) {
    *accum_grad = *accum_grad * this->activation_function->gradient(&layer_input);
}
Shape Activation::output_shape() {
    return this->input_shape;
}
void Activation::initialize(Optimizer optimizer) {}


BatchNormalization::BatchNormalization(double momentum) {
    this->momentum = momentum;
    this->epsilon = 0.01;
    this->trainable = true;
    this->initialized = false;
}
void BatchNormalization::initialize(Optimizer optimizer) {
    this->gamma.set(this->input_shape.first, this->input_shape.second, 1);
    this->beta.set(this->input_shape.first, this->input_shape.second, 0);
    this->gamma_opt = optimizer;
    this->beta_opt = optimizer;
}
string BatchNormalization::layer_name() {
    return this->name;
}
int BatchNormalization::parameters() {
    return this->gamma.get_rows_number() * this->gamma.get_columns_number()
        + this->beta.get_rows_number() * this->beta.get_columns_number();
}
Matrix<double> BatchNormalization::forward_pass(Matrix<double> *X, bool training) {
    if (!this->initialized) {
        this->running_mean = X->calculate_mean();
        this->running_var = X->calculate_variance();
        this->initialized = true;
    }
    Vector<double> mean, var;
    if (training && this->trainable) {
        mean = X->calculate_mean();
        var = X->calculate_variance();
        this->running_mean = this->running_mean * this->momentum + mean * (1 - this->momentum);
        this->running_var = this->running_var * this->momentum + var * (1 - this->momentum);
    } else {
        mean = this->running_mean;
        var = this->running_var;
    }
    // stats to save for backward pass
    this->X_centered = *X - mean; // element-wise matrix minus row vector: each row of X gets the mean subtracted
    this->stddev_inv = Vector<double>(var.size(), 1.0) / (var + this->epsilon).calculate_square_root_elements();
//    this->X_centered.print_preview();
//    var.print_unique();
//    this->stddev_inv.print_unique();

    auto X_norm = this->X_centered * this->stddev_inv;
//    X_norm.print_preview();
//    this->gamma.print_preview();
//    this->beta.print_preview();
//    trace("    X_norm (" << X_norm.get_rows_number() << "," << X_norm.get_columns_number() << ")");
//    trace("    gamma  (" << this->gamma.get_rows_number() << "," << this->gamma.get_columns_number() << ")");
//    trace("    beta   (" << this->beta.get_rows_number() << "," << this->beta.get_columns_number() << ")");
    return X_norm * this->gamma.get_column(0) + this->beta.get_column(0);
}
void BatchNormalization::backward_pass(Matrix<double> *accum_grad, int index) {
    auto gamma = this->gamma;

    if (this->trainable) {
        auto X_norm = this->X_centered * this->stddev_inv;
        auto grad_gamma(this->gamma);
        auto grad_beta(this->beta);
//        trace("    accum_grad (" << accum_grad->get_rows_number() << "," << accum_grad->get_columns_number() << ")");
//        trace("    X_norm     (" << X_norm.get_rows_number() << "," << X_norm.get_columns_number() << ")");
        grad_gamma.set_column(0, (*accum_grad * X_norm).calculate_columns_sum());
//        trace("    grad_gamma (" << grad_gamma.get_rows_number() << "," << grad_gamma.get_columns_number() << ")");
//        trace("    gamma      (" << this->gamma.get_rows_number() << "," << this->gamma.get_columns_number() << ")");
        grad_beta.set_column(0, accum_grad->calculate_columns_sum());

        this->gamma = this->gamma_opt.update(&this->gamma, &grad_gamma);
        this->beta = this ->beta_opt.update(&this->beta, &grad_beta);
    }
    int batch_size = accum_grad->get_rows_number();
//    trace("    accum_grad (" << accum_grad->get_rows_number() << "," << accum_grad->get_columns_number() << ")");
//    trace("    X_centered (" << this->X_centered.get_rows_number() << "," << this->X_centered.get_columns_number() << ")");
    auto part1 = *accum_grad * batch_size - accum_grad->calculate_columns_sum()
            - this->X_centered * (this->stddev_inv * this->stddev_inv) * (*accum_grad * this->X_centered).calculate_columns_sum();
    auto part2 = (gamma * this->stddev_inv * (1.0/batch_size));
//    trace("    part1     (" << part1.get_rows_number() << "," << part1.get_columns_number() << ")");
//    trace("    part2     (" << part2.get_rows_number() << "," << part2.get_columns_number() << ")");
    *accum_grad = part1 * part2.get_column(0);
//    trace("    accum_grad(" << accum_grad->get_rows_number() << "," << accum_grad->get_columns_number() << ")");
}
Shape BatchNormalization::output_shape() {
    return this->input_shape;
}
