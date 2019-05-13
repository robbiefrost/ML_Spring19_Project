//
// Created by Robbie on 5/3/2019.
//

#include "optimizer.h"

using namespace std;

Adam::Adam(double learning_rate, double b1, double b2) {
    this->learning_rate = learning_rate;
    this->b1 = b1;
    this->b2 = b2;
}
Adam::Adam() {
    Adam(0.0002, 0.5, 0.999);
}
Matrix<double> Adam::update(Matrix<double> *w, Matrix<double> *grad_wrt_w) {
    if (!this->initialized) {
        this->initialized = true;
        this->m.set(grad_wrt_w->get_rows_number(), grad_wrt_w->get_columns_number(), 0);
        this->v.set(grad_wrt_w->get_rows_number(), grad_wrt_w->get_columns_number(), 0);
    }
    this->m = this->m * this->b1 + *grad_wrt_w * (1 - this->b1);
    this->v = this->v * this->b2 + (*grad_wrt_w * *grad_wrt_w) * (1 - this->b2);

    auto m_hat = this->m / (1 - this->b1);
    auto v_hat = this->v / (1 - this->b2);
    auto w_updt = (m_hat * this->learning_rate) / (v_hat.calculate_sqrt() + this->eps);

    return *w - w_updt;
}

Adadelta::Adadelta( double rho, double eps) {
    this->rho = rho;
    this->eps = eps;
}
Adadelta::Adadelta() {
    Adadelta(0.95, 1.0e-6);
}
Matrix<double> Adadelta::update(Matrix<double> *w, Matrix<double> *grad_wrt_w) {
    if (!this->initialized) {
        this->initialized = true;
        this->w_updt.set(w->get_rows_number(), w->get_columns_number(), 0);
        this->E_w_updt.set(w->get_rows_number(), w->get_columns_number(), 0);
        this->E_grad.set(grad_wrt_w->get_rows_number(), grad_wrt_w->get_columns_number(), 0);
    }

//    update average gradients at w
    this->E_grad = this->E_grad * this->rho + (*grad_wrt_w * *grad_wrt_w) * (1 - this->rho);

    auto RMS_delta_w = (this->E_w_updt + this->eps).calculate_sqrt();
    auto RMS_grad = (this->E_grad + this->eps).calculate_sqrt();

    auto adaptive_lr = RMS_delta_w / RMS_grad;

    this->w_updt = *grad_wrt_w * adaptive_lr;

    this->E_w_updt = this->E_w_updt * this->rho + (this->w_updt * this->w_updt) * (1 - this->rho);

    return *w - this->w_updt;
}