//
// Created by Robbie on 5/3/2019.
//

#include "neural_network.h"
#include <string>


NeuralNetwork::NeuralNetwork(Optimizer* optimizer, Loss* loss_function) {
    this->optimizer = optimizer;
    this->loss_function = loss_function;
}
void NeuralNetwork::set_trainable(bool trainable) {
    for (auto layer : this->layers) {
        layer->trainable = trainable;
    }
}
void NeuralNetwork::add(Layer* layer) {
    if (!this->layers.empty()) {
        layer->set_input_shape(this->layers.back()->output_shape()); // needs parameters
    }
    if (layer->layer_name() == "Dense" || layer->layer_name() == "BatchNormalization")
        layer->initialize(this->optimizer);
    this->layers.push_back(layer);
}
double NeuralNetwork::test_on_batch(Matrix<double>* X, Matrix<double>* y) {
    Matrix<double> y_pred = this->_forward_pass(X, false);
    auto loss = this->loss_function->loss(y, &y_pred);
    //can add in acc if needed
    return loss.calculate_mean().calculate_mean();
}
double NeuralNetwork::train_on_batch(Matrix<double>* X, Matrix<double>* y) {
    Matrix<double> y_pred = this->_forward_pass(X, true);
    auto loss = this->loss_function->loss(y, &y_pred);
    auto loss_grad = this->loss_function->gradient(y, &y_pred);
    this->_backward_pass(&loss_grad);
    //can add in acc if needed
    return loss.calculate_mean().calculate_mean();
}
Matrix<double> NeuralNetwork::fit(Matrix<double>* X, Matrix<double>* y, int n_epochs, int batch_size) {
//    Vector<double> batch_error;
//    int n_samples = X->get_rows_number();
//    for (int j = 1; j<=n_epochs; j++) {
//        int i = 0;
//        do {
//            int end = min(i+batch_size-1, n_samples-1);
//            double loss = this->train_on_batch(X->get_submatrix_rows(i, end),y->get_submatrix_rows(i, end));
//            i = i+batch_size;
//        } while(i<n_samples);
//    }
    return Matrix<double>();
}
Matrix<double> NeuralNetwork::_forward_pass(Matrix<double>* X, bool training) {
    Matrix<double> layer_output = *X;
    for (auto layer : this->layers)
        layer_output = layer->forward_pass(&layer_output, training);
    return layer_output;
}
void NeuralNetwork::_backward_pass(Matrix<double>* loss_grad){
    for (int i = this->layers.size()-1; i>=0; i--)
        this->layers[i]->backward_pass(loss_grad, i);
}
void NeuralNetwork::summary(string name) {

}