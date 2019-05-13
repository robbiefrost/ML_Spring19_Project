//
// Created by Robbie on 5/3/2019.
//

#include "neural_network.h"
#include <string>
#include <chrono>

#define trace(input) do { if (1) { cout << input << endl; } } while(0)
#define here() do { cout << "  here" << endl; } while (0)
#define there() do { cout << "  there" << endl; } while (0)


NeuralNetwork::NeuralNetwork(string optimizer, Loss* loss_function) {
    this->optimizer = optimizer;
    this->loss_function = loss_function;
}
//void NeuralNetwork::set_trainable(bool trainable) {
//    for (auto layer : this->layers) {
//        layer->trainable = trainable;
//    }
//}
void NeuralNetwork::add(Layer* layer) {
    if (!this->layers.empty()) {
        layer->set_input_shape(this->layers.back()->output_shape());
    }
    if (layer->layer_name() == "Dense" || layer->layer_name() == "BatchNormalization") {
        layer->initialize(this->optimizer);
    }
    this->layers.push_back(layer);
}
//double NeuralNetwork::test_on_batch(Matrix<double>* X, Matrix<double>* y) {
//    Matrix<double> y_pred = this->_forward_pass(X, false);
//    auto loss = this->loss_function->loss(y, &y_pred);
//    //can add in acc if needed
//    return loss.calculate_mean().calculate_mean();
//}
double NeuralNetwork::train_on_batch(Matrix<double>* X, Matrix<double>* y) {
//    auto start = chrono::system_clock::now();
    Matrix<double> y_pred = this->_forward_pass(X, true);
//    auto t1 = chrono::system_clock::now();
//    trace("  fpass   " << (chrono::duration<double>(t1 - start)).count());
//    y_pred.print_preview();
    auto loss = this->loss_function->loss(y, &y_pred);
//    auto t2 = chrono::system_clock::now();
//    trace("  loss    " << (chrono::duration<double>(t2 - t1)).count());
    auto loss_grad = this->loss_function->gradient(y, &y_pred);
//    auto t3 = chrono::system_clock::now();
//    trace("  lgrad   " << (chrono::duration<double>(t3 - t2)).count());
    this->_backward_pass(&loss_grad);
//    auto t4 = chrono::system_clock::now();
//    trace("  bpass   " << (chrono::duration<double>(t4 - t3)).count());
    //can add in acc if needed
    return loss.calculate_mean().calculate_mean();
}
//Matrix<double> NeuralNetwork::fit(Matrix<double>* X, Matrix<double>* y, int n_epochs, int batch_size) {
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
//    return Matrix<double>();
//}
Matrix<double> NeuralNetwork::_forward_pass(Matrix<double>* X, bool training) {
    Matrix<double> layer_output = *X;
    int i = 0;
    for (auto layer : this->layers) {
//        trace(endl<<endl<<i++ << ": " << layer->layer_name());
//        trace("  in (" << layer_output.get_rows_number() << "," << layer_output.get_columns_number() << ")");
//        auto start = chrono::system_clock::now();
        layer_output.set(layer->forward_pass(&layer_output, training));
//        auto end = chrono::system_clock::now();
//        trace("    " << layer->layer_name()<< "  " << (chrono::duration<double>(end - start)).count());
//        layer_output.print_preview();
//        trace("  out(" << layer_output.get_rows_number() << "," << layer_output.get_columns_number() << ")");
    }
    return layer_output;
}
void NeuralNetwork::_backward_pass(Matrix<double>* loss_grad){
    int i = 0;
    for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
//        trace(i << ": " << (*it)->layer_name());
//        trace("  in (" << loss_grad->get_rows_number() << "," << loss_grad->get_columns_number() << ")");
//        auto start = chrono::system_clock::now();
        (*it)->backward_pass(loss_grad, i);
        i++;
//        auto end = chrono::system_clock::now();
//        trace("    " << (*it)->layer_name()<< "  " << (chrono::duration<double>(end - start)).count());
//        trace("  out(" << loss_grad->get_rows_number() << "," << loss_grad->get_columns_number() << ")");
    }
}
void NeuralNetwork::summary(string name) {
    cout << name << endl;
    cout << "Input Shape: (" << this->layers[0]->input_shape.first << ", " << this->layers[0]->input_shape.second << ")" << endl;
    cout << left << setw(27)<< "| Layer type" << setw(15) << "| Parameters " << setw(15) << "| Output Shape" << endl;
    int total_params = 0;
    for (auto layer : this->layers) {
        cout << left  << "| " << layer->layer_name() << setw(25-layer->layer_name().size()) << " "
            << "| " << setw(13) << layer->parameters()
            << "| " << "(" << layer->output_shape().first << "," << layer->output_shape().second << ")" << endl;
        total_params += layer->parameters();
    }
    cout << endl << "Total Parameters: " << total_params << endl << endl;
}
Matrix<double> NeuralNetwork::predict(Matrix<double> *X) {
    return this->_forward_pass(X, false);
}