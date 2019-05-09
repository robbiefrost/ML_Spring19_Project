//
// Created by Robbie on 5/4/2019.
//

#ifndef ML_SPRING19_PROJECT_LAYER_H
#define ML_SPRING19_PROJECT_LAYER_H

#include <string>
#include "optimizer.h"
#include "activation_function.h"
#include "./OpenNN/matrix.h"
#include "./OpenNN/vector.h"

using namespace std;
using namespace OpenNN;

typedef pair<int, int> Shape;

class Layer {
public:
    Shape input_shape;
    bool trainable;
    void set_input_shape(Shape);
    virtual string layer_name();
    virtual int parameters();
    virtual void initialize(Optimizer optimizer);
    virtual Matrix<double> forward_pass(Matrix<double> *X, bool training);
    virtual void backward_pass(Matrix<double> *accum_grad, int index);
    virtual Shape output_shape();
};
class Dense: public Layer {
    string name = "Dense";
    int n_units;
    Matrix<double> layer_input;
    Matrix<double> W, w0;
    Optimizer W_opt, w0_opt;
public:
    Dense(int, Shape);
    string layer_name() override;
    int parameters() override;
    void initialize(Optimizer optimizer) override;
    Matrix<double> forward_pass(Matrix<double> *X, bool training) override;
    void backward_pass(Matrix<double> *accum_grad, int index) override;
    Shape output_shape() override;
};
class Activation: public Layer {
    string name = "Activation";
    string function_name;
    ActivationFunction* activation_function;
    Matrix<double> layer_input;
public:
    Activation(string);
    string layer_name() override;
    int parameters() override;
    void initialize(Optimizer optimizer) override;
    Matrix<double> forward_pass(Matrix<double> *X, bool training) override;
    void backward_pass(Matrix<double> *accum_grad, int index) override;
    Shape output_shape() override;
};
class BatchNormalization: public Layer {
    string name = "BatchNormalization";
    bool initialized;
    double momentum, epsilon;
    Vector<double> running_mean, running_var, stddev_inv;
    Matrix<double> gamma, beta, X_centered;
    Optimizer gamma_opt, beta_opt;
public:
    BatchNormalization(double);
    string layer_name() override;
    int parameters() override;
    void initialize(Optimizer optimizer) override;
    Matrix<double> forward_pass(Matrix<double> *X, bool training) override;
    void backward_pass(Matrix<double> *accum_grad, int index) override;
    Shape output_shape() override;
};

#endif //ML_SPRING19_PROJECT_LAYER_H
