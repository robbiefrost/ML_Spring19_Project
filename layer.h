//
// Created by Robbie on 5/4/2019.
//

#ifndef ML_SPRING19_PROJECT_LAYER_H
#define ML_SPRING19_PROJECT_LAYER_H

#include <string>
#include "optimizer.h"
#include "./OpenNN/matrix.h"
#include "./OpenNN/vector.h"

using namespace std;
using namespace OpenNN;

typedef pair<int, int> shape;

class Layer {
public:
    shape input_shape;
    bool trainable = true;
    void set_input_shape(shape);
    virtual string layer_name();
    virtual int parameters();
    virtual void initialize(Optimizer*);
    virtual Matrix<double> forward_pass(Matrix<double>* X, bool);
    virtual Matrix<double>* backward_pass(Matrix<double>*, int);
    virtual shape output_shape();
};
class Dense:Layer {
    string name = "Dense";
    int n_units;
    Matrix<double> layer_input;
    Matrix<double> W;
    Matrix<double> w0;
    Optimizer W_opt;
    Optimizer w0_opt;
public:
    Dense(int, shape);
    string layer_name() override;
    int parameters() override;
    void initialize(Optimizer*) override;
    Matrix<double> forward_pass(Matrix<double>* X, bool) override;
    Matrix<double>* backward_pass(Matrix<double>*, int) override;
    shape output_shape() override;
};
class Activation:Layer {
    string name = "Activation";
    Matrix<double> layer_input;
public:
    string layer_name() override;
    int parameters() override;
    void initialize(Optimizer*) override;
    Matrix<double> forward_pass(Matrix<double>* X, bool) override;
    Matrix<double>* backward_pass(Matrix<double>*, int) override;
    shape output_shape() override;
};
class BatchNormalization:Layer {
    string name = "BatchNormalization";
public:
    string layer_name() override;
    int parameters() override;
    void initialize(Optimizer*) override;
    Matrix<double> forward_pass(Matrix<double>* X, bool) override;
    Matrix<double>* backward_pass(Matrix<double>*, int) override;
    shape output_shape() override;
};

#endif //ML_SPRING19_PROJECT_LAYER_H
