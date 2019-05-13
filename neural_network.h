//
// Created by Robbie on 5/3/2019.
//

#ifndef ML_SPRING19_PROJECT_NEURAL_NETWORK_H
#define ML_SPRING19_PROJECT_NEURAL_NETWORK_H

#include <string>
#include <vector>
#include "loss_function.h"
#include "optimizer.h"
#include "layer.h"
#include "./OpenNN/matrix.h"

using namespace std;
using namespace OpenNN;

using namespace std;

class NeuralNetwork {
    string optimizer;
    Loss* loss_function;
    Matrix<double> errors; // 0 is training, 1 is validation
    //skipping progress bar
    //skipping validation data

public:
    int batch_size = 0;
    int output_dim;
    vector<Layer*> layers;

    NeuralNetwork(string optimizer, Loss *loss_function);
//    void set_trainable(bool);
    void add(Layer*);
//    double test_on_batch(Matrix<double>*, Matrix<double>*);
    double train_on_batch(Matrix<double>*, Matrix<double>*);
//    Matrix<double> fit(Matrix<double>*, Matrix<double>*, int, int);
    Matrix<double> _forward_pass(Matrix<double>*, bool);
    void _backward_pass(Matrix<double>*);
    void summary(string);
    Matrix<double> predict(Matrix<double> *X);
};


#endif //ML_SPRING19_PROJECT_NEURAL_NETWORK_H
