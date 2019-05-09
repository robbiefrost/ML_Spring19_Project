//
// Created by Robbie on 5/3/2019.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "./OpenNN/matrix.h"

using namespace std;
using namespace OpenNN;

class Optimizer {
    double eps = 1.0e-8;
    double learning_rate, b1, b2;
    bool initialized = false;
    Matrix<double> m,v;
public:
    Optimizer();
    Optimizer(double, double, double);
    Matrix<double> update(Matrix<double>*,Matrix<double>*);
};


#endif //OPTIMIZER_H
