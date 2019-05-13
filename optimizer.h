//
// Created by Robbie on 5/3/2019.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "./OpenNN/matrix.h"

using namespace std;
using namespace OpenNN;

class Optimizer {
public:
    virtual Matrix<double> update(Matrix<double> *w, Matrix<double> *grad_wrt_w)=0;
};

class Adam: public Optimizer {
    double eps = 1.0e-8;
    double learning_rate, b1, b2;
    bool initialized = false;
    Matrix<double> m,v;
public:
    Adam();
    Adam(double, double, double);
    Matrix<double> update(Matrix<double> *w, Matrix<double> *grad_wrt_w) override ;
};
class Adadelta: public Optimizer {
    double eps, rho;
    bool initialized = false;
    Matrix<double> E_w_updt, E_grad, w_updt;
public:
    Adadelta();
    Adadelta(double, double);
    Matrix<double> update(Matrix<double> *w, Matrix<double> *grad_wrt_w) override ;
};


#endif //OPTIMIZER_H
