//
// Created by Robbie on 5/7/2019.
//

#ifndef ML_SPRING19_PROJECT_ACTIVATION_FUNCTION_H
#define ML_SPRING19_PROJECT_ACTIVATION_FUNCTION_H

#include "./OpenNN/matrix.h"

using namespace std;
using namespace OpenNN;

class ActivationFunction {
public:
    virtual Matrix<double> function(Matrix <double> *x)=0;
    virtual Matrix<double> gradient(Matrix <double> *x)=0;
};
class Sigmoid : public ActivationFunction {
public:
    Matrix<double> function(Matrix <double> *x) override;
    Matrix<double> gradient(Matrix <double> *x) override;
};
class TanH : public ActivationFunction {
public:
    Matrix<double> function(Matrix <double> *x) override;
    Matrix<double> gradient(Matrix <double> *x) override;
};
class LeakyReLU : public ActivationFunction {
private:
    double alpha=0.2;
public:
//    LeakyReLU(double alpha);
    Matrix<double> function(Matrix <double> *x) override;
    Matrix<double> gradient(Matrix <double> *x) override;
};


#endif //ML_SPRING19_PROJECT_ACTIVATION_FUNCTION_H
