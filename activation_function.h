//
// Created by Robbie on 5/7/2019.
//

#ifndef ML_SPRING19_PROJECT_ACTIVATION_FUNCTION_H
#define ML_SPRING19_PROJECT_ACTIVATION_FUNCTION_H

#include "./OpenNN/matrix.h"
#include "./OpenNN/vector.h"

using namespace std;
using namespace OpenNN;

struct ActivationFunction {
    virtual inline Matrix<double> operator()(Matrix <double> *x);
    virtual Matrix<double> gradient(Matrix <double> *x);
};
struct Sigmoid : ActivationFunction {
    inline Matrix<double> operator()(Matrix <double> *x) override;
    Matrix<double> gradient(Matrix <double> *x) override;
};
struct TanH : ActivationFunction {
    inline Matrix<double> operator()(Matrix <double> *x) override;
    Matrix<double> gradient(Matrix <double> *x) override;
};
struct LeakyReLU : ActivationFunction {
    inline Matrix<double> operator()(Matrix <double> *x) override;
    Matrix<double> gradient(Matrix <double> *x) override;
};


#endif //ML_SPRING19_PROJECT_ACTIVATION_FUNCTION_H
