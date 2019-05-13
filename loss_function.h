//
// Created by Robbie on 5/3/2019.
//

#ifndef ML_SPRING19_PROJECT_LOSS_FUNCTION_H
#define ML_SPRING19_PROJECT_LOSS_FUNCTION_H

#include "./OpenNN/matrix.h"

using namespace std;
using namespace OpenNN;

class Loss {
public:
    virtual Matrix<double> loss(Matrix<double> *, Matrix<double> *)=0;
    virtual Matrix<double> gradient(Matrix<double> *, Matrix<double> *)=0;
    virtual Matrix<double> acc(Matrix<double> *, Matrix<double> *)=0;
};
class SquareLoss: public Loss {
public:
    Matrix<double> loss(Matrix<double> *, Matrix<double> *) override;
    Matrix<double> gradient(Matrix<double> *, Matrix<double> *) override;
    Matrix<double> acc(Matrix<double> *, Matrix<double> *) override;
};
class CrossEntropy: public Loss {
public:
    Matrix<double> loss(Matrix<double> *, Matrix<double> *) override;
    Matrix<double> gradient(Matrix<double> *, Matrix<double> *) override;
    Matrix<double> acc(Matrix<double> *, Matrix<double> *) override;
};


#endif //ML_SPRING19_PROJECT_LOSS_FUNCTION_H
