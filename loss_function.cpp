//
// Created by Robbie on 5/3/2019.
//

#include "loss_function.h"


Matrix<double> SquareLoss::loss(Matrix<double>* y, Matrix<double>* y_pred) {
    return ((*y - *y_pred)*(*y - *y_pred)) * 0.5;
}
Matrix<double> SquareLoss::gradient(Matrix<double>* y, Matrix<double>* y_pred) {
    return (*y - *y_pred) * -1.0;
}
Matrix<double> SquareLoss::acc(Matrix<double>* y, Matrix<double>* y_pred) {
    return Matrix<double> (1,0);
}
Matrix<double> CrossEntropy::loss(Matrix<double>* y, Matrix<double>* y_pred) {
    return ((*y - *y_pred)*(*y - *y_pred)) * 0.5;
}
Matrix<double> CrossEntropy::gradient(Matrix<double>* y, Matrix<double>* y_pred) {
    return (*y - *y_pred) * -1.0;
}
Matrix<double> CrossEntropy::acc(Matrix<double>* y, Matrix<double>* y_pred) {
    return Matrix<double> (1,0);
}