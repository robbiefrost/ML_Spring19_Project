//
// Created by Robbie on 5/3/2019.
//

#include <vector>
#include "optimizer.h"


class Optimizer {
private:
    double eps = 1.0e-10;
    double learning_rate, b1, b2;
    bool initialized = 0;
    vector<double> m,n;
public:
    Optimizer(lr, b1, b2) {
        learning_rate = lr;
        this.b1 = b1;
        this.b2 = b2;
    }
    void update(w, grad_wrt_w) {
        if (!initialized) {
            m.re
            //v = size of grad_wrt_w
        }
        m = b1 * m + (1 - b1) * grad_wrt_w;


    }
};