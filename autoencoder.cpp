//
// Created by Robbie on 5/3/2019.
//
#include <iostream>
#include <iomanip>
#include "optimizer.h"

using namespace std;


class Autoencoder {
private:
    int img_rows = 28;
    int img_cols = 28;
    int img_dim = img_rows * img_cols;
    int latent_dim = 32;
    Optimizer optimizer(0.0002, 0.5, 0.999);


    void save_images(int epoch, int X) {

    }
public:
    void train(int n_epochs, int batch_size, int save_interval) {

    }
}


int main (int argc, char **argv) {
    ae = new Autoencoder();
    ae.train(200000, 128, 40);
}