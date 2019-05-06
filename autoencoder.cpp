//
// Created by Robbie on 5/3/2019.
//
#include <iostream>
#include <iomanip>
#include "optimizer.h"
#include "loss_function.h"
#include "neural_network.h"
#include "./OpenNN/matrix.h"
#include "./OpenNN/vector.h"

using namespace std;
using namespace OpenNN;

class Autoencoder {
    int img_rows = 28;
    int img_cols = 28;
    int img_dim = img_rows * img_cols;
    int latent_dim = 32;
    Optimizer* optimizer;
    Loss* loss_function;
//    loss_function = new SquareLoss();
    NeuralNetwork* autoencoder;

    NeuralNetwork build_encoder(Optimizer* optimizer, Loss* loss_function) {
        NeuralNetwork encoder(optimizer, loss_function);
//        encoder.add(Dense())
    }
    NeuralNetwork build_decoder(Optimizer* optimizer, Loss* loss_function) {
        NeuralNetwork decoder(optimizer, loss_function);
    }
    void save_images(int epoch, Matrix<double>* X) {

    }
public:
    Autoencoder() {
        optimizer = new Optimizer();
        loss_function = new SquareLoss();
        NeuralNetwork encoder = this->build_encoder(optimizer, loss_function);
        NeuralNetwork decoder = this->build_decoder(optimizer, loss_function);
        autoencoder = new NeuralNetwork(optimizer, loss_function);
        autoencoder->layers.insert(autoencoder->layers.end(), encoder.layers.begin(), encoder.layers.end());
        autoencoder->layers.insert(autoencoder->layers.end(), decoder.layers.begin(), decoder.layers.end());
        autoencoder->output_dim = img_dim;
        autoencoder->summary("Variational Autoencoder");
    }

    void train(int n_epochs, int batch_size, int save_interval) {
        this->autoencoder->batch_size = batch_size;
        Matrix<double> train_data("mnist_train.csv", ',', true);
        Matrix<double> test_data("mnist_test.csv", ',', true);
        Matrix<double> X = train_data.get_submatrix_columns(Vector<size_t>(1,1,train_data.get_columns_number()-1));
        Matrix<double> testX = test_data.get_submatrix_columns(Vector<size_t>(1,1,test_data.get_columns_number()-1));
        X = (X - 127.5) / 127.5;
        testX = (testX - 127.5) / 127.5;
        Vector<size_t> index_vector(0,1,X.get_rows_number());
        for (int epoch = 0; epoch<n_epochs; epoch++) {
            Matrix<double> random_batch = X.get_submatrix_rows(index_vector.get_subvector_random(batch_size));
            auto loss = this->autoencoder->train_on_batch(&random_batch, &random_batch);
            cout << epoch << "[D loss: " << setprecision(6) << loss << "]"<<endl;
            if (epoch % save_interval == 0)
                this->save_images(epoch, &testX);
        }
    }

};


int main (int argc, char **argv) {
    Autoencoder ae;
    ae.train(200000, 128, 40);
}