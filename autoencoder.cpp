//
// Created by Robbie on 5/3/2019.
//
#include <iostream>
#include <iomanip>
#include "./OpenNN/matrix.h"
#include "./OpenNN/vector.h"
#include "optimizer.h"
#include "loss_function.h"
#include "layer.h"
#include "neural_network.h"

using namespace std;
using namespace OpenNN;

#define trace(input) do { if (1) { cout << input << endl; } } while(0)
#define here() do { cout << "  here" << endl; } while (0)
#define there() do { cout << "  there" << endl; } while (0)

class Autoencoder {
    int img_rows = 28;
    int img_cols = 28;
    int img_dim = img_rows * img_cols;
    int latent_dim = 32;
    Optimizer optimizer;
    Loss *loss_function;
    NeuralNetwork* autoencoder;

    NeuralNetwork build_encoder(Optimizer optimizer, Loss* loss_function) {
        NeuralNetwork *encoder = new NeuralNetwork(optimizer, loss_function);
        Dense* dense1 = new Dense(512, Shape(this->img_dim, 1));
        Activation* act1 = new Activation("leaky_relu");
        BatchNormalization* batch1 = new BatchNormalization(0.8);
        Dense* dense2 = new Dense(256, Shape(1, 1));
        Activation* act2 = new Activation("leaky_relu");
        BatchNormalization* batch2 = new BatchNormalization(0.8);
        Dense* dense3 = new Dense(this->latent_dim, Shape(this->img_dim, 0));
        encoder->add(dense1);
        encoder->add(act1);
        encoder->add(batch1);
        encoder->add(dense2);
        encoder->add(act2);
        encoder->add(batch2);
        encoder->add(dense3);
        return *encoder;
    }
    NeuralNetwork build_decoder(Optimizer optimizer, Loss* loss_function) {
        NeuralNetwork decoder(optimizer, loss_function);
        Dense* dense1 = new Dense(256, Shape(this->latent_dim, 1));
        Activation* act1 = new Activation("leaky_relu");
        BatchNormalization* batch1 = new BatchNormalization(0.8);
        Dense* dense2 = new Dense(512, Shape(1, 1));
        Activation* act2 = new Activation("leaky_relu");
        BatchNormalization* batch2 = new BatchNormalization(0.8);
        Dense* dense3 = new Dense(this->img_dim, Shape(1, 1));
        Activation* act3 = new Activation("tanh");
        decoder.add(dense1);
        decoder.add(act1);
        decoder.add(batch1);
        decoder.add(dense2);
        decoder.add(act2);
        decoder.add(batch2);
        decoder.add(dense3);
        decoder.add(act3);
        return decoder;
    }
    void save_images(int epoch, Matrix<double>* X) {

    }
public:
    Autoencoder() {
        this->loss_function = new SquareLoss();
        NeuralNetwork encoder = this->build_encoder(this->optimizer, this->loss_function);
        NeuralNetwork decoder = this->build_decoder(this->optimizer, this->loss_function);
        autoencoder = new NeuralNetwork(this->optimizer, this->loss_function);
        autoencoder->layers.insert(autoencoder->layers.end(), encoder.layers.begin(), encoder.layers.end());
        autoencoder->layers.insert(autoencoder->layers.end(), decoder.layers.begin(), decoder.layers.end());
        autoencoder->output_dim = this->img_dim;
        autoencoder->summary("Variational Autoencoder");
    }

    void train(int n_epochs, int batch_size, int save_interval) {
        this->autoencoder->batch_size = batch_size;
        trace("Loading MNIST data..."<<endl);
        Matrix<double> train_data("../mnist_train.csv", ',', true);
        Matrix<double> test_data("../mnist_test.csv", ',', true);
        Matrix<double> X = train_data.get_submatrix_columns(Vector<size_t>(1,1,train_data.get_columns_number()-1));
        Matrix<double> testX = test_data.get_submatrix_columns(Vector<size_t>(1,1,test_data.get_columns_number()-1));
        X = (X - 127.5) / 127.5;
        testX = (testX - 127.5) / 127.5;
        Vector<size_t> index_vector(0,1,X.get_rows_number());
        for (int epoch = 0; epoch<n_epochs; epoch++) {
            Matrix<double> random_batch = X.get_submatrix_rows(index_vector.get_subvector_random(batch_size));
            trace(1);
            auto loss = this->autoencoder->train_on_batch(&random_batch, &random_batch);
            trace(1);
            cout << epoch << "[D loss: " << setprecision(6) << loss << "]"<<endl;
            trace(1);
            if (epoch % save_interval == 0)
                this->save_images(epoch, &testX);
        }
    }

};


int main (int argc, char **argv) {
    Autoencoder ae;
    ae.train(200000, 128, 40);
}