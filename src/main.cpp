#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <vector>
using namespace std;

// Relative to the directory where the program is executed, not to where the binary is
#define TRAIN_FILE "../training/nn-trainingtest.csv"

#include "./neuralnetwork.hpp"

void usage(const char* program){
    cerr << "uso: " << program << " -l <camadas> -n <neuronios por camada...> [-r <learning rate>] [-e <epocas>] [-f <arquivo csv>]" << endl;
    cerr << "ex.: " << program << " -l 3 -n 2 3 1 -r 0.5 -e 1000 -f ../training/nn-trainingtest.csv" << endl;
}

int main(int argc, char* argv[]){
    srand(time(NULL));

    unsigned int id = 0;
    double learningRate = 1e-3; // must be double: an int would truncate 1e-3 to 0 and the network would never learn
    int epochs = 1;
    string trainFilePath = TRAIN_FILE;
    int num_layers = 0;
    vector<int> num_neurons_per_layer; // sized at runtime, so std::vector instead of a VLA (not standard C++)

    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            num_layers = atoi(argv[i+1]);
            i++;
        } else if(strcmp(argv[i], "-n") == 0) {
            if(num_layers <= 0 || i + num_layers >= argc) {
                usage(argv[0]);
                return 1;
            }
            num_neurons_per_layer.clear();
            for(int j = 0; j < num_layers; j++) {
                num_neurons_per_layer.push_back(atoi(argv[i+j+1]));
            }
            i += num_layers;
        } else if(strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            learningRate = atof(argv[i+1]);
            i++;
        } else if(strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            epochs = atoi(argv[i+1]);
            i++;
        } else if(strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            trainFilePath = argv[i+1];
            i++;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if(num_layers < 2 || (int)num_neurons_per_layer.size() != num_layers) {
        cout << "Error: -l (at least 2) and -n arguments must be provided" << endl;
        usage(argv[0]);
        return 1;
    }

    for(int neurons : num_neurons_per_layer) {
        if(neurons <= 0) {
            cout << "Error: every layer needs at least one neuron" << endl;
            return 1;
        }
    }

    ifstream trainFile(trainFilePath);
    if(!trainFile.is_open()) {
        cerr << "Error: could not open " << trainFilePath << endl;
        return 1;
    }

    NeuralNetwork* neuralnetwork = new NeuralNetwork(id, num_layers, learningRate);

    load_layers(neuralnetwork, num_layers);
    load_neurons(neuralnetwork, num_neurons_per_layer.data(), num_layers);
    load_connections(neuralnetwork);
    // Not called anymore: they overwrote the random initialization with fixed values,
    // and identical/patterned weights keep neurons learning the same thing (symmetry).
    //update_weights_using_arrays(neuralnetwork);
    //update_bias_using_arrays(neuralnetwork);
    //print_nn_io(neuralnetwork);

    string line = "";
    vector<double> input_neurons(num_neurons_per_layer[0]);
    vector<double> output_layer_targets(num_neurons_per_layer[num_layers - 1]);
    int columns = num_neurons_per_layer[0] + num_neurons_per_layer[num_layers - 1];

    /// TREINAMENTO DAS ENTRADAS
    // One epoch = one full pass over the file
    for(int e = 0; e < epochs; e++){
        trainFile.clear();  // clear the EOF flag left by the previous epoch
        trainFile.seekg(0); // and go back to the start of the file

        double epochCost = 0.0;
        long samples = 0;

        while(getline(trainFile, line)){
            string tempStringTrain = "";
            stringstream inputString(line);

            for(int i = 0; i < columns; i++){
                getline(inputString, tempStringTrain, ',');
                double trainData = atof(tempStringTrain.c_str());
                if(i < num_neurons_per_layer[0])
                    input_neurons[i] = trainData;
                else
                    output_layer_targets[i - num_neurons_per_layer[0]] = trainData;
            }

            feed_inputs(neuralnetwork, input_neurons.data(), num_neurons_per_layer.data());
            feed_targets(neuralnetwork, output_layer_targets.data(), num_layers, num_neurons_per_layer.data());

            /// Training with backpropagation
            train_step(neuralnetwork);

            epochCost += neuralnetwork->cost;
            samples++;
        }

        // stderr, so "./main ... > cost.txt" still receives only the cost per sample
        if(samples > 0)
            cerr << "epoca " << e + 1 << ": custo medio = " << epochCost / samples << endl;
    }

    //print_nn_io(neuralnetwork);
    //show_network_biggest_error(neuralnetwork);

    delete neuralnetwork;
    return 0;
}
