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

#include "./neuralnetwork.hpp"

int main(int argc, char* argv[]){
    srand(time(NULL));

    int id = 0, learningRate = 0.01;
    // Check if there are enough command line arguments
    int num_layers = 0;
    int num_neurons_per_layer[argc-5];
    int layer_index = 0;

    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "-l") == 0) {
            num_layers = atoi(argv[i+1]);
            i++;
        } else if(strcmp(argv[i], "-n") == 0) {
            for(int j = 0; j < num_layers; j++) {
                num_neurons_per_layer[j] = atoi(argv[i+j+1]);
            }
            i += num_layers;
        }
    }

    if(num_layers == 0) {
        cout << "Error: -l and -n arguments must be provided" << endl;
        return 1;
    }

    NeuralNetwork* neuralnetwork = new NeuralNetwork(id, num_layers, learningRate);
    
    load_layers(neuralnetwork, num_layers);
    load_neurons(neuralnetwork, num_neurons_per_layer, num_layers);
    load_connections(neuralnetwork);
    //print_nn(neuralnetwork);

    string line = "";
    double input_neurons[num_neurons_per_layer[0]];
    double output_layer_targets[num_neurons_per_layer[num_layers -1]];
    /// TREINAMENTO DAS ENTRADAS
    ifstream trainFile;
    string trainFilePath = "../training/nn-training.csv";
    trainFile.open(trainFilePath);

    while(getline(trainFile, line)){
        string tempStringTrain = "";
        double trainData;

        stringstream inputString(line);
        for(int i = 0; i < (num_neurons_per_layer[0] + num_neurons_per_layer[num_layers - 1]); i++){
            getline(inputString, tempStringTrain, ',');
            trainData = atof(tempStringTrain.c_str());
            if(i < num_neurons_per_layer[0])
                input_neurons[i] = trainData;
            else{
                output_layer_targets[i - num_neurons_per_layer[0]] = trainData;
            }

            string tempStringTrain = "";
        }
            
        feed_inputs(neuralnetwork, input_neurons, num_neurons_per_layer);
        

        feed_targets(neuralnetwork, output_layer_targets, num_layers, num_neurons_per_layer);
        

        /// Training with backpropagation
        //feed_forward(neuralnetwork);
        epoch(neuralnetwork);
        line = "";
    }
    print_nn(neuralnetwork);
    show_network_biggest_error(neuralnetwork);
        
    return 0;
}