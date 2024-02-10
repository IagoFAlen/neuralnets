#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <vector>
#include <cmath>
#include <cassert>

using namespace std;

// STRUCTS DECLARATIONS
struct Connection;
struct Neuron;
struct Layer;
struct NeuralNetwork;

// FUNCTIONS DECLARATIONS
void connect(unsigned int id, Neuron* backwardNeuron, double weight, Neuron* afterwardNeuron);
void add_neuron(Layer* currentLayer, unsigned int id);
void add_layer(NeuralNetwork* currentNeuralNetwork, unsigned int id);
void update_input(Layer* inputLayer, unsigned int neuronID, double input);
void update_target(Layer* outputLayer, unsigned int neuronID, double target);
void feed_inputs(NeuralNetwork* currentNeuralNetwork, vector<double> inputs);
void feed_targets(NeuralNetwork* currentNeuralNetwork, vector<double> targets);
void feed_forward(NeuralNetwork* currentNeuralNetwork);
void show_network_error(NeuralNetwork* currentNeuralNetwork);
double sigmoid(double x);
double sigmoid_derivative(double x);
void backpropagation(NeuralNetwork* currentNeuralNetwork);
void epoch(NeuralNetwork* currentNeuralNetwork);
void print_derivatives_output_layer(NeuralNetwork* currentNeuralNetwork);
void print_derivatives_hidden_layer(NeuralNetwork* currentNeuralNetwork);
void print_nn_io(NeuralNetwork* currentNeuralNetwork);
void print_nn_io(NeuralNetwork* currentNeuralNetwork);


// I'm implement it considering that the connections will be placed on the backward Neurons
struct Connection {
    unsigned int id;
    Neuron* backwardNeuron;
    double weight;
    double dWeight;
    double dEtotal;
    Neuron* afterwardNeuron;
    Connection* next;

    Connection(){
        id = 0;
        backwardNeuron = nullptr;
        weight = 0.0;
        dWeight = 0.0;
        dEtotal = 0.0;
        afterwardNeuron = nullptr;
        next = nullptr;
    }
    ~Connection() {
        if(next != nullptr)
            delete next;
    }
};

struct Neuron {
    unsigned int id;
    double activeOutput; // neuron output represents the input dataset or the prediction (neuron value - (a))
    double output; // z
    double target; // target is the value that should be on output
    double dError;
    double deltaLossFunction; // error gradient loss function
    double deltaActivationFunction; // error gradient activation function
    double deltaZ;
    double deltaWeight; // error gradient respecting weight
    double sumOfNextLayerGradients;
    double bias;
    Connection* connections; // edges connecting two neurons using weights
    Neuron* next;

    Neuron() {
        id = 0;
        activeOutput = 0;
        output = 0;
        bias = 0;
        target = 0;
        dError = 0;
        deltaLossFunction = 0;
        deltaActivationFunction = 0;
        deltaZ = 0;
        deltaWeight = 0;
        sumOfNextLayerGradients = 0;
        connections = nullptr;
        next = nullptr;
    }
    ~Neuron() {
        if(next != nullptr)
            delete next;
    }
};

struct Layer{
    unsigned int id;
    unsigned int neuronsQuantity;
    bool inputLayer;
    bool outputLayer;
    Neuron* neurons;
    Layer* previous;
    Layer* next;

    Layer(){
        id = 0;
        neuronsQuantity = 0;
        inputLayer = false;
        outputLayer = false;
        neurons = nullptr;
        previous = nullptr;
        next = nullptr;
    }
    ~Layer(){
        if(previous != nullptr)
            delete previous;
    }

};

struct NeuralNetwork{
    unsigned int id;
    unsigned int layersQuantity;
    double learningRate;
    double cost;
    double biggestError;
    Layer* layers;
    Layer* inputLayer;

    NeuralNetwork(unsigned int id, unsigned int layersQuantity, double learningRate){
        id = id;
        layersQuantity = layersQuantity;
        learningRate = learningRate;
        cost = 0.0;
        biggestError = 0.0;
        layers = nullptr;
        inputLayer = nullptr;
    }

    ~NeuralNetwork(){
        if(layers != nullptr)
            delete layers;
    }

};

                                                                                    /* INSERTIONS */

void connect(unsigned int id, Neuron* backwardNeuron, double weight, Neuron* afterwardNeuron){
    Connection* newConnection = new Connection();     // Creating a new Connection

    newConnection->id = id; // Setting the id
    newConnection->weight = weight; // Setting the weight

    // Checking if backward Neuron exists
    if(backwardNeuron)
        newConnection->backwardNeuron = backwardNeuron;
    else
        // Backward Neuron is null
        cout << "Backward Neuron can't be found because it's null" << endl;

    // Checking if afterward Neuron exists
    if(afterwardNeuron)
        newConnection->afterwardNeuron = afterwardNeuron;
    else
        // Afterward Neuron is null
        cout << "Afterward Neuron can't be found because it's null" << endl;

    newConnection->next = nullptr;

    if(backwardNeuron->connections == nullptr){
        backwardNeuron->connections = newConnection;
        return;
    }

    // Taking the first connection pointer of backward Neuron
    Connection* currentConnection = backwardNeuron->connections;

    // Passing through all connections from afterwards neuron to take last connection
    while(currentConnection->next != nullptr){
             currentConnection = currentConnection->next; // Changing the connection
    }

    currentConnection->next = newConnection; // Adding a new connection to afterward Neuron

}

void add_neuron(Layer* currentLayer, unsigned int id, double bias){
    Neuron* newNeuron = new Neuron();
    newNeuron->id = id; // Setting the id
    newNeuron->bias = bias; // Bias

    if(!currentLayer) // Checking if the layer exists
        return;

    // Checking if it's first neuron
    if(currentLayer->neurons == nullptr){
        currentLayer->neurons = newNeuron;
        return;
    }

    // Taking the first neuron pointer of layer
    Neuron* currentNeuron = currentLayer->neurons;

    // Passing through all neurons from layer to take last neuron
    while(currentNeuron->next != nullptr){
        currentNeuron = currentNeuron->next;
    }

    currentNeuron->next = newNeuron; // Adding a new neuron to layer

}

// YOU NEED TO BEGIN ADDING THE OUTPUT LAYER
void add_layer(NeuralNetwork* currentNeuralNetwork, unsigned int id){
    Layer* newLayer = new Layer();
    newLayer->id = id; // Setting the id

    if(!currentNeuralNetwork) // Checking if the Neural Network exists
        return;


    // Checking if it's first layer (output layer)
    if(currentNeuralNetwork->layers == nullptr){
        newLayer->outputLayer = true;
        currentNeuralNetwork->layers = newLayer;
        return;
    }

    // Checking if it's last layer (input layer)
    if(id == 0){
        newLayer->inputLayer = true;
        currentNeuralNetwork->inputLayer = newLayer;
    }

    // Taking the first layer pointer of neural network (output layer) - we're starting from the end
    Layer* currentLayer = currentNeuralNetwork->layers;

    // Passing through all layers from neural network to take last layer (input layer)
    while(currentLayer->previous != nullptr){
        currentLayer = currentLayer->previous;
    }

    currentLayer->previous = newLayer; // Adding a new layer to neural network
    currentLayer->previous->next = currentLayer;
    if(currentLayer->previous->inputLayer){
        currentNeuralNetwork->inputLayer = currentLayer->previous;
    }
}
                                                                                    /* LOAD CONFIGS */
void load_layers(NeuralNetwork* currentNeuralNetwork, int num_layers){
    for(int i = num_layers - 1; i >= 0; i--){
        add_layer(currentNeuralNetwork, i);
    }
}

void load_neurons(NeuralNetwork* currentNeuralNetwork, int num_neurons_per_layer[], int num_layers){
    Layer* currentLayer = currentNeuralNetwork->layers;

    for(int i = num_layers - 1; i >= 0; i--){
        for(int j = 0; j < num_neurons_per_layer[i]; j++){
            double bias = static_cast<double>((rand() % 201) - 100) / 100;
            add_neuron(currentLayer, j, bias);
            currentLayer->neuronsQuantity++;
        }
        currentLayer = currentLayer->previous;
    }
}

void load_connections(NeuralNetwork* currentNeuralNetwork){
    for(Layer* currentLayer = currentNeuralNetwork->inputLayer; !(currentLayer->outputLayer); currentLayer = currentLayer->next){
        int i = 0;

        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            for(Neuron* afterwardNeuron = currentLayer->next->neurons; afterwardNeuron != nullptr; afterwardNeuron = afterwardNeuron->next){
                double weight = static_cast<double>((rand() % 201) - 100) / 100;
                connect(i, currentNeuron, weight, afterwardNeuron);
                i++;
            }
        }
    }
}

                                                                       /* UPDATE */
void update_input(Layer* inputLayer, unsigned int neuronID, double input){
    // For each neuron, search the one with the same id and set a new value to neuron
    for(Neuron* currentNeuron = inputLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
        if(currentNeuron->id == neuronID){
            currentNeuron->activeOutput = input;
            return;
        }
    }
}

void update_target(Layer* outputLayer, unsigned int neuronID, double target){
    // For each neuron, search the one with the same id and set a new value to target
    for(Neuron* currentNeuron = outputLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
        if(currentNeuron->id == neuronID){
            currentNeuron->target = target;
            return;
        }
    }
}

                                                                                        /* FEED */
void feed_inputs(NeuralNetwork* currentNeuralNetwork, double inputs[], int num_neurons_per_layer[]){
    // Take the current layer and update its inputs neuron values
    Layer* currentLayer = currentNeuralNetwork->inputLayer;

    for(int i = 0; i < num_neurons_per_layer[0]; i++){
        update_input(currentLayer, i, inputs[i]);
    }

}

void feed_targets(NeuralNetwork* currentNeuralNetwork, double targets[], int num_layers, int num_neurons_per_layer[]){
    Layer* currentLayer = currentNeuralNetwork->layers;

    for(int i = 0; i < num_neurons_per_layer[num_layers - 1]; i++){
        update_target(currentLayer, i, targets[i]);
    }
}

void feed_forward(NeuralNetwork* currentNeuralNetwork){
    for(Layer* currentLayer = currentNeuralNetwork->inputLayer; currentLayer != nullptr; currentLayer = currentLayer->next){
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            if(!(currentLayer->inputLayer)){
                currentNeuron->output += currentNeuron->bias;
                currentNeuron->activeOutput = sigmoid(currentNeuron->output);
            }

            for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
                currentConnection->afterwardNeuron->output += currentConnection->weight * currentConnection->backwardNeuron->activeOutput;
            }
        }
    }

}

void update_bias_manually(NeuralNetwork* currentNeuralNetwork){
    for(Layer* currentLayer = currentNeuralNetwork->inputLayer; currentLayer != nullptr; currentLayer = currentLayer->next){
        cout << "Atualizações na Layer " << currentLayer->id << ":" << endl;
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            cout << "\tAtualizações do neurônio " << currentNeuron->id << ":" << endl;
            double newBias = 0.0;
            cin >> newBias;
            currentNeuron->bias = newBias;
        }
    }
}

/* update connections manually*/
void update_weights_manually(NeuralNetwork* currentNeuralNetwork){
    for(Layer* currentLayer = currentNeuralNetwork->inputLayer; currentLayer != nullptr; currentLayer = currentLayer->next){
        cout << "Atualizações na Layer " << currentLayer->id << ":" << endl;
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            cout << "\tAtualizações do neurônio " << currentNeuron->id << ":" << endl;
            for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
                cout << "\t\tAtualizando a conexão entre (" << currentConnection->backwardNeuron->id << ") e (" << currentConnection->afterwardNeuron->id << ")" << endl; 
                double newWeight = 0.0;
                cin >> newWeight;
                currentConnection->weight = newWeight;
            }
        }
    }
}

void update_weights_using_arrays(NeuralNetwork* currentNeuralNetwork){
    double  weightsUpdates[] = {0.30, 0.13, 0.25, 0.42, 0.67, 0.84, 0.54, 0.52}; //These values are just for example purposes
    int i = 0;
    for(Layer* currentLayer = currentNeuralNetwork->inputLayer; currentLayer != nullptr; currentLayer = currentLayer->next){
        //cout << "Atualizações na Layer " << currentLayer->id << ":" << endl;
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            //cout << "\tAtualizações do neurônio " << currentNeuron->id << ":" << endl;
            for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
                //cout << "\t\tAtualizando a conexão entre (" << currentConnection->backwardNeuron->id << ") e (" << currentConnection->afterwardNeuron->id << ")" << endl; 
                double newWeight = weightsUpdates[i];
                currentConnection->weight = newWeight;
                i++;
            }
        }
    }
}

void update_bias_using_arrays(NeuralNetwork* currentNeuralNetwork){
    double  biasUpdates[] = {0.0, 0.42, 0.67}; //These values are just for example purposes
    int i = 0;
    for(Layer* currentLayer = currentNeuralNetwork->inputLayer; currentLayer != nullptr; currentLayer = currentLayer->next){
        //cout << "Atualizações na Layer " << currentLayer->id << ":" << endl;
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            //cout << "\tAtualizações do bias no neurônio " << currentNeuron->id << ":" << endl;
            currentNeuron->bias = biasUpdates[i];
        }
        i++;
    }
}

                                                                                    /* PRINT FUNCTIONS */
void show_network_error(NeuralNetwork* currentNeuralNetwork){
    //cout << "The network error is: "<< currentNeuralNetwork->cost << endl;
    cout << currentNeuralNetwork->cost << endl;

}

void show_network_biggest_error(NeuralNetwork* currentNeuralNetwork){
    cout << "The biggest error is: " << currentNeuralNetwork->biggestError << endl;
}

void print_nn_io(NeuralNetwork* currentNeuralNetwork){
    cout << "----------------------------------------------------------------------------------------------------------------" << endl;
    for(Layer* currentLayer = currentNeuralNetwork->inputLayer; currentLayer != nullptr; currentLayer = currentLayer->next){
        cout << "Layer " << currentLayer->id << ":" << endl;
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            cout << "\tNeuron " << currentNeuron->id << ": " << currentNeuron->activeOutput << " - target: " << currentNeuron->target  << " - bias: " << currentNeuron->bias 
            << " - Loss Function gradient: " << currentNeuron->deltaLossFunction << " - Activation Function gradient: " << currentNeuron->deltaActivationFunction
            << " - Error Gradient: " << currentNeuron->dError << endl;
            for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
                cout << "\t\tConnection " << currentConnection->id << ": ("
                << currentConnection->backwardNeuron->id << ") _ " << currentConnection->weight << " _ (" << currentConnection->afterwardNeuron->id << ")" 
                << " -> Weight gradient: " << currentConnection->dWeight << endl;
            }
        }
    }
}

void print_nn_oi(NeuralNetwork* currentNeuralNetwork){
    cout << "----------------------------------------------------------------------------------------------------------------" << endl;
    for(Layer* currentLayer = currentNeuralNetwork->layers; currentLayer != nullptr; currentLayer = currentLayer->previous){
        cout << "Layer " << currentLayer->id << ":" << endl;
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            cout << "\tNeuron " << currentNeuron->id << ": " << currentNeuron->activeOutput << " - target: " << currentNeuron->target  << " - bias: " << currentNeuron->bias << endl;
            for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
                cout << "\t\tConnection " << currentConnection->id << ": ("
                << currentConnection->backwardNeuron->id << ") _ " << currentConnection->weight << " _ (" << currentConnection->afterwardNeuron->id << ")" << endl;
            }
        }
    }
}
                                                                                    /* ACTIVATION FUNCTIONS */
double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

                                                                                   /* DERIVATIVE OF ACTIVATION FUNCTION */
// remember to check if the derivative is correct
double sigmoid_derivative(double x){
    // x = sigmoid(z)
    return x * (1 - x);
}

void print_derivatives_output_layer(NeuralNetwork* currentNeuralNetwork){
    Layer* outputLayer = currentNeuralNetwork->layers;

    for(Neuron* currentNeuron = outputLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
        cout << "Gradiente da função de custo: " << currentNeuron->deltaLossFunction << endl;
        cout << "Gradiente da função de ativação: " << currentNeuron->deltaActivationFunction << endl;
        cout << "Gradiente do erro: " << currentNeuron->dError << endl;
        for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
            cout << "Atualizou o gradiente do erro para o neurônio "<< currentConnection->backwardNeuron->id << " para: " << currentConnection->backwardNeuron->dError << endl;
        }
    }
}

void print_derivatives_hidden_layer(NeuralNetwork* currentNeuralNetwork){
    Layer* outputLayer = currentNeuralNetwork->layers;

    for(Neuron* currentNeuron = outputLayer->neurons; currentNeuron = nullptr; currentNeuron = currentNeuron->next){
        for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
            cout << "Atualizou o gradiente do erro para o neurônio "<< currentConnection->backwardNeuron->id << " para: " << currentConnection->backwardNeuron->dError << endl;
        }
    }
}

                                                                                    /* ERROR VERIFICATION WITH cost */
void calculate_error_cost(NeuralNetwork* currentNeuralNetwork){
    Layer* outputLayer = currentNeuralNetwork->layers;
    double totalCost = 0.0;

    for(Neuron* currentNeuron = outputLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
        double error = currentNeuron->target - currentNeuron->activeOutput;
        error = (error * error)/ outputLayer->neuronsQuantity;
        //cout << "Error : " << error << endl;
        totalCost +=  error;
    }

    currentNeuralNetwork->cost = totalCost;
    //cout << "Total Error: " <<  totalCost << endl;
    if(currentNeuralNetwork->cost > currentNeuralNetwork->biggestError){
        currentNeuralNetwork->biggestError = currentNeuralNetwork->cost;
    }
}

                                                                                    /* BACKPROGATION PROCESS */
void calculate_output_derivatives(NeuralNetwork* currentNeuralNetwork){
    Layer* outputLayer = currentNeuralNetwork->layers;

    for(Neuron* currentNeuron = outputLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
        // Loss Function, Activation Function and Error Gradients
        currentNeuron->deltaLossFunction        = -2*(currentNeuron->target - currentNeuron->activeOutput)/outputLayer->neuronsQuantity;
        currentNeuron->deltaActivationFunction  = sigmoid_derivative(currentNeuron->activeOutput);
        currentNeuron->dError                   = currentNeuron->deltaActivationFunction * currentNeuron->deltaLossFunction;

    }

}

void calculate_hidden_derivatives(NeuralNetwork* currentNeuralNetwork){
    for(Layer* currentLayer = currentNeuralNetwork->layers->previous; currentLayer != nullptr; currentLayer = currentLayer->previous){
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
                currentConnection->dWeight = currentConnection->afterwardNeuron->dError * currentConnection->afterwardNeuron->activeOutput;
                currentNeuron->dError += (currentConnection->afterwardNeuron->dError * currentConnection->weight); //currentConnection->backwardNeuron->dError +=

            }
            currentNeuron->deltaActivationFunction = sigmoid_derivative(currentNeuron->activeOutput);
            currentNeuron->dError *=  currentNeuron->deltaActivationFunction;
        }
    }
}


void update_backprogated_gradients_to_weights(NeuralNetwork* currentNeuralNetwork){
     for(Layer* currentLayer = currentNeuralNetwork->layers->previous; currentLayer != nullptr; currentLayer = currentLayer->previous){
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
                currentConnection->weight += ((-currentNeuralNetwork->learningRate) * currentConnection->dWeight);
            }
        }
    }
}

void update_backprogated_gradients_to_biases(NeuralNetwork* currentNeuralNetwork){
     for(Layer* currentLayer = currentNeuralNetwork->layers->previous; currentLayer != nullptr; currentLayer = currentLayer->previous){
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
                currentConnection->backwardNeuron->bias += ((-currentNeuralNetwork->learningRate) * currentConnection->afterwardNeuron->dError);
            }
        }
    }
}

void backpropagation(NeuralNetwork* currentNeuralNetwork){
    calculate_output_derivatives(currentNeuralNetwork);
    calculate_hidden_derivatives(currentNeuralNetwork);
    update_backprogated_gradients_to_weights(currentNeuralNetwork);
    update_backprogated_gradients_to_biases(currentNeuralNetwork);
}

void epoch(NeuralNetwork* currentNeuralNetwork){
    feed_forward(currentNeuralNetwork);
    //print_nn_io(currentNeuralNetwork);
    calculate_error_cost(currentNeuralNetwork);
    //print_nn_oi(currentNeuralNetwork);
    show_network_error(currentNeuralNetwork);
    backpropagation(currentNeuralNetwork);


}
