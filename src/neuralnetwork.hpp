#include <cstdlib> 
#include <cstdio> 
#include <ctime> 
#include <vector>

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
void calculate_mse(NeuralNetwork* currentNeuralNetwork);
void outputlayer_backpropagation(NeuralNetwork* currentNeuralNetwork);
void hiddenlayer_backpropagation(NeuralNetwork* currentNeuralNetwork);
void backpropagation(NeuralNetwork* currentNeuralNetwork);
void epoch(NeuralNetwork* currentNeuralNetwork);



// I'm implement it considering that the connections will be placed on the afterward Neurons, check the image on assets
struct Connection {
    unsigned int id;
    Neuron* backwardNeuron;
    double weight;
    Neuron* afterwardNeuron;
    Connection* next;

    Connection(){
        id = 0;
        backwardNeuron = nullptr;
        weight = 0.0;
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
    double output; // neuron output represents the input dataset or the prediction (neuron value)
    double target; // target is the value that should be on output
    double deltaLossFunction; // error gradient loss function
    double deltaActivationFunction; // error gradient activation function
    double deltaWeight; // error gradient respecting weight
    double sumOfNextLayerGradients; 
    double bias;
    Connection* connections; // edges connecting two neurons using weights 
    Neuron* next;

    Neuron() {
        id = 0;
        output = 0;
        bias = 0;
        target = 0;
        deltaLossFunction = 0;
        deltaActivationFunction = 0;
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
    bool fed;
    Neuron* neurons;
    Layer* previous;
    Layer* next;

    Layer(){
        id = 0;
        neuronsQuantity = 0;
        inputLayer = false;
        outputLayer = false;
        fed = false;
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
    double mse;
    double biggestError;
    Layer* layers;
    Layer* inputLayer;

    NeuralNetwork(unsigned int id, unsigned int layersQuantity, double learningRate){
        id = id;
        layersQuantity = layersQuantity;
        learningRate = learningRate;
        mse = 0.0;
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

    if(afterwardNeuron->connections == nullptr){
        afterwardNeuron->connections = newConnection;
        return;
    }

    // Taking the first connection pointer of afterward Neuron
    Connection* currentConnection = afterwardNeuron->connections;
    
    // Passing through all connections from afterwards neuron to take last connection
    while(currentConnection->next != nullptr){
             currentConnection = currentConnection->next; // Changing the connection
    }

    currentConnection->next = newConnection; // Adding a new connection to afterward Neuron

}

void add_neuron(Layer* currentLayer, unsigned int id){
    Neuron* newNeuron = new Neuron();
    newNeuron->id = id; // Setting the id

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

void testing_next(NeuralNetwork* currentNeuralNetwork){
    for(Layer* currentLayer = currentNeuralNetwork->inputLayer; currentLayer != nullptr; currentLayer = currentLayer->next){
        cout << "Entrou aqui" << endl;
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
            add_neuron(currentLayer, j);
            currentLayer->neuronsQuantity++;
        }
        currentLayer = currentLayer->previous;
    }
}

void load_connections(NeuralNetwork* currentNeuralNetwork){
    for(Layer* currentLayer = currentNeuralNetwork->layers; !(currentLayer->inputLayer); currentLayer = currentLayer->previous){
        int i = 0;
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            for(Neuron* previousNeuron = currentLayer->previous->neurons; previousNeuron != nullptr; previousNeuron = previousNeuron->next){
                double weight = static_cast<double>((rand() % 201) - 100) / 100;
                connect(i, previousNeuron, weight, currentNeuron);
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
            currentNeuron->output = input;
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
    for(Layer* currentLayer = currentNeuralNetwork->inputLayer->next; currentLayer != nullptr; currentLayer = currentLayer->next){
        // For each neuron on layer, update its output considering the sum of weights multiplied by output neuron value from previous layer
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
                currentConnection->afterwardNeuron->output += currentConnection->weight * currentConnection->backwardNeuron->output;
            }
            currentNeuron->output += currentNeuron->bias;
            currentNeuron->output = sigmoid(currentNeuron->output);
        }
    }

}
                                                                                    /* PRINT FUNCTIONS */
void show_network_error(NeuralNetwork* currentNeuralNetwork){
    cout << "The network error is: "<< currentNeuralNetwork->mse << endl;
}

void show_network_biggest_error(NeuralNetwork* currentNeuralNetwork){
    cout << "The biggest error is: " << currentNeuralNetwork->biggestError << endl;
}

void print_layer(Layer* currentLayer) {
    cout << "\033[38;2;255;255;255m";
    cout << "Layer " << currentLayer->id << ":" << endl;
    Neuron* currentNeuron = currentLayer->neurons;
    while (currentNeuron != nullptr) {
        cout << "\033[38;2;231;76;60m";
        cout << "\tNeuron " << currentNeuron->id << ": " << currentNeuron->output << " - target: " << currentNeuron->target << endl;
        Connection* currentConnection = currentNeuron->connections;
        while(currentConnection != nullptr){
            std::cout << "\033[38;2;41;128;185m";
            cout << "\t\tConnection " << currentConnection->id << ": (" << currentConnection->backwardNeuron->id << ") _ " << 
            currentConnection->weight << " _ (" << currentConnection->afterwardNeuron->id << ")" << endl;
            currentConnection = currentConnection->next;
        }
        currentNeuron = currentNeuron->next;
    }
}

void print_nn(NeuralNetwork* currentNeuralNetwork){
    for(Layer* currentLayer = currentNeuralNetwork->layers; currentLayer != nullptr; currentLayer = currentLayer->previous){
        print_layer(currentLayer);
    }
}
                                                                                    /* ACTIVATION FUNCTIONS */
double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

                                                                                   /* DERIVATIVE OF ACTIVATION FUNCTION */
// remember to check if the derivative is correct
double sigmoid_derivative(double x){
    double output = sigmoid(x);
    return output * (1 - output);
}

                                                                                    /* ERROR VERIFICATION WITH MSE */
void calculate_mse(NeuralNetwork* currentNeuralNetwork){
    Layer* outputLayer = currentNeuralNetwork->layers;

    double totalError = 0.0;
    for(Neuron* currentNeuron = outputLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
        double error = (currentNeuron->output - currentNeuron->target);
        totalError += error * error;
    }

    currentNeuralNetwork->mse = totalError/outputLayer->neuronsQuantity;

    if(currentNeuralNetwork->mse > currentNeuralNetwork->biggestError){
        currentNeuralNetwork->biggestError = currentNeuralNetwork->mse;
    }

}

                                                                                    /* BACKPROGATION PROCCES */
void calculate_gradient_outputlayer(NeuralNetwork* currentNeuralNetwork){
    Layer* outputLayer = currentNeuralNetwork->layers;

    for(Neuron* currentNeuron = outputLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
        currentNeuron->deltaLossFunction = (currentNeuron->output - currentNeuron->target);
        currentNeuron->deltaActivationFunction = sigmoid_derivative(currentNeuron->output);
        currentNeuron->deltaWeight = currentNeuron->deltaLossFunction * currentNeuron->deltaActivationFunction;
    }

}

void calculate_gradient_hiddenlayers_and_update_weights(NeuralNetwork* currentNeuralNetwork){
    
    for(Layer* currentLayer = currentNeuralNetwork->layers; currentLayer != nullptr;  currentLayer = currentLayer->next) {
       
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
                currentConnection->weight +=  ((-currentNeuralNetwork->learningRate) * currentConnection->afterwardNeuron->deltaWeight);
            }
            currentNeuron->bias += (-currentNeuralNetwork->learningRate) * currentNeuron->deltaLossFunction;

        }
        
        for(Neuron* currentNeuron = currentLayer->neurons; currentNeuron != nullptr; currentNeuron = currentNeuron->next){
            for(Connection* currentConnection = currentNeuron->connections; currentConnection != nullptr; currentConnection = currentConnection->next){
                currentConnection->backwardNeuron->deltaWeight += currentConnection->afterwardNeuron->deltaWeight * currentConnection->weight;

            }
        }
    }
}

void backpropagation(NeuralNetwork* currentNeuralNetwork){
    calculate_gradient_outputlayer(currentNeuralNetwork);
    calculate_gradient_hiddenlayers_and_update_weights(currentNeuralNetwork);
}

void epoch(NeuralNetwork* currentNeuralNetwork){
    feed_forward(currentNeuralNetwork);
    calculate_mse(currentNeuralNetwork);
    show_network_error(currentNeuralNetwork);
    backpropagation(currentNeuralNetwork);
}
