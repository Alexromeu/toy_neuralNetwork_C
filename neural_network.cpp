#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>


#define MAX_NEURONS_PER_LAY 100
#define TOTAL_LAYERS 4
#define INPUT_LAYER_MAXVAL 10
#define OUTPUT_LAYER_MAXVAL 2

struct Neuron {
    float value;
    float bias;
    std::vector<float> weights;
};

struct Layer {
    int count;
    std::vector<Neuron*> neurons;
};


class Neural_Network {
    public:
    Neural_Network();
    ~Neural_Network();

    void initialize_network();
    void forward();                  // one forward pass + cost, for demo
    std::vector<Layer*> network;     // matrix of layers

    private:
    size_t num_layers = TOTAL_LAYERS;
    Layer* output_layer = {};
    float xavier_init(int n_in, int n_out);    
    float random_num(float min, float max);
    float sigmoid(float x);
    float derivate_sigmoid(float activation);
    float feed_neuron(Neuron *neuron, const std::vector<float> &input_layer);   //return activation a 
    void layer_processing(Layer *in_layer, Layer *out_layer);                   //this is done when fordward()
    void print_layer(Layer *layer);
    std::vector<float> mean_error(const std::vector<float> &desire_output);     // feed with desire output returns the squared error
    std::vector<float> set_desire_layer();                                      //random just for testing,  changing is crusial for real examples
    float cost(const std::vector<float> &sqrd_diff_array);                      // sqrd_diff_array -> come from mean_error() its what want to decrease 
    float cost_gradient();                                                      // its what we need to decrease
};


Neural_Network::Neural_Network() {
    srand((unsigned)time(nullptr));
    initialize_network();
}

Neural_Network::~Neural_Network() {
    for (Layer *layer : network) {
        for (Neuron *n : layer->neurons) {
            delete n;
        }
        delete layer;
    }
}


void Neural_Network::initialize_network() {
    for (size_t i = 0; i < num_layers; i++) {
        Layer *layer = new Layer();

        if (i == 0) {
            layer->count = INPUT_LAYER_MAXVAL;
        }
        else if (i == num_layers - 1) {
            layer->count = OUTPUT_LAYER_MAXVAL;
        }
        else {
            layer->count = MAX_NEURONS_PER_LAY;
        }

        int prev_count = (i > 0) ? network[i - 1]->count : 0;

        for (int j = 0; j < layer->count; j++) {
            Neuron *n = new Neuron();
            n->value = 0.0f;
            n->bias = 0.0f;

            if (i > 0) {
                n->weights.resize(prev_count);
                for (int k = 0; k < prev_count; k++) {
                    n->weights[k] = xavier_init(prev_count, layer->count);
                }
            }

            layer->neurons.push_back(n);
        }

        network.push_back(layer);
    }

    output_layer = network.back();
}


float Neural_Network::feed_neuron(Neuron *neuron, const std::vector<float> &input_layer) {
    float result = 0.0f;

    for (size_t i = 0; i < input_layer.size(); i++) {
        result += neuron->weights[i] * input_layer[i];
    }

    return result + neuron->bias; // res to pass to next layer neuronS
}

// this processes layer 1 and passes data to layer 2 editing layer 2
// same process repeats in layer 3 inheriting data from 2
void Neural_Network::layer_processing(Layer *in_layer, Layer *out_layer) {
    std::vector<float> input_values = {};

    for (int i = 0; i < in_layer->count; i++) {
        input_values.push_back(in_layer->neurons[i]->value);
    }

    for (int lo = 0; lo < out_layer->count; lo++) {
        float raw_sum = feed_neuron(out_layer->neurons[lo], input_values);
        out_layer->neurons[lo]->value = sigmoid(raw_sum);
    }
}


float Neural_Network::random_num(float min, float max) {
    float scale = rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

float Neural_Network::xavier_init(int n_in, int n_out) {
    float limit = sqrtf(6.0f / (n_in + n_out));
    return random_num(-limit, limit);
}

float Neural_Network::sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float Neural_Network::derivate_sigmoid(float a) {
    return sigmoid(a) * (1 - sigmoid(a));
}



void Neural_Network::print_layer(Layer *layer) {
    printf("==========\n");
    printf("Neurons: %d\n", layer->count);

    for (int i = 0; i < layer->count; i++) {
        Neuron *n = layer->neurons[i];

        printf("Neuron %d:\n", i);
        printf("  value = %f\n", n->value);
        printf("  bias  = %f\n", n->bias);
        printf("  weights: ");

        for (size_t w = 0; w < n->weights.size(); w++) {
            printf("%f ", n->weights[w]);
        }

        printf("\n");
    }

    printf("\n");
}


std::vector<float> Neural_Network::mean_error(const std::vector<float> &desire_output) {
    std::vector<float> squared_differences = {};

    for (int neuron = 0; neuron < OUTPUT_LAYER_MAXVAL; neuron++) {
        float diff = desire_output[neuron] - output_layer->neurons[neuron]->value;
        squared_differences.push_back(diff * diff);
    }

    return squared_differences;
}

std::vector<float> Neural_Network::set_desire_layer() {
    std::vector<float> desire_output = {};

    for (int i = 0; i < OUTPUT_LAYER_MAXVAL; i++) {
        desire_output.push_back(random_num(1, 5)); // provisional
    }

    return desire_output;
}


float Neural_Network::cost(const std::vector<float> &sqrd_diff_array) {
    float result = 0.0f;
    for (size_t i = 0; i < sqrd_diff_array.size(); i++) {
        result += sqrd_diff_array[i];
    }

    return result;
}


void Neural_Network::forward() {
    // feed some input values into the input layer
    for (int i = 0; i < network[0]->count; i++) {
        network[0]->neurons[i]->value = random_num(0.0f, 1.0f);
    }

    // propagate through every subsequent layer
    for (size_t i = 1; i < network.size(); i++) {
        layer_processing(network[i - 1], network[i]);
    }

    print_layer(output_layer);

    std::vector<float> desire = set_desire_layer();  //this has to be done manualy an idea could be tokenize text of course
    std::vector<float> errors = mean_error(desire);
    printf("cost = %f\n", cost(errors));
}

float Neural_Network::cost_gradient() {
    return 0.0;
}


int main() {
    Neural_Network nn;
    nn.forward();
    return 0;
}
