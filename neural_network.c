#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define MAX_WEIGHTS_PER_NEURON 100
#define MAX_NEURONS_PER_LAY 100
#define TOTAL_LAYERS 4 
#define INPUT_LAYER_MAXVAL 1
#define OUTPUT_LAYER_MAXVAL 1

enum Layer_type {IS_INPUT, IS_HIDDEN, IS_OUTPUT};

//OJO: weights are associated with connections
typedef struct __neuron_t {
    float value;
    float bias;
    float weights[MAX_WEIGHTS_PER_NEURON];
} neuron_t;

typedef struct __input_t {
    float input_values[MAX_NEURONS_PER_LAY];
} input_t;

typedef struct __layer_t {
    int count;
    neuron_t *neurons;   //elements here go after passed to sigmoid... middle layer
} layer_t;  

typedef struct __neural_network_t {
    int num_layers;
    layer_t *layers;
    layer_t *output_layer;
} neural_network_t;

float random_num(float min, float max) {
    float scale = rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

float xavier_init(int n_in, int n_out) { 
    float limit = sqrtf(6.0f / (n_in + n_out)); 
    return random_num(-limit, limit); 
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


void initialize_network(neural_network_t *network) {
    network->layers = malloc(network->num_layers * sizeof(layer_t));

    for (int i = 0; i < network->num_layers; i++) {
        layer_t *layer = &network->layers[i];

        if (i == 0) {
            layer->count = INPUT_LAYER_MAXVAL;
        }
        else if (i == network->num_layers - 1) {
            layer->count = OUTPUT_LAYER_MAXVAL;
        }
        else {
            layer->count = MAX_NEURONS_PER_LAY;
        }

        layer->neurons = malloc(layer->count * sizeof(neuron_t));

        for (int j = 0; j < layer->count; j++) {
            neuron_t *n = &layer->neurons[j];
            n->value = 0;
            n->bias = 0;

            for (int k = 0; k < MAX_WEIGHTS_PER_NEURON; k++) {
                if (i > 0) {
                    int prev = network->layers[i-1].count;
                    int curr = layer->count;
                    n->weights[k] = xavier_init(prev, curr);
                }
            }
        }
    }

    network->output_layer = &network->layers[network->num_layers - 1];
}


float feed_neuron(neuron_t *neuron, float *input_layer, int input_count) {
    int i = 0;
    float result = 0.0f;

    while (i < input_count) {
        result += neuron->weights[i] * input_layer[i]; 
        i++;
    }

    return result + neuron->bias; //res to pass to next layer neuronS
}



//this process layer 1 and pass data to layer 2 editing layer 2
//same process will repeat in layer 3 inheriting data from 2
void layer_processing(layer_t *in_layer, layer_t *out_layer) {
    size_t in_lay_size = in_layer->count;
    float input_values[in_lay_size];
    for (int i = 0; i < in_layer->count; i++) {
        input_values[i] = in_layer->neurons[i].value;
    }

    for (int lo = 0; lo < out_layer->count; lo++) {
        float raw_sum = feed_neuron(&out_layer->neurons[lo], input_values, in_layer->count);
        out_layer->neurons[lo].value = sigmoid(raw_sum);
    }
    
}

void destroy_network(neural_network_t* network) {
    for (int l = 0; l < network->num_layers; l++) {
            free(network->layers[l].neurons);
    }
    free(network->layers);
    free(network);
}

void print_layer(const char *name, layer_t *layer) {
    printf("===== %s =====\n", name);
    printf("Neurons: %d\n", layer->count);

    for (int i = 0; i < layer->count; i++) {
        neuron_t *n = &layer->neurons[i];

        printf("Neuron %d:\n", i);
        printf("  value = %f\n", n->value);
        printf("  bias  = %f\n", n->bias);
        printf("  weights: ");

        for (int w = 0; w < MAX_WEIGHTS_PER_NEURON; w++) {
            printf("%f ", n->weights[w]);
        }

        printf("\n");
    }

    printf("\n");
}

float* mean_error(layer_t *output, float *desire_output) {
    size_t output_size = OUTPUT_LAYER_MAXVAL * sizeof(float);
    float *squared_differences = malloc(output_size);
    for (int neuron = 0; neuron < OUTPUT_LAYER_MAXVAL; neuron++) {
        float diff = desire_output[neuron] - output->neurons[neuron].value;
        float sqr_diff = pow(diff, 2);
        squared_differences[neuron] = sqr_diff;
    }

    return squared_differences;
}

float* set_desire_layer() {
    float *desire_output = malloc(OUTPUT_LAYER_MAXVAL * sizeof(float));

    for (int i = 0; i < OUTPUT_LAYER_MAXVAL; i++) {
        desire_output[i] = random_num(1, 5); //provitional
    }

    return desire_output;
} 

float cost(float* sqrd_diff_array, size_t output_size) {
    float result = 0;
    for (int i = 0; i < output_size; i++) {
        result += sqrd_diff_array[i]; 
    }

    return result;
}

int main(void) {
    srand(time(NULL));
    neural_network_t *network = malloc(sizeof(neural_network_t));
    network->num_layers = TOTAL_LAYERS;
    initialize_network(network);
    float* desire_lay = set_desire_layer();
    layer_t *layer1 = &(network->layers[0]);
    layer_t *layer2 = &(network->layers[1]);
    layer_t *layer3 = &(network->layers[2]);
    layer_t *output_layer = network->output_layer;

    print_layer("Layer input", layer1);
    print_layer("Layer layer2", layer2);
    print_layer("Layer layer3", layer3);
    print_layer("Layer output layer", output_layer);

    layer_processing(layer1, layer2);
    layer_processing(layer2, layer3);
    layer_processing(layer3, output_layer);

    printf("PRINTING OUTPUT: \n");
    for (int i = 0; i < output_layer->count; i++) {
        printf("output #:%d %f\n",i, output_layer->neurons[i].value);
    }

    float* diff = mean_error(output_layer, desire_lay);
    printf("mean error per output:\n");
    for (int k = 0; k < OUTPUT_LAYER_MAXVAL; k++) {
        printf("-> %f\n", diff[k]);
    }

    printf("COST: %f\n", cost(diff, OUTPUT_LAYER_MAXVAL));
    
    free(desire_lay);
    destroy_network(network);
    return 0;
}
//gcc neural_network.c -o neural -lm


//and then figure out how to implement gradient of the cost function here

//experiment:
// the input will be a number from 1 to 10; 
// the output is a float that if bigger that 0.5 num is even