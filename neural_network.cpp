#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <string_view>



struct Neuron {
    float value;
    float bias;
    float delta = 0;
    std::vector<float> weights;   // weights[i] <- neuron i of the PREVIOUS layer
};

struct Layer {
    size_t count;
    std::vector<Neuron*> neurons;
};


class Neural_Network {
    public:
    Neural_Network(size_t neu_per_lay, size_t total_lay, size_t input_size, size_t output_size);
    ~Neural_Network();

    void initialize_network(size_t neu_per_lay, size_t total_lay, size_t input_size, size_t output_size);
    void forward(const std::vector<float> &input);              // silent forward pass
    float forward_cost(const std::vector<float> &input);         // forward + cost, no printing
    float train_step(const std::vector<float> &input, float learning_rate);
    void gradient_check(const std::vector<float> &input, size_t L, size_t j, size_t i);
    void train_network(std::vector<std::vector<float>> &training_data,
                       std::vector<std::vector<float>> &labels,
                       float learning_rate, size_t epocs);

    std::vector<Layer*> network;     // matrix of layers
    std::vector<float> desire_output;
    Layer *output_layer = {};
    void set_desired_output(const std::vector<float> &output_desire);
    void print_layer(Layer *layer);
    std::vector<float> make_estimate(std::vector<float> &input);

    private:
    size_t num_layers;
    float xavier_init(size_t n_in, size_t n_out);
    float random_num(float min, float max);
    float sigmoid(float x, bool derivate = false);
    void feed_input_layer(const std::vector<float> &input);
    float feed_neuron(Neuron *neuron, const std::vector<float> &input_layer);   //return activation a
    void layer_processing(Layer *in_layer, Layer *out_layer);                   //this is done when fordward()
    std::vector<float> mean_error(const std::vector<float> &desire_output);     // feed with desire output returns the squared error
    float cost(const std::vector<float> &sqrd_diff_array);                      // sqrd_diff_array -> come from mean_error() its what want to decrease

    void set_deltas();                       // step 1: blame for the output layer
    void backprop_deltas();                  // step 2: push blame back through layers 2 and 1
    void apply_updates(float learning_rate); // step 3: only now do the weights move
};


Neural_Network::Neural_Network(size_t neu_per_lay, size_t total_lay, size_t input_size, size_t output_size) {
    srand((unsigned)time(nullptr));
    initialize_network( neu_per_lay,  total_lay,  input_size,  output_size);
}

Neural_Network::~Neural_Network() {
    for (Layer *layer : network) {
        for (Neuron *n : layer->neurons) {
            delete n;
        }
        delete layer;
    }
}


void Neural_Network::initialize_network(size_t neu_per_lay, size_t total_lay, size_t input_size, size_t output_size) {
    this->num_layers = total_lay;
    for (size_t i = 0; i < total_lay; i++) {
        Layer *layer = new Layer();

        if (i == 0) {
            layer->count = input_size;
        }
        else if (i == num_layers - 1) {
            layer->count = output_size;
        }
        else {
            layer->count = neu_per_lay;
        }

        size_t prev_count = (i > 0) ? network[i - 1]->count : 0;

        for (size_t j = 0; j < layer->count; j++) {
            Neuron *n = new Neuron();
            n->value = 0.0f;
            n->bias = 0.0f;

            if (i > 0) {
                n->weights.resize(prev_count);
                for (size_t k = 0; k < prev_count; k++) {
                    n->weights[k] = xavier_init(prev_count, layer->count);
                }
            }

            layer->neurons.push_back(n);
        }

        network.push_back(layer);
    }

    output_layer = network.back();
    desire_output.assign(output_layer->count, 0.0f);
}

void Neural_Network::set_desired_output(const std::vector<float> &output_desire) {

    // A silent return here would train every sample against a stale target,
    // so this is fatal on purpose.
    if (output_desire.empty()) {
        fprintf(stderr, "output_desire empty\n");
        exit(EXIT_FAILURE);
    }

    if (output_desire.size() != this->desire_output.size()) {
        fprintf(stderr, "desired output size %zu should match the output layer size %zu\n",
                output_desire.size(), this->desire_output.size());
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < output_desire.size(); i++) {
        desire_output[i] = output_desire[i];
    }
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

    for (size_t i = 0; i < in_layer->count; i++) {
        input_values.push_back(in_layer->neurons[i]->value);
    }

    for (size_t lo = 0; lo < out_layer->count; lo++) {
        float raw_sum = feed_neuron(out_layer->neurons[lo], input_values);
        out_layer->neurons[lo]->value = sigmoid(raw_sum);
    }
}


float Neural_Network::random_num(float min, float max) {
    float scale = rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

float Neural_Network::xavier_init(size_t n_in, size_t n_out) {
    float limit = sqrtf(6.0f / (float)(n_in + n_out));
    return random_num(-limit, limit);
}

// NOTE: the derivative form expects the ACTIVATION a, never the raw sum z.
float Neural_Network::sigmoid(float x, bool derivate) {
    if (derivate) return x * (1 - x);
    return 1.0f / (1.0f + expf(-x));
}


void Neural_Network::print_layer(Layer *layer) {
    printf("==========\n");
    printf("Neurons: %zu\n", layer->count);

    for (size_t i = 0; i < layer->count; i++) {
        Neuron *n = layer->neurons[i];

        printf("Neuron %zu:\n", i);
        printf("  value = %f\n", n->value);
        printf("  bias  = %f\n", n->bias);
        printf("  delta = %f\n", n->delta);
    }

    printf("\n");
}


std::vector<float> Neural_Network::mean_error(const std::vector<float> &desire_output) {
    std::vector<float> squared_differences = {};

    for (size_t neuron = 0; neuron < output_layer->count; neuron++) {
        float diff = desire_output[neuron] - output_layer->neurons[neuron]->value;
        squared_differences.push_back(diff * diff);
    }

    return squared_differences;
}


float Neural_Network::cost(const std::vector<float> &sqrd_diff_array) {
    double result = 0.0;   // accumulate wide: the gradient check subtracts two near-equal costs
    for (size_t i = 0; i < sqrd_diff_array.size(); i++) {
        result += sqrd_diff_array[i];
    }

    return (float)result;
}

void Neural_Network::feed_input_layer(const std::vector<float> &input) {
    if (input.size() != network[0]->count) {
        std::cout << "input size should match the input layer size" << "\n";
        return;
    }
    // feed some input values into the input layer
    for (size_t i = 0; i < network[0]->count; i++) {
        network[0]->neurons[i]->value = input[i];
    }
}

void Neural_Network::forward(const std::vector<float> &input) {
    feed_input_layer(input);

    for (size_t i = 1; i < network.size(); i++) {
        layer_processing(network[i - 1], network[i]);
    }
}

float Neural_Network::forward_cost(const std::vector<float> &input) {
    forward(input);
    return cost(mean_error(desire_output));
}

std::vector<float> Neural_Network::make_estimate(std::vector<float> &input) {
    std::vector<float> output = {}; 
    this->forward(input);

    for (int i = 0; i < output_layer->count; i++) {
        output.push_back(output_layer->neurons[i]->value);
    }
    
    return output;
}

// ---------------------------------------------------------------- backprop

// Step 1. Output layer: delta = dC/dz = 2(a - y) * a(1 - a)
void Neural_Network::set_deltas() {
    Layer *l = this->output_layer;

    for (size_t i = 0; i < l->count; i++) {
        float a = l->neurons[i]->value;
        float y = desire_output[i];
        l->neurons[i]->delta = 2.0f * (a - y) * sigmoid(a, true);
    }
}

// Step 2. Hidden layers, walking backwards. Layer 0 is input: no weights, no blame.
// For neuron j we gather the COLUMN of weights leaving it -- j stays fixed, k moves.
void Neural_Network::backprop_deltas() {
    for (size_t L = num_layers - 2; L >= 1; L--) {
        Layer *layer = network[L];
        Layer *next  = network[L + 1];

        for (size_t j = 0; j < layer->count; j++) {
            float sum = 0.0f;

            for (size_t k = 0; k < next->count; k++) {
                sum += next->neurons[k]->delta * next->neurons[k]->weights[j];
            }

            float a = layer->neurons[j]->value;
            layer->neurons[j]->delta = sum * sigmoid(a, true);
        }
    }
}

// Step 3. Every delta is final, so it is safe to move the weights now.
// gradient of weight i = this neuron's delta * the activation arriving on wire i
void Neural_Network::apply_updates(float learning_rate) {
    for (size_t L = 1; L < num_layers; L++) {
        Layer *layer = network[L];
        Layer *prev  = network[L - 1];

        for (size_t j = 0; j < layer->count; j++) {
            Neuron *n = layer->neurons[j];

            for (size_t i = 0; i < prev->count; i++) {
                n->weights[i] -= learning_rate * n->delta * prev->neurons[i]->value;
            }

            n->bias -= learning_rate * n->delta;   // bias has no incoming wire to scale it
        }
    }
}

// Returns the cost BEFORE this step's update, so a training loop can watch it fall.
float Neural_Network::train_step(const std::vector<float> &input, float learning_rate) {
    float c = forward_cost(input);

    set_deltas();
    backprop_deltas();
    apply_updates(learning_rate);

    return c;
}

// labels[i] is the target vector for training_data[i], so every sample carries
// its own answer instead of sharing one stale desire_output.
void Neural_Network::train_network(std::vector<std::vector<float>> &training_data,
                                   std::vector<std::vector<float>> &labels,
                                   float learning_rate, size_t epocs) {
    if (training_data.size() != labels.size()) {
        fprintf(stderr, "training data (%zu) and labels (%zu) must be the same length\n",
                training_data.size(), labels.size());
        exit(EXIT_FAILURE);
    }

    for (size_t steps = 0; steps < epocs; steps++) {
        for (size_t i = 0; i < training_data.size(); i++) {
            set_desired_output(labels[i]);
            train_step(training_data[i], learning_rate);
        }
    }
}


// Nudge one weight, measure the cost slope directly, compare against backprop.
// This is the only thing that proves the maths above is right.
void Neural_Network::gradient_check(const std::vector<float> &input, size_t L, size_t j, size_t i) {
    const float eps = 1e-3f;
    Neuron *n = network[L]->neurons[j];
    float saved = n->weights[i];

    n->weights[i] = saved + eps;
    float c_plus = forward_cost(input);

    n->weights[i] = saved - eps;
    float c_minus = forward_cost(input);

    n->weights[i] = saved;
    float numeric = (c_plus - c_minus) / (2.0f * eps);

    // clean pass on the unperturbed weights, then read the analytic gradient
    forward_cost(input);
    set_deltas();
    backprop_deltas();
    float analytic = n->delta * network[L - 1]->neurons[i]->value;

    printf("grad check L=%zu j=%zu i=%zu | numeric % .6f | analytic % .6f | diff %.2e\n",
           L, j, i, numeric, analytic, fabsf(numeric - analytic));
}


// Raw weight (~90) next to raw height (~1.6) drives the first hidden layer deep
// into the flat tails of the sigmoid, where a(1-a) is ~0 and no gradient survives.
// Centre and scale both features so the raw sums start near 0.
static const float H_MEAN = 1.70f, H_SCALE = 0.15f;
static const float W_MEAN = 70.0f, W_SCALE = 20.0f;

static std::vector<float> scale_input(float height, float weight) {
    return { (height - H_MEAN) / H_SCALE, (weight - W_MEAN) / W_SCALE };
}

int main() {
    // 1 - male
    // 0 - female
    // [0] - height
    // [1] - weight
    std::vector<std::vector<float>> in_total = {
        scale_input(1.62f, 58.0f),
        scale_input(1.78f, 82.5f),
        scale_input(1.55f, 50.0f),
        scale_input(1.85f, 90.0f)
    };

    // one target vector per sample -- these are labels, not a single output vector.
    // sigmoid outputs live in (0,1) -- targets must too, or the cost can never reach 0
    std::vector<std::vector<float>> desire_outputs = { {0.0f}, {1.0f}, {0.0f}, {1.0f} };

    Neural_Network nn(100, 4, 2, 1);

    printf("--- gradient check ---\n");
    nn.set_desired_output(desire_outputs[0]);      // the checks below cost against sample 0
    // nn.gradient_check(in_total[0], 3, 0, 7);    // output layer weight
    // nn.gradient_check(in_total[0], 2, 5, 12);   // hidden layer weight
    // nn.gradient_check(in_total[0], 1, 3, 4);    // first hidden layer weight

    printf("\n--- training ---\n");
    
    nn.train_network(in_total, desire_outputs, 0.4, 1000);

    printf("\n--- result ---\n");
    for (size_t i = 0; i < in_total.size(); i++) {
        nn.set_desired_output(desire_outputs[i]);
        float c = nn.forward_cost(in_total[i]);
        printf("sample %zu | target %.1f | estimate %f | cost %f\n",
               i, desire_outputs[i][0], nn.output_layer->neurons[0]->value, c);
    }


    while (true) {
        std::cout << "Lets test it! (height weight, e.g. \"1.62 58\") : \n";
        std::string usr_in;
        if (!std::getline(std::cin, usr_in)) break;   // EOF / ctrl-D

        // commas count as separators too
        for (char &c : usr_in) if (c == ',') c = ' ';

        std::vector<float> in;
        std::istringstream iss(usr_in);
        float v;
        while (iss >> v) in.push_back(v);

        if (in.size() != 2) {
            std::cout << "need exactly 2 numbers (height weight)\n";
            continue;
        }

        // the net was trained on scaled features, so scale what the user typed too
        std::vector<float> scaled = scale_input(in[0], in[1]);
        std::vector<float> out = nn.make_estimate(scaled);

        std::cout << "Estimate Result: " << out[0]
                  << (out[0] >= 0.5f ? "  (male)" : "  (female)") << "\n";
    }

    
    return 0;
}
