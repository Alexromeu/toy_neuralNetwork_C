# toy_neuralNetwork_C

A minimal feed‑forward neural network written from scratch in C++ — no external libraries, no linear algebra framework, just neurons, weights and loops.

This project is an educational exploration of how a network actually learns: forward propagation, the cost function, backpropagation of the error, and gradient descent on the weights and biases.

The goal is clarity and correctness, not performance or completeness.

> The original C version is kept as `neural_network.c` for reference. Active development happens in `neural_network.cpp`.

---

## Features

- Configurable architecture: neurons per hidden layer, number of layers, input size, output size
- Xavier weight initialization
- Sigmoid activation
- Forward propagation
- Mean squared error cost
- **Backpropagation** in three explicit steps — output deltas, hidden deltas, weight update
- **Numerical gradient checking** to verify the analytic gradient is correct
- Batch-free (per-sample) training loop over labelled data
- Interactive prompt to test the trained network

---

## Build

```sh
g++ -std=c++17 neural_network.cpp -o neural
./neural
```

The old C version:

```sh
gcc neural_network.c -o neural_network -lm
```

---

## The demo

`main()` trains a 2 → 100 → 100 → 1 network to guess *male / female* from **height (m)** and **weight (kg)**:

| height | weight | target |
| ------ | ------ | ------ |
| 1.62   | 58.0   | 0 (female) |
| 1.78   | 82.5   | 1 (male)   |
| 1.55   | 50.0   | 0 (female) |
| 1.85   | 90.0   | 1 (male)   |

Four samples is a toy dataset — it demonstrates that the gradient works, not that the classifier generalises.

After training it drops into a loop where you can type your own values:

```
Lets test it! (height weight, e.g. "1.62 58") :
1.80 85
Estimate Result: 0.973  (male)
```

### Input scaling

Raw weight (~90) next to raw height (~1.6) pushes the first hidden layer deep into the flat tails of the sigmoid, where `a(1-a) ≈ 0` and no gradient survives. Both features are therefore centred and scaled by `scale_input()` before entering the network — and user input is scaled the same way.

---

## How it works

```
Z = w · a⁽ˡ⁻¹⁾ + b        weighted sum of the previous layer
a = sigmoid(Z)            activation of the current layer
C = Σ (a − y)²            cost against the desired output y
```

The chain rule gives the slope of the cost with respect to a single weight:

```
dC     dZ   da   dC
--  =  -- · -- · --
dw     dw   dz   da
```

which expands to `a⁽ˡ⁻¹⁾ · sigmoid'(z⁽ˡ⁾) · 2(a⁽ˡ⁾ − y)`.

Backpropagation is split so the ordering is impossible to get wrong:

1. `set_deltas()` — blame for the output layer: `delta = 2(a − y) · a(1 − a)`
2. `backprop_deltas()` — push blame backwards; for neuron *j* gather the **column** of weights leaving it
3. `apply_updates()` — every delta is final, so it is now safe to move the weights

> **Note:** `sigmoid(x, true)` expects the *activation* `a`, never the raw sum `z`.

### Gradient check

`gradient_check()` nudges one weight by ±ε, measures the cost slope directly, and prints it next to the analytic gradient from backprop. It is the only thing that proves the maths above is right. The calls in `main()` are commented out — uncomment them to run it:

```cpp
nn.gradient_check(in_total[0], 3, 0, 7);    // output layer weight
nn.gradient_check(in_total[0], 2, 5, 12);   // hidden layer weight
nn.gradient_check(in_total[0], 1, 3, 4);    // first hidden layer weight
```

---

## API sketch

```cpp
Neural_Network nn(100, 4, 2, 1);   // neurons/layer, total layers, input size, output size

nn.train_network(training_data, labels, /*learning_rate=*/0.4f, /*epochs=*/1000);

std::vector<float> out = nn.make_estimate(input);
```

Targets must live in `(0, 1)` — the output neuron is a sigmoid, so a target of exactly 0 or 1 can never be reached and the cost never quite hits zero.

---

## Files

| file | purpose |
| ---- | ------- |
| `neural_network.cpp` | the network — current implementation |
| `neural_network.c`   | the original C version, forward pass only |
| `doc.txt`            | working notes and derivations |

---

## Roadmap

- Other activation functions (ReLU, tanh, softmax output)
- Mini-batch gradient descent
- Save / load trained weights
- A real dataset
