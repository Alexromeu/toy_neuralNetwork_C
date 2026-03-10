# toy_neuralNetwork_C

A minimal feed‑forward neural network implemented in C.  
This project is an educational exploration of how neural networks work internally—layer sizing, weight initialization, forward propagation, and memory layout—without external libraries.

The goal is clarity and correctness, not performance or completeness.

---

## Features

- Simple multi‑layer feed‑forward architecture
- Fixed‑size layers and neurons for predictable memory layout
- Weight initialization (random)
- Forward propagation with activation function
- Fully self‑contained single‑file implementation (`neural_network.c`)
- Designed for incremental expansion (training, backprop, dynamic sizing, etc.)

---

## Build

Compile with GCC:

```sh
gcc neural_network.c -o neural_network -lm

