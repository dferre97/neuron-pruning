# Neuron Pruning Experiments

This repository contains experiments on neuron pruning. The goal is to compare different approaches for pruning neurons and weights in neural networks and evaluate their performance.

## Overview

The repository implements and compares three approaches:

1. **Neuron-level Popup Model**: Prunes entire neurons using a score-based selection method inspired by edge-popup.
2. **Weight-level Popup Model (Edge-Popup)**: Prunes individual weights using a score-based selection method.
3. **Standard MLP**: A baseline fully connected network trained without pruning.

All models typically share the same architecture (2 layers with the same number of neurons) and are evaluated on training and test accuracy.
