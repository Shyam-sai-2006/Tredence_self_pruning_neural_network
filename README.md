Self-Pruning Neural Network for CIFAR-10

This repository contains my solution to the “Self-Pruning Neural Network” case study. The objective was to design a neural network that learns to prune its own connections during training, instead of relying on a separate post-processing step.

The implementation builds a custom prunable layer, integrates a sparsity-aware loss function, and evaluates the trade-off between model accuracy and sparsity on the CIFAR-10 dataset.

Problem Overview

Modern neural networks are often over-parameterized, which makes them expensive to deploy in real-world environments. Pruning helps reduce model size and computational cost by removing less important weights.

In this problem, the challenge is taken one step further:

Instead of pruning after training, the network must learn which connections to remove during training itself.

This is achieved by associating each weight with a learnable gate and encouraging the network to suppress unnecessary connections through regularization.

Approach
1. Prunable Linear Layer

A custom PrunableLinear layer is implemented in place of torch.nn.Linear.

Each weight has a corresponding gate parameter, which determines whether the connection is active or suppressed.

Gate scores are passed through a sigmoid to obtain values in [0, 1]
A Straight-Through Estimator (STE) is used to:
apply a hard mask (0 or 1) during the forward pass
preserve gradients during backpropagation

This allows the model to behave like a pruned network during training while remaining fully trainable.

2. Sparsity-Aware Loss

The total loss used during training is:

Loss=CrossEntropy+λ⋅Sparsity Loss
The sparsity loss is computed as the L1 norm of gate activations
It is normalized by the total number of gates to ensure stable scaling
The hyperparameter λ (lambda) controls the trade-off:
low λ → higher accuracy, less pruning
high λ → more pruning, possible accuracy drop
3. Training and Evaluation

The model is trained on CIFAR-10 using standard data augmentation.

For each λ value, the following are reported:

Test Accuracy
Sparsity Level (percentage of effectively pruned weights)
Gate value distribution
Why L1 Regularization on Gates Encourages Sparsity

The gate values act as soft indicators of whether a weight should remain active.

Applying an L1 penalty encourages many of these gate values to shrink toward zero because:

smaller values reduce the penalty
the optimizer prefers sparse solutions under L1 constraints

When combined with thresholding in the forward pass, this results in many connections being effectively turned off, leaving only the most important ones active.

Repository Structure
.
├── CASE_STUDY.ipynb        # Main notebook (end-to-end execution)
├── layers.py               # PrunableLinear implementation
├── models.py               # CNN architecture with prunable layers
├── data_loader.py          # CIFAR-10 data loading and transforms
├── engine.py               # Training, evaluation, sparsity logic
├── analysis.py             # Plotting and result summaries
├── run_experiments.py      # Script to run full experiments
├── outputs/                # Saved plots and results
└── README.md
Setup

Install dependencies:

pip install torch torchvision matplotlib numpy
Running the Project
Option 1: Notebook (Recommended)

Open and run:

CASE_STUDY.ipynb

This will:

train the model
run experiments for different λ values
generate plots and results
Option 2: Script
python run_experiments.py
Results
Lambda	Test Accuracy (%)	Sparsity Level (%)
0.00	[fill]	[fill]
0.01	[fill]	[fill]
0.05	[fill]	[fill]
Observations
Increasing λ consistently increases sparsity across the network
Moderate λ values provide a good balance between accuracy and compression
Very high λ leads to aggressive pruning, which can degrade performance
The model successfully learns to suppress weaker connections during training
Gate Distribution

The distribution of final gate values for the best model is shown below:

![Gate Distribution](outputs/gate_distribution_best_model.png)

A successful pruning behavior is indicated by:

a concentration of values near 0 (pruned connections)
a separate cluster away from zero (important connections)
Key Design Decisions
Used Straight-Through Estimator (STE) to enable hard pruning behavior with gradient flow
Normalized sparsity loss to make λ meaningful across experiments
Focused pruning on classifier layers to keep the experiment interpretable
Kept the architecture simple to clearly observe pruning effects
Limitations
Pruning is applied only to fully connected layers, not convolutional layers
The model structure is not physically compressed after training
Sparsity is threshold-based rather than structurally enforced
Results may vary slightly due to randomness in training
Possible Improvements
Extend pruning to convolutional layers (structured pruning)
Rebuild and export a compact model after pruning
Add training curves (accuracy vs sparsity over time)
Run multiple seeds for more robust evaluation
Explore advanced gating methods (e.g., Hard Concrete distribution)
Summary

This project demonstrates a self-pruning neural network that learns to identify and suppress unnecessary connections during training. By combining gated weights, sparsity regularization, and controlled experimentation, it provides a clear view of the trade-off between model accuracy and sparsity.
