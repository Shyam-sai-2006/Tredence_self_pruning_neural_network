# Self-Pruning Neural Networks: Differentiable Sparsity on CIFAR-10

This repository implements a custom neural network architecture capable of **autonomous weight pruning** during the training phase. By integrating learnable gates directly into the layer logic, the network identifies and removes redundant connections, optimizing the balance between computational complexity and classification accuracy.

## 1. Technical Methodology

### The Pruning Mechanism
The core of this implementation is the `PrunableLinear` module. Unlike standard linear layers, each weight $W$ is paired with a learnable gate score $s$. The effective weights used in the model are computed as:

$$W_{effective} = W \odot \text{Sigmoid}(s)$$

To ensure the model performs "hard" pruning (truly zeroing out weights) while remaining trainable, I implemented a **Straight-Through Estimator (STE)**. 

**Logic of the STE Implementation:**
* **Forward Pass:** The model applies a hard binary threshold ($< 0.01$) to the gates. This forces the hardware to bypass these weights, achieving true sparsity during inference.
* **Backward Pass:** The threshold is bypassed during backpropagation, and gradients flow through the "soft" sigmoid function. This allows the optimizer to update gate scores even after they have been "pruned," enabling the model to recover important connections if they become relevant during later training stages.

### Loss Formulation
The network optimizes a dual-objective loss function:
$$\text{Total Loss} = \text{CrossEntropy}(y, \hat{y}) + \lambda \cdot \text{Sparsity Loss}$$

The **Sparsity Loss** is calculated as the **mean** of the activated gate values. Normalizing by the mean ensures that the hyperparameter $\lambda$ remains scale-invariant, providing consistent pruning pressure regardless of the layer width or the total number of parameters.

## 2. Project Structure

The project is organized into modular Python files generated directly from the master notebook to ensure clean separation of concerns:

* **`layers.py`**: Implementation of the `PrunableLinear` class and the Straight-Through Estimator (STE) logic.
* **`models.py`**: Defines the `CIFAR10PrunableNet` architecture (CNN feature extractor + Prunable MLP head).
* **`engine.py`**: Core training logic, evaluation protocols, and the scale-invariant sparsity loss calculation.
* **`analysis.py`**: Visualization utilities for generating gate distribution histograms.
* **`run_experiments.py`**: Orchestration script that executes the hyperparameter sweep across $\lambda$ values and outputs the final performance metrics.
* **`CASE_STUDY.ipynb`**: The master environment for data management, environment setup, and experimental execution.

## 3. Experimental Results

The model was evaluated against the CIFAR-10 dataset across four levels of sparsity pressure ($\lambda$). 

| Lambda ($\lambda$) | Test Accuracy | Sparsity Level | Observations |
| :--- | :--- | :--- | :--- |
| 0.0 | 82.49% | 0.00% | Dense Baseline. |
| 10.0 | 81.04% | 52.85% | 50% reduction with minimal impact. |
| 50.0 | 80.68% | 74.51% | High compression efficiency. |
| **100.0** | **80.91%** | **89.60%** | **Optimal Frontier.** |

### **Visualizing Sparsity: The Bimodal Distribution**
At the optimal frontier ($\lambda=100$), the gate scores converge to a clear bimodal distribution (peaking at 0.0 and 1.0). This confirms that the STE successfully forced the network to make binary decisions—either a weight is vital (1.0) or it is redundant (0.0).

*(Note: Visualization plots are automatically generated in the `outputs/` directory after running the experiment suite.)*

## 4. Technical Challenges & Lessons Learned

* **Vanishing Gradients in Hard Pruning:** Initial iterations without the STE resulted in "soft" gates that never reached zero, failing to provide actual computational savings. The implementation of a Straight-Through Estimator was critical to achieving "hard" pruning while maintaining a differentiable path.
* **Late Crystallization Dynamics:** Empirical results show that at high $\lambda$, the model remains dense for the majority of training to establish feature representations, then prunes rapidly in the final epoch. Attempting to force sparsity too early in the training process led to accuracy collapse.
* **Numerical Stability:** Shifting from a sum-based to a mean-based sparsity loss allowed the $\lambda$ hyperparameter to generalize across different architecture scales without requiring manual retuning for every layer.

## 5. Usage

To replicate the experimental results and generate the sparsity-accuracy table:

1. Open `CASE_STUDY.ipynb` in Google Colab (ensure GPU acceleration is enabled).
2. Run all cells to generate the local `.py` modules and set up the directory structure.
3. Execute the experiment suite:
```bash
python run_experiments.py
