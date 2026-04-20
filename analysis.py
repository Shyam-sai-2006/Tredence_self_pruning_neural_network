import torch
import matplotlib.pyplot as plt
import numpy as np
from layers import PrunableLinear

def plot_gate_distributions(model, lambda_val, save_path=None):
    """
    Extracts all gate scores from PrunableLinear layers, applies the sigmoid,
    and plots a histogram to visualize the sparsity distribution.
    """
    all_gates = []
    
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                all_gates.extend(gates)
                
    if not all_gates:
        print("No PrunableLinear layers found.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(all_gates, bins=50, color='teal', alpha=0.7, edgecolor='black')
    plt.title(f'Gate Value Distribution (Lambda = {lambda_val})')
    plt.xlabel('Gate Value (Sigmoid Output)')
    plt.ylabel('Frequency (Number of Weights)')
    plt.grid(axis='y', alpha=0.75)
    
    # Add a vertical line to show our 0.01 pruning threshold
    plt.axvline(x=0.01, color='red', linestyle='dashed', linewidth=2, label='Pruning Threshold (0.01)')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def print_markdown_table(results):
    """
    Generates the final comparative table required by the case study.
    """
    print("\n### Final Experiment Results")
    print("| Lambda | Test Accuracy (%) | Sparsity Level (%) |")
    print("| :--- | :--- | :--- |")
    for res in results:
        print(f"| {res['lambda']} | {res['accuracy']:.2f}% | {res['sparsity']:.2f}% |")
