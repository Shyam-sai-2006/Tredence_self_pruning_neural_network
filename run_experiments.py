import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np

from models import CIFAR10PrunableNet
from data_loader import get_cifar10_loaders
from engine import train_one_epoch, evaluate, calculate_sparsity
from analysis import plot_gate_distributions, print_markdown_table

# 1. Ensure Reproducibility (Crucial for Engineering)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42) # Lock in the random factors
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    os.makedirs('outputs', exist_ok=True)

    batch_size = 128
    epochs = 20
    learning_rate = 0.001
    
    # Because we now use .mean() instead of .sum(), we must increase Lambda
    # 0.0: Baseline | 1.0: Moderate Sparsity | 2.5: High Sparsity
    lambda_values = [0.0, 10.0, 50.0, 100.0] 
    
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    results = []

    for lam in lambda_values:
        print(f"\n{'='*40}")
        print(f"Starting Experiment with Lambda: {lam}")
        print(f"{'='*40}")
        
        model = CIFAR10PrunableNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, trainloader, criterion, optimizer, device, lam
            )
            current_sparsity = calculate_sparsity(model)
            
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Sparsity: {current_sparsity:.2f}%")
        
        print("Evaluating on Test Set...")
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        final_sparsity = calculate_sparsity(model)
        
        print(f"--> Final Test Acc: {test_acc:.2f}% | Final Sparsity: {final_sparsity:.2f}%")
        
        # Save the best highly-pruned model for deploy.py
        if lam == 2.5:
            torch.save(model.state_dict(), "outputs/best_pruned_model.pth")
            print("Saved highly pruned model to outputs/best_pruned_model.pth")
            
        results.append({'lambda': lam, 'accuracy': test_acc, 'sparsity': final_sparsity})
        plot_path = f"outputs/gate_dist_lambda_{lam}.png"
        plot_gate_distributions(model, lam, save_path=plot_path)

    print_markdown_table(results)

if __name__ == "__main__":
    main()
