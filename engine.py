import torch
from layers import PrunableLinear

def get_sparsity_loss(model):
    """
    Calculates the L1 penalty of the gates using the MEAN.
    Using mean() instead of sum() makes the Lambda hyperparameter 
    scale-invariant (it works the same even if we add more layers).
    """
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    total_elements = 0
    
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            l1_loss += gates.sum()
            total_elements += gates.numel()
            
    # Normalize the loss
    if total_elements > 0:
        return l1_loss / total_elements
    return l1_loss

def train_one_epoch(model, dataloader, criterion, optimizer, device, lambda_val):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Classification Loss
        cls_loss = criterion(outputs, targets)
        
        # Scale-Invariant Sparsity Loss
        sparsity_loss = get_sparsity_loss(model)
        
        # Total Loss
        loss = cls_loss + (lambda_val * sparsity_loss)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def calculate_sparsity(model, threshold=0.01):
    total_weights = 0
    pruned_weights = 0
    
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total_weights += gates.numel()
                pruned_weights += (gates < threshold).sum().item()
                
    if total_weights == 0:
        return 0.0
    return (pruned_weights / total_weights) * 100.0
