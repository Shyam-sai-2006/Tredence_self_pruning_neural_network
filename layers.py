import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable gates (Start neutral so the optimizer discovers what to prune)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming initialization for weights (Industry standard for ReLu networks)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize gates at 0.0 (Sigmoid(0) = 0.5)
        nn.init.constant_(self.gate_scores, 0.0)

    def forward(self, x):
        # 1. Soft gates (differentiable)
        soft_gates = torch.sigmoid(self.gate_scores)
        
        # 2. Hard threshold (Physical Zeros for the forward pass)
        hard_mask = (soft_gates >= 0.01).float()
        
        # 3. Straight-Through Estimator (The "Senior" Math Trick)
        # Forward pass uses `hard_mask`. 
        # Backward pass gradients flow through `soft_gates`.
        ste_gates = hard_mask.detach() - soft_gates.detach() + soft_gates
        
        # Apply the absolute mask to the weights
        pruned_weights = self.weight * ste_gates
        
        return F.linear(x, pruned_weights, self.bias)
