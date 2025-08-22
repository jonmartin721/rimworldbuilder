#!/usr/bin/env python3
"""Test GPU functionality for RTX 5090"""

import torch
import os

# Set environment variables for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

print("="*60)
print("GPU Test for RTX 5090")
print("="*60)

# Basic info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Try basic tensor operations
    print("\nTesting basic operations...")
    try:
        # Create tensors
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        
        # Basic math
        z = x + y
        print("Addition: OK")
        
        # Matrix multiplication
        z = torch.matmul(x, y)
        print("Matrix multiplication: OK")
        
        # Neural network layer
        layer = torch.nn.Linear(100, 50).cuda()
        output = layer(x)
        print("Neural network layer: OK")
        
        # Backward pass
        loss = output.mean()
        loss.backward()
        print("Backward pass: OK")
        
        print("\nAll GPU operations successful!")
        
    except Exception as e:
        print(f"\nGPU operation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No GPU available")

print("="*60)