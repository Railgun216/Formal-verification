#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demonstration of robustness verification"""

import numpy as np
from robustness_verifier import ReLUNetwork, RobustnessVerifier

def main():
    print("=" * 70)
    print("Neural Network Robustness Verification Demo")
    print("=" * 70)
    
    # Create network
    print("\n[Step 1] Creating ReLU network...")
    np.random.seed(42)
    W1 = np.random.randn(32, 784) * 0.1
    b1 = np.zeros(32)
    W2 = np.random.randn(10, 32) * 0.1
    b2 = np.zeros(10)
    network = ReLUNetwork([W1, W2], [b1, b2])
    print(f"  ✓ Network: {network.num_layers} layers, input={network.weights[0].shape[1]}, output={network.weights[-1].shape[0]}")
    
    # Create input
    print("\n[Step 2] Creating test input...")
    x0 = np.random.rand(784)
    x0 = x0 / np.max(x0)
    cls = network.predict(x0)
    print(f"  ✓ Input shape: {x0.shape}, predicted class: {cls}")
    
    # Verify
    print("\n[Step 3] Running verification...")
    verifier = RobustnessVerifier(network, method="smt", use_cegar=False)
    
    for eps in [0.01, 0.03, 0.05]:
        res, cex = verifier.verify_robustness(x0, eps)
        status = "✓ SAFE" if res.value == "SAFE" else ("✗ UNSAFE" if res.value == "UNSAFE" else "? UNKNOWN")
        print(f"  ε={eps:.3f}: {status}")
        if cex is not None:
            print(f"    Counterexample: class {cls} → {network.predict(cex)}")
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()


