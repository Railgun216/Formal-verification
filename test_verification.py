"""
Quick test script for robustness verification
"""

from robustness_verifier import ReLUNetwork, RobustnessVerifier
import numpy as np

def main():
    print("=" * 60)
    print("Quick Robustness Verification Test")
    print("=" * 60)
    
    # Create a simple 2-layer network
    print("\n1. Creating test network...")
    W1 = np.random.randn(32, 784) * 0.1
    b1 = np.zeros(32)
    W2 = np.random.randn(10, 32) * 0.1
    b2 = np.zeros(10)
    
    network = ReLUNetwork([W1, W2], [b1, b2])
    print(f"   ✓ Network created: {network.num_layers} layers")
    print(f"   Input dim: {network.weights[0].shape[1]}, Output dim: {network.weights[-1].shape[0]}")
    
    # Create test input
    print("\n2. Creating test input...")
    x0 = np.random.rand(784)
    x0 = x0 / np.max(x0)  # Normalize
    nominal_class = network.predict(x0)
    print(f"   ✓ Input created, nominal class: {nominal_class}")
    
    # Test verification
    print("\n3. Running verification...")
    verifier = RobustnessVerifier(network, use_exact_check=True, use_cegar=True)
    
    epsilons = [0.01, 0.03, 0.05]
    for epsilon in epsilons:
        print(f"\n   Testing ε = {epsilon:.3f}...", end=" ")
        result, counterexample = verifier.verify_robustness(x0, epsilon)
        print(f"Result: {result.value}")
        
        if counterexample is not None:
            adv_class = network.predict(counterexample)
            print(f"      ⚠️  Counterexample: class {nominal_class} → {adv_class}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()


