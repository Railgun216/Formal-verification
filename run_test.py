"""Run verification test and save output"""
import sys
from robustness_verifier import ReLUNetwork, RobustnessVerifier
import numpy as np

def test_verification():
    print("=" * 60)
    print("Robustness Verification Test")
    print("=" * 60)
    
    # Create a simple 2-layer network
    print("\n1. Creating test network...")
    np.random.seed(42)  # For reproducibility
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
    logits = network.forward(x0)
    print(f"   ✓ Input created")
    print(f"   Nominal class: {nominal_class}")
    print(f"   Logits: {logits}")
    
    # Test verification with simpler settings first
    print("\n3. Running verification (simplified, no CEGAR)...")
    verifier = RobustnessVerifier(network, method="smt", use_cegar=False)
    
    epsilons = [0.01, 0.03, 0.05]
    for epsilon in epsilons:
        print(f"\n   Testing ε = {epsilon:.3f}...", end=" ", flush=True)
        try:
            result, counterexample = verifier.verify_robustness(x0, epsilon)
            print(f"Result: {result.value}")
            
            if counterexample is not None:
                adv_class = network.predict(counterexample)
                print(f"      ⚠️  Counterexample found: class {nominal_class} → {adv_class}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n4. Testing with exact check enabled...")
    verifier2 = RobustnessVerifier(network, method="smt", use_cegar=False)
    epsilon = 0.01
    print(f"   Testing ε = {epsilon:.3f} with exact check...", end=" ", flush=True)
    try:
        result, counterexample = verifier2.verify_robustness(x0, epsilon)
        print(f"Result: {result.value}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_verification()


