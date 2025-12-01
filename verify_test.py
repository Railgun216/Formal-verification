"""Comprehensive verification test with file output"""
import sys
import numpy as np
from robustness_verifier import ReLUNetwork, RobustnessVerifier, VerificationResult

# Redirect output to file
output_file = open('verification_results.txt', 'w', encoding='utf-8')

def print_to_file(*args, **kwargs):
    """Print to both console and file"""
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_file)

print_to_file("=" * 70)
print_to_file("Neural Network Robustness Verification Test")
print_to_file("=" * 70)

# Test 1: Create a simple network
print_to_file("\n[Test 1] Creating a 2-layer ReLU network...")
np.random.seed(42)
W1 = np.random.randn(32, 784) * 0.1
b1 = np.zeros(32)
W2 = np.random.randn(10, 32) * 0.1
b2 = np.zeros(10)

network = ReLUNetwork([W1, W2], [b1, b2])
print_to_file(f"✓ Network created successfully")
print_to_file(f"  - Number of layers: {network.num_layers}")
print_to_file(f"  - Input dimension: {network.weights[0].shape[1]}")
print_to_file(f"  - Output dimension: {network.weights[-1].shape[0]}")

# Test 2: Create test input
print_to_file("\n[Test 2] Creating test input...")
x0 = np.random.rand(784)
x0 = x0 / np.max(x0)  # Normalize to [0, 1]
nominal_class = network.predict(x0)
logits = network.forward(x0)
print_to_file(f"✓ Test input created")
print_to_file(f"  - Input shape: {x0.shape}")
print_to_file(f"  - Nominal class: {nominal_class}")
print_to_file(f"  - Top 3 logits: {np.argsort(logits)[-3:][::-1]}")

# Test 3: Star set creation
print_to_file("\n[Test 3] Testing Star set creation...")
verifier = RobustnessVerifier(network, use_exact_check=False, use_cegar=False)
input_star = verifier.create_input_star(x0, 0.01)
lb, ub = input_star.get_bounds()
print_to_file(f"✓ Star set created for ε=0.01")
print_to_file(f"  - Center shape: {input_star.center.shape}")
print_to_file(f"  - Basis shape: {input_star.basis.shape}")
print_to_file(f"  - Bounds: min={lb.min():.4f}, max={ub.max():.4f}")

# Test 4: Reachability analysis
print_to_file("\n[Test 4] Testing reachability analysis...")
output_star = verifier.reachability.reach(input_star)
output_lb, output_ub = output_star.get_bounds()
print_to_file(f"✓ Reachability analysis completed")
print_to_file(f"  - Output bounds shape: {output_lb.shape}")
print_to_file(f"  - Output range: [{output_lb.min():.4f}, {output_ub.max():.4f}]")

# Test 5: Robustness verification with different epsilons
print_to_file("\n[Test 5] Running robustness verification...")
epsilons = [0.01, 0.03, 0.05]

for epsilon in epsilons:
    print_to_file(f"\n  Testing ε = {epsilon:.3f}...")
    result, counterexample = verifier.verify_robustness(x0, epsilon)
    print_to_file(f"  Result: {result.value}")
    
    if counterexample is not None:
        adv_class = network.predict(counterexample)
        print_to_file(f"  ⚠️  Counterexample found!")
        print_to_file(f"     Original class: {nominal_class}")
        print_to_file(f"     Adversarial class: {adv_class}")

# Test 6: With exact checking
print_to_file("\n[Test 6] Testing with exact checking enabled...")
verifier_exact = RobustnessVerifier(network, use_exact_check=True, use_cegar=False)
result, counterexample = verifier_exact.verify_robustness(x0, 0.01)
print_to_file(f"✓ Exact check completed")
print_to_file(f"  Result: {result.value}")

# Test 7: With CEGAR
print_to_file("\n[Test 7] Testing with CEGAR refinement...")
verifier_cegar = RobustnessVerifier(network, use_exact_check=False, use_cegar=True, max_cegar_iterations=2)
result, counterexample = verifier_cegar.verify_robustness(x0, 0.01)
print_to_file(f"✓ CEGAR refinement completed")
print_to_file(f"  Result: {result.value}")

print_to_file("\n" + "=" * 70)
print_to_file("All tests completed successfully!")
print_to_file("=" * 70)

output_file.close()
print("\nResults saved to verification_results.txt")


