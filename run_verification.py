"""Run verification and save results"""
import numpy as np
from robustness_verifier import ReLUNetwork, RobustnessVerifier

# Create results
results = []

# Create network
np.random.seed(42)
W1 = np.random.randn(32, 784) * 0.1
b1 = np.zeros(32)
W2 = np.random.randn(10, 32) * 0.1
b2 = np.zeros(10)
network = ReLUNetwork([W1, W2], [b1, b2])
results.append(f"Network created: {network.num_layers} layers")

# Create input
x0 = np.random.rand(784)
x0 = x0 / np.max(x0)
nominal_class = network.predict(x0)
results.append(f"Input created, nominal class: {nominal_class}")

# Verify
verifier = RobustnessVerifier(network, method="smt", use_cegar=False)
epsilons = [0.01, 0.03, 0.05]

for eps in epsilons:
    result, counterexample = verifier.verify_robustness(x0, eps)
    results.append(f"ε={eps:.3f}: {result.value}")
    if counterexample is not None:
        adv_class = network.predict(counterexample)
        results.append(f"  Counterexample: {nominal_class} → {adv_class}")

# Save to file
with open('results.txt', 'w') as f:
    f.write('\n'.join(results))

print("Verification completed! Results saved to results.txt")


