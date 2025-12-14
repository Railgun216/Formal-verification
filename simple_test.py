import numpy as np
from robustness_verifier import ReLUNetwork, RobustnessVerifier

# Simple test
print("Starting test...")
W1 = np.random.randn(10, 5) * 0.1
b1 = np.zeros(10)
W2 = np.random.randn(3, 10) * 0.1
b2 = np.zeros(3)

net = ReLUNetwork([W1, W2], [b1, b2])
print(f"Network created: {net.num_layers} layers")

x0 = np.random.rand(5)
print(f"Input created: shape {x0.shape}")

verifier = RobustnessVerifier(net, method="smt", use_cegar=False)
print("Verifier created")

result, counterexample = verifier.verify_robustness(x0, 0.01)
print(f"Verification result: {result.value}")
print("Test completed successfully!")


