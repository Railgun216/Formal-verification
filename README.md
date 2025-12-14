# Verifying Local Robustness of ReLU Classifiers using SMT/MILP

This project implements local robustness verification for ReLU MLP classifiers using SMT (Satisfiability Modulo Theories) and MILP (Mixed Integer Linear Programming) methods. The verification is performed using the Z3 SMT solver.

## Features

- **SMT/MILP Verification**: Exact verification using SMT and MILP encodings
- **Z3 Solver Integration**: Uses Z3 SMT solver for constraint solving
- **ReLU Encoding**: Big-M method for encoding ReLU activations as mixed-integer constraints
- **Local Robustness Verification**: Verify that predictions remain invariant within ℓ∞-ball neighborhoods
- **CEGAR Refinement**: Optional counterexample-guided abstraction refinement
- **Multiple Model Formats**: Support for loading models from `.mat` (MATLAB) and `.onnx` (ONNX) files

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) For ONNX model support:
```bash
pip install onnx onnxruntime
```

3. Z3 solver is required for SMT/MILP verification:
```bash
pip install z3-solver
```

## Usage

### Basic Usage

```python
from robustness_verifier import ReLUNetwork, RobustnessVerifier
import numpy as np

# Load your network (or create a dummy one for testing)
network = ReLUNetwork.from_mat("model.mat")
# or
network = ReLUNetwork.from_onnx("model.onnx")

# Create verifier with SMT method
verifier = RobustnessVerifier(network, method="smt", use_cegar=False)

# Or use MILP method
# verifier = RobustnessVerifier(network, method="milp", use_cegar=False)

# Define seed input and epsilon
x0 = np.random.rand(784) / 255.0  # Normalized input
epsilon = 0.01  # ℓ∞-ball radius

# Verify robustness
result, counterexample = verifier.verify_robustness(x0, epsilon)

if result.value == "SAFE":
    print("✓ Network is robust at this input")
elif result.value == "UNSAFE":
    print("✗ Counterexample found!")
    print(f"  Original class: {network.predict(x0)}")
    print(f"  Adversarial class: {network.predict(counterexample)}")
else:
    print("? Verification inconclusive")
```

### Running the Example

```bash
# Run the main example
python robustness_verifier.py

# Run the comprehensive example
python example_usage.py

# Test model loading
python example_usage.py test
```

## Project Structure

```
.
├── robustness_verifier.py  # Main verification module
├── example_usage.py        # Example scripts and usage demonstrations
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Methodology

### Input Set

The verifier considers an ℓ∞-ball around a seed input:

```
X = {x ∈ R^d | ||x - x0||∞ ≤ ε}
```

where `x0` is the seed input and `ε` is the perturbation radius.

### Specification

Local robustness requires that the predicted class remains constant:

```
∀x ∈ X: arg max_k f_k(x) = c
```

where `c = arg max_k f_k(x0)` is the nominal class. This is equivalent to checking that all margins are non-negative:

```
fc(x) - fk(x) ≥ 0  for all k ≠ c
```

### Verification Process

1. **SMT/MILP Encoding**: Encode the ReLU network and robustness property as SMT/MILP constraints
   - Input constraints: ℓ∞-ball bounds
   - Linear layers: y = W*x + b
   - ReLU activations: Big-M encoding with binary variables
   - Output constraints: Check for counterexamples (f_k(x) > f_c(x))
2. **Constraint Solving**: Use Z3 SMT solver to check satisfiability
3. **Result Interpretation**:
   - **SAFE**: No counterexample found (network is robust)
   - **UNSAFE**: Counterexample found (network is not robust)
   - **UNKNOWN**: Solver timeout or error
4. **CEGAR Refinement** (optional): Split the input region and re-verify on sub-regions if needed

## Model Format

### MATLAB (.mat) Format

The `.mat` file should contain weight matrices and bias vectors:
- `W1`, `W2`, ... (weight matrices)
- `b1`, `b2`, ... (bias vectors)

### ONNX Format

Standard ONNX model files are supported. The network should be a feedforward MLP with ReLU activations.

## Configuration

The `RobustnessVerifier` class accepts several parameters:

- `method`: Verification method - "smt" or "milp" (default: "smt")
- `use_cegar`: Enable CEGAR-style refinement (default: False)
- `max_cegar_iterations`: Maximum number of CEGAR refinement iterations (default: 5)

### SMT vs MILP

- **SMT (Satisfiability Modulo Theories)**: Uses Z3's SMT solver with real arithmetic and boolean variables for ReLU encoding
- **MILP (Mixed Integer Linear Programming)**: Uses the same encoding but can leverage MILP-specific optimizations (currently uses Z3's SMT solver which handles MILP constraints)

## Technical Details

### ReLU Encoding

ReLU activations are encoded using the Big-M method:

- For each ReLU unit, introduce a binary variable δ ∈ {0, 1}
- If δ = 1 (active): y = x, x ≥ 0
- If δ = 0 (inactive): y = 0, x ≤ 0
- Constraints: y ≥ 0, y ≥ x, y ≤ x + M(1-δ), y ≤ M·δ

### SMT Constraints

The verification problem is encoded as:
- Input constraints: x ∈ [x0 - ε, x0 + ε]
- Network constraints: Linear layers and ReLU encodings
- Property constraints: ∃k ≠ c: f_k(x) > f_c(x) (checking for counterexamples)

## Limitations

- SMT/MILP verification can be slow for large networks
- Z3 solver may timeout on complex problems
- Big-M constant needs to be chosen appropriately
- For very large networks, consider using abstraction-based methods or specialized verifiers (Marabou, α-β-CROWN)

## References

- NNV 2.0: Neural Network Verification Tool
- Star Abstractions for Neural Network Verification
- CEGAR for Neural Network Verification

## License

This is an educational implementation for verification research.


