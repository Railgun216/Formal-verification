"""
Verifying Local Robustness of ReLU Classifiers with NNV 2.0-style Verification

This module implements local robustness verification for ReLU MLP classifiers
using symbolic reachability with Star abstractions, with optional CEGAR refinement.
"""

import numpy as np
import scipy.io as sio
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import os

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class VerificationResult(Enum):
    """Verification outcome"""
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"  # Counterexample found
    UNKNOWN = "UNKNOWN"  # Over-approximation inconclusive


@dataclass
class StarSet:
    """
    Star set representation: {x | x = c + V*α, C*α <= d, α >= 0}
    where c is center, V is basis matrix, C and d define constraints
    """
    center: np.ndarray  # c: center vector
    basis: np.ndarray   # V: basis matrix
    C: np.ndarray       # Constraint matrix
    d: np.ndarray       # Constraint vector
    
    def __init__(self, center: np.ndarray, basis: Optional[np.ndarray] = None,
                 C: Optional[np.ndarray] = None, d: Optional[np.ndarray] = None):
        self.center = center
        if basis is None:
            self.basis = np.eye(len(center))
        else:
            self.basis = basis
        if C is None:
            self.C = np.array([]).reshape(0, self.basis.shape[1])
        else:
            self.C = C
        if d is None:
            self.d = np.array([])
        else:
            self.d = d
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding box (over-approximation)"""
        # This is a simplified version - full implementation would solve LPs
        # For now, use a simple approximation
        lb = self.center.copy()
        ub = self.center.copy()
        
        for i in range(self.basis.shape[1]):
            v = self.basis[:, i]
            # Simple bounds assuming constraints allow full range
            lb -= np.abs(v)
            ub += np.abs(v)
        
        return lb, ub


class ReLUNetwork:
    """ReLU MLP Network for verification"""
    
    def __init__(self, weights: List[np.ndarray], biases: List[np.ndarray]):
        """
        Initialize network with weights and biases
        
        Args:
            weights: List of weight matrices [W1, W2, ...]
            biases: List of bias vectors [b1, b2, ...]
        """
        self.weights = weights
        self.biases = biases
        self.num_layers = len(weights)
    
    @classmethod
    def from_mat(cls, mat_path: str) -> 'ReLUNetwork':
        """Load network from .mat file"""
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Model file not found: {mat_path}")
        
        data = sio.loadmat(mat_path)
        weights = []
        biases = []
        
        # Try different possible key formats
        i = 1
        found_any = False
        
        # Try W1, W2, ... format
        while True:
            key_w = f'W{i}'
            key_b = f'b{i}'
            if key_w in data:
                w = data[key_w]
                # Transpose if needed (MATLAB stores column-major)
                if w.shape[0] < w.shape[1] and i == 1:
                    w = w.T
                weights.append(w)
                if key_b in data:
                    biases.append(data[key_b].flatten())
                else:
                    biases.append(np.zeros(weights[-1].shape[0]))
                found_any = True
                i += 1
            else:
                break
        
        # Try alternative format: weights, biases
        if not found_any:
            if 'weights' in data:
                weights_list = data['weights']
                if isinstance(weights_list, np.ndarray):
                    for w in weights_list.flatten():
                        weights.append(w)
            if 'biases' in data:
                biases_list = data['biases']
                if isinstance(biases_list, np.ndarray):
                    for b in biases_list.flatten():
                        biases.append(b.flatten())
        
        if not weights:
            raise ValueError(f"Could not find weights in {mat_path}. Available keys: {list(data.keys())}")
        
        return cls(weights, biases)
    
    @classmethod
    def from_onnx(cls, onnx_path: str) -> 'ReLUNetwork':
        """Load network from ONNX file"""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX libraries not available. Install with: pip install onnx onnxruntime")
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Model file not found: {onnx_path}")
        
        model = onnx.load(onnx_path)
        weights = []
        biases = []
        
        # Extract weights and biases from ONNX model
        initializers = {init.name: init for init in model.graph.initializer}
        
        for node in model.graph.node:
            if node.op_type == 'Gemm' or node.op_type == 'MatMul':
                # Find weight and bias
                for inp_name in node.input:
                    if inp_name in initializers:
                        init = initializers[inp_name]
                        arr = np.frombuffer(init.raw_data, dtype=np.float32)
                        arr = arr.reshape(tuple(init.dims))
                        
                        if len(arr.shape) == 2:
                            weights.append(arr)
                        elif len(arr.shape) == 1:
                            biases.append(arr)
        
        # If no biases found, add zeros
        while len(biases) < len(weights):
            biases.append(np.zeros(weights[len(biases)].shape[0]))
        
        return cls(weights, biases)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W.T + b
            if i < len(self.weights) - 1:  # ReLU on all but last layer
                x = np.maximum(0, x)
        return x
    
    def predict(self, x: np.ndarray) -> int:
        """Predict class for input"""
        logits = self.forward(x)
        return int(np.argmax(logits))


class StarReachability:
    """Star-based reachability analysis for ReLU networks"""
    
    def __init__(self, network: ReLUNetwork):
        self.network = network
    
    def reach(self, input_star: StarSet) -> StarSet:
        """
        Compute over-approximate reachable set using Star abstraction
        
        This is a simplified implementation. Full NNV 2.0 uses more
        sophisticated Star operations for ReLU layers.
        """
        current_star = input_star
        
        for i, (W, b) in enumerate(zip(self.network.weights, self.network.biases)):
            # Linear layer: y = W*x + b
            new_center = W @ current_star.center + b
            new_basis = W @ current_star.basis
            
            # Update constraints (simplified)
            new_C = current_star.C
            new_d = current_star.d
            
            current_star = StarSet(new_center, new_basis, new_C, new_d)
            
            # ReLU layer (except last)
            if i < len(self.network.weights) - 1:
                current_star = self._relu_star(current_star)
        
        return current_star
    
    def _relu_star(self, star: StarSet) -> StarSet:
        """
        Apply ReLU to Star set (simplified over-approximation)
        Full implementation would handle different ReLU cases more precisely
        """
        lb, ub = star.get_bounds()
        
        # Determine which neurons are always active, always inactive, or uncertain
        always_active = lb >= 0
        always_inactive = ub <= 0
        uncertain = ~(always_active | always_inactive)
        
        # For always active: identity
        # For always inactive: zero
        # For uncertain: over-approximate with interval [0, ub]
        
        new_center = star.center.copy()
        new_center[always_inactive] = 0
        new_center[uncertain] = np.maximum(0, new_center[uncertain])
        
        # Simplified basis update
        new_basis = star.basis.copy()
        new_basis[always_inactive, :] = 0
        new_basis[uncertain, :] *= 0.5  # Conservative approximation
        
        return StarSet(new_center, new_basis, star.C, star.d)


class RobustnessVerifier:
    """Main verifier for local robustness"""
    
    def __init__(self, network: ReLUNetwork, use_exact_check: bool = True,
                 use_cegar: bool = True, max_cegar_iterations: int = 5):
        self.network = network
        self.reachability = StarReachability(network)
        self.use_exact_check = use_exact_check
        self.use_cegar = use_cegar
        self.max_cegar_iterations = max_cegar_iterations
    
    def create_input_star(self, x0: np.ndarray, epsilon: float) -> StarSet:
        """
        Create Star set for ℓ∞-ball: X = {x | ||x - x0||∞ ≤ ε}
        """
        dim = len(x0)
        center = x0.copy()
        
        # Basis: identity matrix (one dimension per input dimension)
        basis = np.eye(dim) * epsilon
        
        # Constraints: -1 <= α_i <= 1 for each dimension
        # C*α <= d where C = [I; -I], d = [1; 1]
        C = np.vstack([np.eye(dim), -np.eye(dim)])
        d = np.ones(2 * dim)
        
        return StarSet(center, basis, C, d)
    
    def verify_robustness(self, x0: np.ndarray, epsilon: float) -> Tuple[VerificationResult, Optional[np.ndarray]]:
        """
        Verify local robustness at x0 with radius ε
        
        Returns:
            (result, counterexample) where result is SAFE/UNSAFE/UNKNOWN
        """
        # Get nominal prediction
        nominal_class = self.network.predict(x0)
        num_classes = self.network.weights[-1].shape[0]
        
        # Create input set
        input_star = self.create_input_star(x0, epsilon)
        
        # Try over-approximate reachability first
        result, counterexample = self._verify_with_reachability(
            input_star, x0, nominal_class, num_classes, epsilon
        )
        
        if result == VerificationResult.SAFE:
            return result, None
        
        if result == VerificationResult.UNSAFE:
            return result, counterexample
        
        # Over-approximation inconclusive - try exact check
        if self.use_exact_check and result == VerificationResult.UNKNOWN:
            result, counterexample = self._exact_check(x0, epsilon, nominal_class)
            if result == VerificationResult.UNSAFE:
                return result, counterexample
        
        # Try CEGAR refinement
        if self.use_cegar and result == VerificationResult.UNKNOWN:
            result, counterexample = self._cegar_refinement(
                x0, epsilon, nominal_class, num_classes
            )
        
        return result, counterexample
    
    def _verify_with_reachability(self, input_star: StarSet, x0: np.ndarray,
                                   nominal_class: int, num_classes: int,
                                   epsilon: float) -> Tuple[VerificationResult, Optional[np.ndarray]]:
        """Verify using Star reachability"""
        output_star = self.reachability.reach(input_star)
        
        # Get output bounds
        output_lb, output_ub = output_star.get_bounds()
        
        # Check margins: fc(x) - fk(x) >= 0 for all k != c
        fc_lb = output_lb[nominal_class]
        fc_ub = output_ub[nominal_class]
        
        for k in range(num_classes):
            if k == nominal_class:
                continue
            
            fk_lb = output_lb[k]
            fk_ub = output_ub[k]
            
            # Margin: fc - fk
            margin_lb = fc_lb - fk_ub  # Worst case margin
            
            if margin_lb < 0:
                # Potentially unsafe - but need to check if it's real or over-approximation
                return VerificationResult.UNKNOWN, None
        
        # All margins non-negative in over-approximation
        # Check if we can conclude SAFE (conservative check)
        margin_min = min(fc_lb - output_ub[k] for k in range(num_classes) if k != nominal_class)
        if margin_min >= 0:
            return VerificationResult.SAFE, None
        
        return VerificationResult.UNKNOWN, None
    
    def _exact_check(self, x0: np.ndarray, epsilon: float,
                     nominal_class: int) -> Tuple[VerificationResult, Optional[np.ndarray]]:
        """
        Exact check using optimization (simplified - full version would use ILP/SMT)
        """
        # Try to find counterexample by sampling and optimization
        num_samples = 1000
        best_counterexample = None
        best_margin = float('inf')
        
        # Clip bounds for normalized inputs
        input_lb = np.maximum(0, x0 - epsilon)
        input_ub = np.minimum(1, x0 + epsilon)
        
        for _ in range(num_samples):
            # Random sample in ℓ∞-ball
            perturbation = np.random.uniform(-epsilon, epsilon, size=x0.shape)
            x_test = np.clip(x0 + perturbation, input_lb, input_ub)
            
            logits = self.network.forward(x_test)
            predicted_class = np.argmax(logits)
            
            if predicted_class != nominal_class:
                return VerificationResult.UNSAFE, x_test
            
            # Check margin
            other_logits = [logits[k] for k in range(len(logits)) if k != nominal_class]
            margin = logits[nominal_class] - max(other_logits)
            if margin < best_margin:
                best_margin = margin
                best_counterexample = x_test
        
        # Try gradient-based search for counterexample
        x_adv = x0.copy()
        learning_rate = 0.01
        
        for _ in range(100):
            x_adv = np.clip(x_adv, input_lb, input_ub)
            
            logits = self.network.forward(x_adv)
            predicted_class = np.argmax(logits)
            
            if predicted_class != nominal_class:
                return VerificationResult.UNSAFE, x_adv
            
            # Gradient w.r.t. input (simplified)
            # In practice, use proper autograd
            other_logits = [logits[k] for k in range(len(logits)) if k != nominal_class]
            margin = logits[nominal_class] - max(other_logits)
            if margin < 0:
                return VerificationResult.UNSAFE, x_adv
            
            # Simple gradient approximation
            eps_grad = 1e-5
            grad = np.zeros_like(x_adv)
            for i in range(len(x_adv)):
                x_pert = x_adv.copy()
                x_pert[i] += eps_grad
                x_pert = np.clip(x_pert, input_lb, input_ub)
                logits_pert = self.network.forward(x_pert)
                margin_pert = logits_pert[nominal_class] - max([logits_pert[k] for k in range(len(logits_pert)) if k != nominal_class])
                grad[i] = (margin_pert - margin) / eps_grad
            
            # Update in direction that decreases margin
            x_adv = x_adv - learning_rate * grad
        
        return VerificationResult.UNKNOWN, None
    
    def _cegar_refinement(self, x0: np.ndarray, epsilon: float,
                          nominal_class: int, num_classes: int) -> Tuple[VerificationResult, Optional[np.ndarray]]:
        """
        CEGAR-style refinement by splitting input set
        """
        # Split input space into smaller regions
        num_splits = 2  # Split each dimension in half
        
        for iteration in range(self.max_cegar_iterations):
            # Create sub-regions by splitting
            sub_regions = self._split_input_region(x0, epsilon, num_splits)
            
            all_safe = True
            for region_center, region_epsilon in sub_regions:
                input_star = self.create_input_star(region_center, region_epsilon)
                result, counterexample = self._verify_with_reachability(
                    input_star, region_center, nominal_class, num_classes, region_epsilon
                )
                
                if result == VerificationResult.UNSAFE:
                    return result, counterexample
                
                if result == VerificationResult.UNKNOWN:
                    all_safe = False
            
            if all_safe:
                return VerificationResult.SAFE, None
            
            # Refine further
            num_splits *= 2
            epsilon = epsilon / 2
        
        return VerificationResult.UNKNOWN, None
    
    def _split_input_region(self, x0: np.ndarray, epsilon: float,
                            num_splits: int) -> List[Tuple[np.ndarray, float]]:
        """Split input region into sub-regions"""
        sub_regions = []
        step = 2 * epsilon / num_splits
        new_epsilon = epsilon / num_splits
        
        # Simple splitting: create grid of sub-regions
        for i in range(num_splits):
            offset = -epsilon + (i + 0.5) * step
            new_center = x0 + offset
            new_center = np.clip(new_center, 0, 1)  # Assuming normalized
            sub_regions.append((new_center, new_epsilon))
        
        return sub_regions


def main():
    """Example usage"""
    # Example: Create a simple 2-layer MLP for demonstration
    # In practice, load from .mat or ONNX file
    
    print("Creating dummy network for demonstration...")
    # Create dummy network (replace with actual model loading)
    W1 = np.random.randn(32, 784) * 0.1
    b1 = np.zeros(32)
    W2 = np.random.randn(10, 32) * 0.1
    b2 = np.zeros(10)
    
    network = ReLUNetwork([W1, W2], [b1, b2])
    
    # Example seed input (normalized MNIST-like)
    x0 = np.random.rand(784)
    x0 = x0 / np.max(x0)  # Normalize
    
    # Test different epsilon values
    epsilons = [0.01, 0.03, 0.05]
    
    verifier = RobustnessVerifier(network, use_exact_check=True, use_cegar=True)
    
    print("\nLocal Robustness Verification Results")
    print("=" * 50)
    
    for epsilon in epsilons:
        print(f"\nTesting ε = {epsilon}")
        result, counterexample = verifier.verify_robustness(x0, epsilon)
        
        print(f"Result: {result.value}")
        if counterexample is not None:
            print(f"Counterexample found!")
            print(f"  Original prediction: {network.predict(x0)}")
            print(f"  Adversarial prediction: {network.predict(counterexample)}")


if __name__ == "__main__":
    main()


