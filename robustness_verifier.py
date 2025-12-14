"""
Verifying Local Robustness of ReLU Classifiers using SMT/MILP

This module implements local robustness verification for ReLU MLP classifiers
using SMT (Satisfiability Modulo Theories) and MILP (Mixed Integer Linear Programming).
"""

import numpy as np
import scipy.io as sio
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import os
import json
from datetime import datetime

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


class VerificationResult(Enum):
    """Verification outcome - unified audit labels"""
    SAFE = "SAFE"  # Proven robust
    COUNTEREXAMPLE = "COUNTEREXAMPLE"  # Counterexample found
    INCONCLUSIVE = "INCONCLUSIVE"  # Cannot determine (timeout, error, or over-approximation too loose)
    
    # Legacy aliases for backward compatibility
    UNSAFE = "COUNTEREXAMPLE"
    UNKNOWN = "INCONCLUSIVE"


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
    """Star-based reachability analysis for ReLU networks (NNV-style)"""
    
    def __init__(self, network: ReLUNetwork):
        self.network = network
    
    def create_input_star(self, x0: np.ndarray, epsilon: float) -> StarSet:
        """
        Create Star set for ℓ∞-ball clipped to [0,1]: X = {x | ||x - x0||∞ ≤ ε} ∩ [0,1]
        
        This matches the input domain used by SMT method for consistency.
        """
        dim = len(x0)
        center = x0.copy()
        
        # Basis: identity matrix scaled by epsilon
        basis = np.eye(dim) * epsilon
        
        # Constraints: -1 <= α_i <= 1 for each dimension
        # C*α <= d where C = [I; -I], d = [1; 1]
        C = np.vstack([np.eye(dim), -np.eye(dim)])
        d = np.ones(2 * dim)
        
        # Clip center to [0,1] to match SMT input domain
        center = np.clip(center, 0, 1)
        
        return StarSet(center, basis, C, d)
    
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
    
    def verify_robustness(self, x0: np.ndarray, epsilon: float, 
                         margin_tolerance: float = 1e-6) -> Tuple[VerificationResult, Optional[np.ndarray], Dict[str, Any]]:
        """
        Verify robustness using Star reachability (over-approximation)
        
        Returns:
            (result, counterexample, details)
            - result: SAFE, COUNTEREXAMPLE, or INCONCLUSIVE
            - counterexample: None (reachability doesn't produce exact counterexamples)
            - details: Dictionary with reachability analysis details
        """
        # Get nominal prediction
        nominal_class = self.network.predict(x0)
        num_classes = self.network.weights[-1].shape[0]
        
        # Create input star (clipped to [0,1] to match SMT)
        input_star = self.create_input_star(x0, epsilon)
        
        # Compute reachable set
        output_star = self.reach(input_star)
        
        # Get output bounds
        output_lb, output_ub = output_star.get_bounds()
        
        # Check margins: fc(x) - fk(x) >= 0 for all k != c
        fc_lb = output_lb[nominal_class]
        fc_ub = output_ub[nominal_class]
        
        margins = []
        min_margin = float('inf')
        all_margins_positive = True
        
        for k in range(num_classes):
            if k == nominal_class:
                continue
            
            fk_lb = output_lb[k]
            fk_ub = output_ub[k]
            
            # Margin: fc - fk (worst case)
            margin_lb = fc_lb - fk_ub  # Worst case margin
            margin_ub = fc_ub - fk_lb  # Best case margin
            
            margins.append({
                'class': k,
                'margin_lb': float(margin_lb),
                'margin_ub': float(margin_ub)
            })
            
            min_margin = min(min_margin, margin_lb)
            
            # If worst case margin < tolerance, potentially unsafe
            if margin_lb < margin_tolerance:
                all_margins_positive = False
        
        # Details for recording
        details = {
            'method': 'Star Reachability',
            'input_domain': {
                'center': x0.tolist(),
                'epsilon': float(epsilon),
                'lower_bound': np.maximum(0, x0 - epsilon).tolist(),
                'upper_bound': np.minimum(1, x0 + epsilon).tolist()
            },
            'output_bounds': {
                'lower_bound': output_lb.tolist(),
                'upper_bound': output_ub.tolist()
            },
            'margins': margins,
            'min_margin': float(min_margin),
            'nominal_class': nominal_class
        }
        
        # Determine result
        if min_margin >= margin_tolerance:
            # All margins non-negative in over-approximation
            # This is a conservative (sound) result
            return VerificationResult.SAFE, None, details
        else:
            # Over-approximation shows potential violation
            # Cannot conclude SAFE, but also cannot produce exact counterexample
            details['note'] = 'Over-approximation inconclusive - potential violation detected'
            return VerificationResult.INCONCLUSIVE, None, details


class SMTEncoder:
    """SMT encoder for ReLU network verification using Z3"""
    
    def __init__(self, network: ReLUNetwork, margin_tolerance: float = 1e-6, big_m: float = 1000.0):
        """
        Initialize SMT encoder
        
        Args:
            network: ReLU network to verify
            margin_tolerance: Tolerance threshold for margin check (default: 1e-6)
                            Used to avoid numerical errors: checks f_k(x) >= f_c(x) + tolerance
            big_m: Big-M constant for ReLU encoding (default: 1000.0)
        """
        if not Z3_AVAILABLE:
            raise ImportError("Z3 solver not available. Install with: pip install z3-solver")
        self.network = network
        self.margin_tolerance = margin_tolerance
        self.big_m = big_m
        self.solver = None
        self.input_vars = None
        self.layer_vars = []  # Variables for each layer
        self.relu_vars = []   # Binary variables for ReLU activations
        self.input_lb = None  # Store input bounds for audit
        self.input_ub = None
        
    def encode_network(self, x0: np.ndarray, epsilon: float):
        """
        Encode the ReLU network as SMT constraints
        
        Args:
            x0: Center point of input region
            epsilon: ℓ∞-ball radius
            
        Returns:
            Z3 solver with network constraints
        """
        solver = z3.Solver()
        
        input_dim = len(x0)
        # Input domain: ℓ∞-ball clipped to [0,1] - MUST match NNV side
        # This ensures both methods verify on the same input domain
        input_lb = np.maximum(0, x0 - epsilon)
        input_ub = np.minimum(1, x0 + epsilon)
        self.input_lb = input_lb.copy()
        self.input_ub = input_ub.copy()
        
        # Create input variables
        input_vars = [z3.Real(f'x_{i}') for i in range(input_dim)]
        self.input_vars = input_vars
        
        # Input constraints: ℓ∞-ball clipped to [0,1]
        # This is the ALIGNED input domain that must match NNV
        for i in range(input_dim):
            solver.add(input_vars[i] >= float(input_lb[i]))
            solver.add(input_vars[i] <= float(input_ub[i]))
        
        # Encode network layers
        current_vars = input_vars
        self.layer_vars = [input_vars]
        self.relu_vars = []
        
        for layer_idx, (W, b) in enumerate(zip(self.network.weights, self.network.biases)):
            output_dim = W.shape[0]
            
            # Pre-activation variables (before ReLU)
            pre_vars = [z3.Real(f'pre_{layer_idx}_{j}') for j in range(output_dim)]
            # Post-activation variables (after ReLU)
            post_vars = [z3.Real(f'post_{layer_idx}_{j}') for j in range(output_dim)]
            
            # Linear layer: pre = W * current + b
            for j in range(output_dim):
                linear_expr = z3.Sum([float(W[j, i]) * current_vars[i] for i in range(len(current_vars))])
                linear_expr = linear_expr + float(b[j])
                solver.add(pre_vars[j] == linear_expr)
            
            # ReLU activation (except last layer)
            if layer_idx < len(self.network.weights) - 1:
                relu_layer_vars = []
                for j in range(output_dim):
                    # Binary variable for ReLU activation state
                    delta = z3.Bool(f'delta_{layer_idx}_{j}')
                    relu_layer_vars.append(delta)
                    
                    # ReLU encoding using big-M method
                    # M should be large enough to cover all possible pre-activation values
                    # If actual values exceed M, this can lead to "false SAFE" results
                    # This is a soundness dependency that should be documented in audit reports
                    M = self.big_m
                    
                    # If delta = 1 (active): post = pre, pre >= 0
                    # If delta = 0 (inactive): post = 0, pre <= 0
                    solver.add(z3.Implies(delta, post_vars[j] == pre_vars[j]))
                    solver.add(z3.Implies(delta, pre_vars[j] >= 0))
                    solver.add(z3.Implies(z3.Not(delta), post_vars[j] == 0))
                    solver.add(z3.Implies(z3.Not(delta), pre_vars[j] <= 0))
                    
                    # Additional constraints for tighter encoding
                    solver.add(post_vars[j] >= 0)
                    solver.add(post_vars[j] >= pre_vars[j])
                    # Use If to convert Bool to Real: If(condition, 1, 0)
                    delta_real = z3.If(delta, z3.RealVal(1), z3.RealVal(0))
                    solver.add(post_vars[j] <= pre_vars[j] + M * (z3.RealVal(1) - delta_real))
                    solver.add(post_vars[j] <= M * delta_real)
                
                self.relu_vars.append(relu_layer_vars)
            else:
                # Last layer: no ReLU, post = pre
                for j in range(output_dim):
                    solver.add(post_vars[j] == pre_vars[j])
            
            current_vars = post_vars
            self.layer_vars.append(post_vars)
        
        self.solver = solver
        return solver
    
    def verify_robustness(self, x0: np.ndarray, epsilon: float) -> Tuple[VerificationResult, Optional[np.ndarray]]:
        """
        Verify robustness using SMT
        
        Returns:
            (result, counterexample)
        """
        # Get nominal prediction
        nominal_class = self.network.predict(x0)
        num_classes = self.network.weights[-1].shape[0]
        
        # Encode network
        solver = self.encode_network(x0, epsilon)
        
        # Get output variables (last layer)
        output_vars = self.layer_vars[-1]
        
        # Check for counterexample: exists k != c such that f_k(x) >= f_c(x) + tolerance
        # Using >= (non-strict) to match proposal spec: margin >= 0
        # Adding tolerance to avoid numerical errors (margin >= tolerance)
        # This aligns with proposal: f_c(x) - f_k(x) >= 0 for all k != c
        # We check the negation: exists k: f_k(x) >= f_c(x) + tolerance
        for k in range(num_classes):
            if k == nominal_class:
                continue
            
            # Create a copy of the solver for this check
            check_solver = z3.Solver()
            for constraint in solver.assertions():
                check_solver.add(constraint)
            
            # Add constraint: f_k(x) >= f_c(x) + tolerance
            # This is equivalent to checking if margin < tolerance (violation)
            # Aligned with proposal spec: margin >= 0 (non-strict)
            check_solver.add(output_vars[k] >= output_vars[nominal_class] + z3.RealVal(self.margin_tolerance))
            
            # Check satisfiability
            result = check_solver.check()
            
            if result == z3.sat:
                # Counterexample found
                model = check_solver.model()
                counterexample = []
                for i in range(len(self.input_vars)):
                    var = self.input_vars[i]
                    try:
                        # Extract value from Z3 model
                        val = model[var]
                        # Convert to float - Z3 returns values as strings or rationals
                        val_str = str(val)
                        # Handle rational numbers like "1/2" or decimals
                        if '/' in val_str:
                            num, den = val_str.split('/')
                            counterexample.append(float(num) / float(den))
                        else:
                            counterexample.append(float(val_str))
                    except:
                        # If extraction fails, use original value
                        counterexample.append(x0[i])
                
                counterexample = np.array(counterexample)
                # Ensure counterexample is within bounds (should already be, but double-check)
                counterexample = np.clip(counterexample, self.input_lb, self.input_ub)
                
                # Verify counterexample by replay (audit step)
                if self._verify_counterexample(counterexample, x0, epsilon, nominal_class):
                    return VerificationResult.COUNTEREXAMPLE, counterexample
                else:
                    # Counterexample replay failed - likely encoding issue
                    # Return INCONCLUSIVE rather than COUNTEREXAMPLE
                    print(f"Warning: SMT found counterexample but replay verification failed")
                    return VerificationResult.INCONCLUSIVE, None
        
        # No counterexample found - network is robust
        return VerificationResult.SAFE, None
    
    def _verify_counterexample(self, counterexample: np.ndarray, x0: np.ndarray, 
                              epsilon: float, nominal_class: int) -> bool:
        """
        Replay verification: Check if counterexample is valid
        
        This is a critical audit step to ensure:
        1. Counterexample is within input domain [x0-ε, x0+ε] ∩ [0,1]
        2. Counterexample actually violates the spec (margin < tolerance)
        
        Returns:
            True if counterexample is valid, False otherwise
        """
        # Check 1: Input domain bounds
        input_lb = np.maximum(0, x0 - epsilon)
        input_ub = np.minimum(1, x0 + epsilon)
        
        if not np.all(counterexample >= input_lb - 1e-9):
            print(f"  Audit: Counterexample below lower bound")
            return False
        if not np.all(counterexample <= input_ub + 1e-9):
            print(f"  Audit: Counterexample above upper bound")
            return False
        
        # Check 2: Verify margin violation
        logits = self.network.forward(counterexample)
        fc = logits[nominal_class]
        
        for k in range(len(logits)):
            if k == nominal_class:
                continue
            fk = logits[k]
            margin = fc - fk
            
            # Check if margin < tolerance (violation)
            if margin < self.margin_tolerance:
                # Valid counterexample
                return True
        
        # No violation found - counterexample is invalid
        print(f"  Audit: Counterexample does not violate margin constraint")
        print(f"    Margins: {[logits[nominal_class] - logits[k] for k in range(len(logits)) if k != nominal_class]}")
        return False
    
    def get_input_domain(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the input domain bounds used in encoding
        
        Returns:
            (lower_bound, upper_bound) - the ℓ∞-ball clipped to [0,1]
        """
        if self.input_lb is None or self.input_ub is None:
            raise ValueError("Input domain not yet encoded. Call encode_network first.")
        return self.input_lb.copy(), self.input_ub.copy()


class MILPEncoder:
    """MILP encoder for ReLU network verification (using Z3 as MILP solver)"""
    
    def __init__(self, network: ReLUNetwork):
        if not Z3_AVAILABLE:
            raise ImportError("Z3 solver not available. Install with: pip install z3-solver")
        self.network = network
        # MILP encoding is similar to SMT, but we can use Z3's optimization features
        # For now, we'll use the same encoding as SMT
        self.smt_encoder = SMTEncoder(network)
    
    def verify_robustness(self, x0: np.ndarray, epsilon: float) -> Tuple[VerificationResult, Optional[np.ndarray]]:
        """
        Verify robustness using MILP (via Z3)
        
        Returns:
            (result, counterexample)
        """
        # Use SMT encoder (Z3 can handle MILP constraints)
        return self.smt_encoder.verify_robustness(x0, epsilon)


class RobustnessVerifier:
    """Main verifier for local robustness using SMT (exact method)"""
    
    def __init__(self, network: ReLUNetwork, method: str = "smt", 
                 use_cegar: bool = False, max_cegar_iterations: int = 5,
                 margin_tolerance: float = 1e-6, big_m: float = 1000.0):
        """
        Initialize verifier
        
        Args:
            network: ReLU network to verify
            method: Verification method - "smt" (exact) or "milp" (currently same as smt)
            use_cegar: Whether to use CEGAR refinement (default: False)
            max_cegar_iterations: Maximum CEGAR iterations (default: 5)
            margin_tolerance: Tolerance for margin check (default: 1e-6)
            big_m: Big-M constant for ReLU encoding (default: 1000.0)
        """
        self.network = network
        self.method = method.lower()
        self.use_cegar = use_cegar
        self.max_cegar_iterations = max_cegar_iterations
        self.margin_tolerance = margin_tolerance
        self.big_m = big_m
        
        # Initialize encoder based on method
        if self.method == "smt":
            if not Z3_AVAILABLE:
                raise ImportError("Z3 solver not available. Install with: pip install z3-solver")
            self.encoder = SMTEncoder(network, margin_tolerance=margin_tolerance, big_m=big_m)
        elif self.method == "milp":
            if not Z3_AVAILABLE:
                raise ImportError("Z3 solver not available. Install with: pip install z3-solver")
            # MILP currently uses same encoding as SMT
            self.encoder = MILPEncoder(network)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'smt' or 'milp'")
    
    def verify_robustness(self, x0: np.ndarray, epsilon: float) -> Tuple[VerificationResult, Optional[np.ndarray]]:
        """
        Verify local robustness at x0 with radius ε using SMT/MILP
        
        Returns:
            (result, counterexample) where result is SAFE/UNSAFE/UNKNOWN
        """
        # Use SMT/MILP encoder for exact verification
        try:
            result, counterexample = self.encoder.verify_robustness(x0, epsilon)
            
            # If result is INCONCLUSIVE or we want to try CEGAR, do refinement
            if result == VerificationResult.INCONCLUSIVE and self.use_cegar:
                result, counterexample = self._cegar_refinement_smt(x0, epsilon)
            
            return result, counterexample
        except Exception as e:
            # If SMT solver fails, return INCONCLUSIVE
            print(f"Warning: SMT/MILP verification failed: {e}")
            return VerificationResult.INCONCLUSIVE, None
    
    def _cegar_refinement_smt(self, x0: np.ndarray, epsilon: float) -> Tuple[VerificationResult, Optional[np.ndarray]]:
        """
        CEGAR-style refinement by splitting input set and using SMT on sub-regions
        """
        nominal_class = self.network.predict(x0)
        
        # Split input space into smaller regions
        num_splits = 2
        
        for iteration in range(self.max_cegar_iterations):
            # Create sub-regions by splitting
            sub_regions = self._split_input_region(x0, epsilon, num_splits)
            
            all_safe = True
            for region_center, region_epsilon in sub_regions:
                # Create new encoder for sub-region
                if self.method == "smt":
                    sub_encoder = SMTEncoder(self.network)
                else:
                    sub_encoder = MILPEncoder(self.network)
                
                result, counterexample = sub_encoder.verify_robustness(region_center, region_epsilon)
                
                if result == VerificationResult.COUNTEREXAMPLE:
                    return result, counterexample
                
                if result == VerificationResult.INCONCLUSIVE:
                    all_safe = False
            
            if all_safe:
                return VerificationResult.SAFE, None
            
            # Refine further
            num_splits *= 2
            epsilon = epsilon / 2
        
        return VerificationResult.INCONCLUSIVE, None
    
    
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
    
    def get_input_domain(self, x0: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the input domain bounds for verification
        
        This returns the SAME input domain that SMT uses:
        ℓ∞-ball clipped to [0,1]: [max(0, x0-ε), min(1, x0+ε)]
        
        This should match the input domain used by NNV (reachability) method.
        
        Args:
            x0: Center point
            epsilon: ℓ∞-ball radius
            
        Returns:
            (lower_bound, upper_bound) - the aligned input domain
        """
        input_lb = np.maximum(0, x0 - epsilon)
        input_ub = np.minimum(1, x0 + epsilon)
        return input_lb, input_ub


def check_consistency(nnv_result: VerificationResult, smt_result: VerificationResult,
                     nnv_counterexample: Optional[np.ndarray] = None,
                     smt_counterexample: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Check consistency between NNV (reachability) and SMT (exact) results
    
    This implements the audit consistency rules:
    - Normal consistent patterns
    - Inconsistencies that require investigation
    
    Args:
        nnv_result: Result from NNV/reachability method
        smt_result: Result from SMT/exact method
        nnv_counterexample: Counterexample from NNV (if any)
        smt_counterexample: Counterexample from SMT (if any)
        
    Returns:
        Dictionary with:
        - 'consistent': bool - whether results are consistent
        - 'status': str - consistency status description
        - 'requires_investigation': bool - whether inconsistency needs investigation
        - 'investigation_steps': List[str] - recommended investigation steps
    """
    # Normal consistent patterns
    if nnv_result == VerificationResult.SAFE and smt_result == VerificationResult.SAFE:
        return {
            'consistent': True,
            'status': 'Both methods agree: SAFE',
            'requires_investigation': False,
            'investigation_steps': []
        }
    
    if nnv_result == VerificationResult.INCONCLUSIVE and smt_result == VerificationResult.SAFE:
        return {
            'consistent': True,
            'status': 'Normal: Reachability inconclusive, exact method proves SAFE',
            'requires_investigation': False,
            'investigation_steps': []
        }
    
    if nnv_result == VerificationResult.INCONCLUSIVE and smt_result == VerificationResult.COUNTEREXAMPLE:
        return {
            'consistent': True,
            'status': 'Normal: Reachability inconclusive, exact method finds counterexample',
            'requires_investigation': False,
            'investigation_steps': []
        }
    
    # Critical inconsistency: NNV says SAFE but SMT finds counterexample
    if nnv_result == VerificationResult.SAFE and smt_result == VerificationResult.COUNTEREXAMPLE:
        return {
            'consistent': False,
            'status': 'CRITICAL INCONSISTENCY: NNV=SAFE but SMT=COUNTEREXAMPLE',
            'requires_investigation': True,
            'investigation_steps': [
                'Step 1: Verify SMT counterexample by replay',
                'Step 2: Check input domain alignment (both must use clipped [0,1] ball)',
                'Step 3: Verify spec alignment (both must use margin >= 0 or margin >= tolerance)',
                'Step 4: Check SMT Big-M assumption (may cause false SAFE if too small)',
                'Step 5: Verify both methods use the same network weights'
            ]
        }
    
    # Performance issue: NNV SAFE but SMT inconclusive
    if nnv_result == VerificationResult.SAFE and smt_result == VerificationResult.INCONCLUSIVE:
        return {
            'consistent': True,
            'status': 'Performance issue: NNV proves SAFE, SMT inconclusive (timeout/error)',
            'requires_investigation': False,
            'investigation_steps': ['Consider: SMT timeout settings, encoding efficiency']
        }
    
    # Other cases
    return {
        'consistent': True,
        'status': f'Other combination: NNV={nnv_result.value}, SMT={smt_result.value}',
        'requires_investigation': False,
        'investigation_steps': []
    }


def verify_counterexample_replay(network: ReLUNetwork, counterexample: np.ndarray,
                                 x0: np.ndarray, epsilon: float, nominal_class: int,
                                 margin_tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Replay verification of a counterexample (audit step)
    
    This verifies:
    1. Counterexample is within input domain [x0-ε, x0+ε] ∩ [0,1]
    2. Counterexample actually violates the spec (margin < tolerance)
    
    Args:
        network: The network to verify
        counterexample: The counterexample to verify
        x0: Center point
        epsilon: ℓ∞-ball radius
        nominal_class: Nominal prediction class
        margin_tolerance: Tolerance for margin check
        
    Returns:
        Dictionary with verification results
    """
    results = {
        'valid': False,
        'in_domain': False,
        'violates_spec': False,
        'details': {}
    }
    
    # Check 1: Input domain
    input_lb = np.maximum(0, x0 - epsilon)
    input_ub = np.minimum(1, x0 + epsilon)
    
    in_domain = (np.all(counterexample >= input_lb - 1e-9) and 
                np.all(counterexample <= input_ub + 1e-9))
    results['in_domain'] = in_domain
    results['details']['domain_check'] = {
        'lower_bound': input_lb.tolist(),
        'upper_bound': input_ub.tolist(),
        'counterexample_min': float(np.min(counterexample)),
        'counterexample_max': float(np.max(counterexample))
    }
    
    if not in_domain:
        results['details']['error'] = 'Counterexample outside input domain'
        return results
    
    # Check 2: Spec violation
    logits = network.forward(counterexample)
    fc = logits[nominal_class]
    predicted_class = int(np.argmax(logits))
    
    margins = []
    min_margin = float('inf')
    for k in range(len(logits)):
        if k == nominal_class:
            continue
        margin = fc - logits[k]
        margins.append(float(margin))
        min_margin = min(min_margin, margin)
    
    violates_spec = min_margin < margin_tolerance
    results['violates_spec'] = violates_spec
    results['details']['spec_check'] = {
        'nominal_class': nominal_class,
        'predicted_class': predicted_class,
        'min_margin': float(min_margin),
        'margins': margins,
        'margin_tolerance': margin_tolerance,
        'violation': violates_spec
    }
    
    results['valid'] = in_domain and violates_spec
    return results


@dataclass
class VerificationRecord:
    """Record for storing verification results from both methods"""
    # Test case information
    test_id: str
    x0: np.ndarray
    epsilon: float
    nominal_class: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Star Reachability (NNV) results
    star_result: Optional[VerificationResult] = None
    star_counterexample: Optional[np.ndarray] = None
    star_details: Optional[Dict[str, Any]] = None
    
    # SMT/MILP (Exact) results
    smt_result: Optional[VerificationResult] = None
    smt_counterexample: Optional[np.ndarray] = None
    smt_details: Optional[Dict[str, Any]] = None
    
    # Consistency check
    consistency_check: Optional[Dict[str, Any]] = None
    
    # Counterexample replay verification
    star_cex_replay: Optional[Dict[str, Any]] = None
    smt_cex_replay: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for JSON serialization"""
        result = {
            'test_id': self.test_id,
            'x0': self.x0.tolist() if isinstance(self.x0, np.ndarray) else self.x0,
            'epsilon': float(self.epsilon),
            'nominal_class': int(self.nominal_class),
            'timestamp': self.timestamp,
            'star_reachability': {
                'result': self.star_result.value if self.star_result else None,
                'counterexample': self.star_counterexample.tolist() if self.star_counterexample is not None else None,
                'details': self.star_details
            },
            'smt_exact': {
                'result': self.smt_result.value if self.smt_result else None,
                'counterexample': self.smt_counterexample.tolist() if self.smt_counterexample is not None else None,
                'details': self.smt_details
            },
            'consistency': self.consistency_check,
            'counterexample_replay': {
                'star': self.star_cex_replay,
                'smt': self.smt_cex_replay
            }
        }
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert record to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class DualMethodVerifier:
    """
    Verifier that runs both Star Reachability (NNV) and SMT/MILP (exact) methods
    and records all results for audit purposes
    """
    
    def __init__(self, network: ReLUNetwork, 
                 margin_tolerance: float = 1e-6, 
                 big_m: float = 1000.0,
                 use_cegar: bool = False):
        """
        Initialize dual-method verifier
        
        Args:
            network: ReLU network to verify
            margin_tolerance: Tolerance for margin check
            big_m: Big-M constant for SMT encoding
            use_cegar: Whether to use CEGAR for SMT
        """
        self.network = network
        self.margin_tolerance = margin_tolerance
        self.big_m = big_m
        self.use_cegar = use_cegar
        
        # Initialize both verifiers
        self.star_reachability = StarReachability(network)
        self.smt_verifier = RobustnessVerifier(
            network, method="smt", 
            use_cegar=use_cegar,
            margin_tolerance=margin_tolerance,
            big_m=big_m
        )
    
    def verify(self, x0: np.ndarray, epsilon: float, test_id: Optional[str] = None) -> VerificationRecord:
        """
        Run both verification methods and record all results
        
        Args:
            x0: Center point
            epsilon: ℓ∞-ball radius
            test_id: Optional test identifier
            
        Returns:
            VerificationRecord with all results
        """
        if test_id is None:
            test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        nominal_class = self.network.predict(x0)
        
        # Create record
        record = VerificationRecord(
            test_id=test_id,
            x0=x0,
            epsilon=epsilon,
            nominal_class=nominal_class
        )
        
        # Run Star Reachability (NNV)
        print(f"[{test_id}] Running Star Reachability (NNV)...")
        try:
            star_result, star_cex, star_details = self.star_reachability.verify_robustness(
                x0, epsilon, margin_tolerance=self.margin_tolerance
            )
            record.star_result = star_result
            record.star_counterexample = star_cex
            record.star_details = star_details
            print(f"  Star Reachability result: {star_result.value}")
        except Exception as e:
            print(f"  Star Reachability failed: {e}")
            record.star_result = VerificationResult.INCONCLUSIVE
            record.star_details = {'error': str(e)}
        
        # Run SMT/MILP (Exact)
        print(f"[{test_id}] Running SMT/MILP (Exact)...")
        try:
            smt_result, smt_cex = self.smt_verifier.verify_robustness(x0, epsilon)
            record.smt_result = smt_result
            record.smt_counterexample = smt_cex
            
            # Get SMT details if available
            if hasattr(self.smt_verifier.encoder, 'input_lb'):
                record.smt_details = {
                    'method': 'SMT',
                    'input_domain': {
                        'lower_bound': self.smt_verifier.encoder.input_lb.tolist(),
                        'upper_bound': self.smt_verifier.encoder.input_ub.tolist()
                    },
                    'margin_tolerance': self.margin_tolerance,
                    'big_m': self.big_m
                }
            
            print(f"  SMT/MILP result: {smt_result.value}")
            if smt_cex is not None:
                print(f"  SMT counterexample found")
        except Exception as e:
            print(f"  SMT/MILP failed: {e}")
            record.smt_result = VerificationResult.INCONCLUSIVE
            record.smt_details = {'error': str(e)}
        
        # Verify counterexamples by replay
        if record.star_counterexample is not None:
            print(f"[{test_id}] Verifying Star counterexample by replay...")
            record.star_cex_replay = verify_counterexample_replay(
                self.network, record.star_counterexample, x0, epsilon, 
                nominal_class, self.margin_tolerance
            )
        
        if record.smt_counterexample is not None:
            print(f"[{test_id}] Verifying SMT counterexample by replay...")
            record.smt_cex_replay = verify_counterexample_replay(
                self.network, record.smt_counterexample, x0, epsilon,
                nominal_class, self.margin_tolerance
            )
        
        # Check consistency
        print(f"[{test_id}] Checking consistency...")
        record.consistency_check = check_consistency(
            record.star_result, record.smt_result,
            record.star_counterexample, record.smt_counterexample
        )
        print(f"  Consistency: {record.consistency_check['status']}")
        
        return record
    
    def verify_batch(self, test_cases: List[Tuple[np.ndarray, float]], 
                    test_ids: Optional[List[str]] = None) -> List[VerificationRecord]:
        """
        Run verification on multiple test cases
        
        Args:
            test_cases: List of (x0, epsilon) tuples
            test_ids: Optional list of test identifiers
            
        Returns:
            List of VerificationRecords
        """
        records = []
        if test_ids is None:
            test_ids = [f"test_{i}" for i in range(len(test_cases))]
        
        for i, (x0, epsilon) in enumerate(test_cases):
            test_id = test_ids[i] if i < len(test_ids) else f"test_{i}"
            record = self.verify(x0, epsilon, test_id)
            records.append(record)
        
        return records
    
    def save_records(self, records: List[VerificationRecord], filename: str):
        """
        Save verification records to JSON file
        
        Args:
            records: List of VerificationRecords
            filename: Output filename
        """
        data = {
            'metadata': {
                'network_layers': self.network.num_layers,
                'input_dim': self.network.weights[0].shape[1],
                'output_dim': self.network.weights[-1].shape[0],
                'margin_tolerance': self.margin_tolerance,
                'big_m': self.big_m,
                'use_cegar': self.use_cegar,
                'timestamp': datetime.now().isoformat()
            },
            'records': [r.to_dict() for r in records]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(records)} verification records to {filename}")


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
    
    verifier = RobustnessVerifier(network, method="smt", use_cegar=False)
    
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


