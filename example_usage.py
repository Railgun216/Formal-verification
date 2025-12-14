"""
Example script for loading models and running verification

This script demonstrates how to use the robustness verifier with different
model formats and test configurations.
"""

import numpy as np
from robustness_verifier import ReLUNetwork, RobustnessVerifier, VerificationResult
import os

# Check if ONNX is available
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def load_mnist_sample(index: int = 0, dataset_size: int = 784) -> np.ndarray:
    """
    Load a sample from MNIST dataset (placeholder - use actual data)
    
    Args:
        index: Sample index
        dataset_size: Input dimension (784 for MNIST)
    
    Returns:
        Normalized sample vector
    """
    # In practice, load from actual MNIST data
    # For demonstration, generate random normalized sample
    np.random.seed(index)
    sample = np.random.rand(dataset_size)
    return sample / 255.0  # Normalize to [0, 1] range


def create_dummy_network(input_dim: int = 784, hidden_sizes: List[int] = [64, 32], 
                         output_dim: int = 10) -> ReLUNetwork:
    """
    Create a dummy ReLU MLP network for testing
    
    Args:
        input_dim: Input dimension
        hidden_sizes: List of hidden layer sizes
        output_dim: Output dimension (number of classes)
    
    Returns:
        ReLUNetwork instance
    """
    weights = []
    biases = []
    
    # Input layer
    prev_size = input_dim
    for hidden_size in hidden_sizes:
        W = np.random.randn(hidden_size, prev_size) * 0.1
        b = np.zeros(hidden_size)
        weights.append(W)
        biases.append(b)
        prev_size = hidden_size
    
    # Output layer
    W_out = np.random.randn(output_dim, prev_size) * 0.1
    b_out = np.zeros(output_dim)
    weights.append(W_out)
    biases.append(b_out)
    
    return ReLUNetwork(weights, biases)


def verify_single_sample(network: ReLUNetwork, x0: np.ndarray, 
                         epsilon: float, verbose: bool = True) -> Dict:
    """
    Verify robustness for a single sample
    
    Returns:
        Dictionary with verification results
    """
    verifier = RobustnessVerifier(network, method="smt", use_cegar=False)
    
    if verbose:
        print(f"  Verifying sample with ε = {epsilon:.3f}...", end=" ")
    
    result, counterexample = verifier.verify_robustness(x0, epsilon)
    
    result_dict = {
        "result": result.value,
        "epsilon": epsilon,
        "nominal_class": network.predict(x0),
        "has_counterexample": counterexample is not None
    }
    
    if counterexample is not None:
        result_dict["adversarial_class"] = network.predict(counterexample)
        result_dict["counterexample"] = counterexample
    
    if verbose:
        print(f"{result.value}")
        if counterexample is not None:
            print(f"    ⚠️  Counterexample found! Class changed from {result_dict['nominal_class']} to {result_dict['adversarial_class']}")
    
    return result_dict


def verify_multiple_samples(network: ReLUNetwork, samples: List[np.ndarray],
                            epsilon: float, verbose: bool = True) -> Dict[str, int]:
    """
    Verify robustness for multiple samples
    
    Returns:
        Dictionary with result counts
    """
    verifier = RobustnessVerifier(network, method="smt")
    
    results = {
        "SAFE": 0,
        "UNSAFE": 0,
        "UNKNOWN": 0
    }
    
    for i, x0 in enumerate(samples):
        result, _ = verifier.verify_robustness(x0, epsilon)
        results[result.value] += 1
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples")
    
    return results


def run_verification_experiment():
    """Run a complete verification experiment"""
    
    print("=" * 70)
    print("Local Robustness Verification Experiment")
    print("=" * 70)
    
    # Create or load network
    print("\n1. Loading network...")
    print("   (Using dummy network - replace with actual model loading)")
    
    # Option 1: Create dummy network
    network = create_dummy_network(input_dim=784, hidden_sizes=[64, 32], output_dim=10)
    
    # Option 2: Load from .mat file (uncomment if you have a model)
    # try:
    #     network = ReLUNetwork.from_mat("model.mat")
    #     print("   ✓ Loaded network from model.mat")
    # except FileNotFoundError:
    #     print("   ⚠ Model file not found, using dummy network")
    #     network = create_dummy_network()
    
    # Option 3: Load from ONNX file (uncomment if you have a model)
    # try:
    #     network = ReLUNetwork.from_onnx("model.onnx")
    #     print("   ✓ Loaded network from model.onnx")
    # except FileNotFoundError:
    #     print("   ⚠ Model file not found, using dummy network")
    #     network = create_dummy_network()
    
    print(f"   Network architecture: {network.num_layers} layers")
    print(f"   Input dimension: {network.weights[0].shape[1]}")
    print(f"   Output dimension: {network.weights[-1].shape[0]}")
    
    # Test with different epsilon values
    epsilons = [0.01, 0.03, 0.05]
    
    print("\n2. Running verification experiments...")
    print("-" * 70)
    
    for epsilon in epsilons:
        print(f"\n{'='*70}")
        print(f"Verification with ε = {epsilon}")
        print(f"{'='*70}")
        
        # Test on a few samples
        num_samples = 5
        samples = [load_mnist_sample(i) for i in range(num_samples)]
        
        print(f"\nTesting {num_samples} samples:")
        all_results = []
        
        for i, x0 in enumerate(samples):
            print(f"\nSample {i+1}:")
            result_dict = verify_single_sample(network, x0, epsilon, verbose=True)
            all_results.append(result_dict)
        
        # Summary statistics
        result_counts = {
            "SAFE": sum(1 for r in all_results if r["result"] == "SAFE"),
            "UNSAFE": sum(1 for r in all_results if r["result"] == "UNSAFE"),
            "UNKNOWN": sum(1 for r in all_results if r["result"] == "UNKNOWN")
        }
        
        print(f"\nSummary for ε = {epsilon}:")
        print(f"  SAFE:    {result_counts['SAFE']}/{num_samples} ({result_counts['SAFE']/num_samples*100:.1f}%)")
        print(f"  UNSAFE:  {result_counts['UNSAFE']}/{num_samples} ({result_counts['UNSAFE']/num_samples*100:.1f}%)")
        print(f"  UNKNOWN: {result_counts['UNKNOWN']}/{num_samples} ({result_counts['UNKNOWN']/num_samples*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Experiment completed!")
    print("=" * 70)


def test_model_loading():
    """Test model loading from different formats"""
    print("Testing model loading...")
    
    # Test .mat loading
    print("\n1. Testing .mat file loading...")
    # Create a dummy .mat file for testing
    try:
        import scipy.io as sio
        test_weights = [np.random.randn(10, 5), np.random.randn(3, 10)]
        test_biases = [np.zeros(10), np.zeros(3)]
        
        test_data = {
            'W1': test_weights[0],
            'b1': test_biases[0],
            'W2': test_weights[1],
            'b2': test_biases[1]
        }
        sio.savemat('test_model.mat', test_data)
        
        network = ReLUNetwork.from_mat('test_model.mat')
        print("   ✓ Successfully loaded from .mat file")
        print(f"   Network has {network.num_layers} layers")
        
        # Clean up
        os.remove('test_model.mat')
    except Exception as e:
        print(f"   ✗ Error loading .mat file: {e}")
    
    # Test ONNX loading (if available)
    if ONNX_AVAILABLE:
        print("\n2. Testing ONNX file loading...")
        print("   (Skipped - requires actual ONNX model file)")
    else:
        print("\n2. ONNX support not available")
        print("   Install with: pip install onnx onnxruntime")


if __name__ == "__main__":
    import sys
    from typing import List
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_model_loading()
    else:
        run_verification_experiment()

