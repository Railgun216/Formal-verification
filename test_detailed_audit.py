"""
Detailed test for audit features - counterexample replay and consistency checking
"""
import numpy as np
from robustness_verifier import (
    ReLUNetwork, RobustnessVerifier, VerificationResult,
    check_consistency, verify_counterexample_replay
)

def test_counterexample_replay_detailed():
    """Test counterexample replay with detailed scenarios"""
    print("=" * 70)
    print("Detailed Counterexample Replay Test")
    print("=" * 70)
    
    # Create a simple network
    np.random.seed(42)
    W1 = np.random.randn(3, 2) * 0.1
    b1 = np.zeros(3)
    W2 = np.random.randn(2, 3) * 0.1
    b2 = np.zeros(2)
    network = ReLUNetwork([W1, W2], [b1, b2])
    
    x0 = np.array([0.5, 0.5])
    epsilon = 0.1
    nominal_class = network.predict(x0)
    
    # Test case 1: Valid counterexample (within domain)
    print("\n[Test Case 1] Counterexample within domain")
    cex1 = np.array([0.55, 0.55])  # Within [0.4, 0.6]
    result1 = verify_counterexample_replay(
        network, cex1, x0, epsilon, nominal_class
    )
    print(f"  In domain: {result1['in_domain']}")
    print(f"  Violates spec: {result1['violates_spec']}")
    print(f"  Valid: {result1['valid']}")
    
    # Test case 2: Counterexample outside domain
    print("\n[Test Case 2] Counterexample outside domain")
    cex2 = np.array([0.8, 0.8])  # Outside [0.4, 0.6]
    result2 = verify_counterexample_replay(
        network, cex2, x0, epsilon, nominal_class
    )
    print(f"  In domain: {result2['in_domain']}")
    print(f"  Valid: {result2['valid']}")
    assert result2['in_domain'] == False, "Should detect out-of-domain counterexample"
    
    print("\n✓ Counterexample replay correctly validates domain and spec")
    print()


def test_consistency_scenarios():
    """Test various consistency scenarios"""
    print("=" * 70)
    print("Consistency Check Scenarios")
    print("=" * 70)
    
    scenarios = [
        ("Both SAFE", VerificationResult.SAFE, VerificationResult.SAFE, True, False),
        ("NNV INCONCLUSIVE, SMT SAFE", VerificationResult.INCONCLUSIVE, VerificationResult.SAFE, True, False),
        ("NNV INCONCLUSIVE, SMT COUNTEREXAMPLE", VerificationResult.INCONCLUSIVE, VerificationResult.COUNTEREXAMPLE, True, False),
        ("CRITICAL: NNV SAFE, SMT COUNTEREXAMPLE", VerificationResult.SAFE, VerificationResult.COUNTEREXAMPLE, False, True),
        ("NNV SAFE, SMT INCONCLUSIVE", VerificationResult.SAFE, VerificationResult.INCONCLUSIVE, True, False),
    ]
    
    for name, nnv_result, smt_result, expected_consistent, expected_investigation in scenarios:
        result = check_consistency(nnv_result, smt_result)
        print(f"\n[{name}]")
        print(f"  Consistent: {result['consistent']} (expected: {expected_consistent})")
        print(f"  Requires investigation: {result['requires_investigation']} (expected: {expected_investigation})")
        print(f"  Status: {result['status']}")
        
        if result['investigation_steps']:
            print(f"  Investigation steps: {len(result['investigation_steps'])}")
            for i, step in enumerate(result['investigation_steps'][:2], 1):
                print(f"    {i}. {step}")
        
        assert result['consistent'] == expected_consistent, f"Consistency mismatch for {name}"
        assert result['requires_investigation'] == expected_investigation, f"Investigation mismatch for {name}"
    
    print("\n✓ All consistency scenarios correctly handled")
    print()


def test_input_domain_consistency():
    """Test that input domain is consistently defined"""
    print("=" * 70)
    print("Input Domain Consistency Test")
    print("=" * 70)
    
    # Create network
    W1 = np.random.randn(3, 2) * 0.1
    b1 = np.zeros(3)
    W2 = np.random.randn(2, 3) * 0.1
    b2 = np.zeros(2)
    network = ReLUNetwork([W1, W2], [b1, b2])
    
    x0 = np.array([0.3, 0.7])
    epsilon = 0.2
    
    verifier = RobustnessVerifier(network, method="smt")
    
    # Get input domain from verifier
    input_lb, input_ub = verifier.get_input_domain(x0, epsilon)
    
    # Manually compute expected domain
    expected_lb = np.maximum(0, x0 - epsilon)
    expected_ub = np.minimum(1, x0 + epsilon)
    
    print(f"  x0: {x0}")
    print(f"  epsilon: {epsilon}")
    print(f"  Computed domain: [{input_lb}, {input_ub}]")
    print(f"  Expected domain: [{expected_lb}, {expected_ub}]")
    
    assert np.allclose(input_lb, expected_lb), "Lower bound mismatch"
    assert np.allclose(input_ub, expected_ub), "Upper bound mismatch"
    
    # Test edge case: x0 near boundaries
    print("\n  Testing edge cases:")
    x0_edge1 = np.array([0.05, 0.5])
    lb1, ub1 = verifier.get_input_domain(x0_edge1, 0.1)
    print(f"    x0={x0_edge1}, ε=0.1: [{lb1}, {ub1}]")
    assert lb1[0] >= 0, "Should clip to 0"
    
    x0_edge2 = np.array([0.95, 0.5])
    lb2, ub2 = verifier.get_input_domain(x0_edge2, 0.1)
    print(f"    x0={x0_edge2}, ε=0.1: [{lb2}, {ub2}]")
    assert ub2[0] <= 1, "Should clip to 1"
    
    print("\n✓ Input domain consistently defined and clipped")
    print()


def test_margin_tolerance():
    """Test margin tolerance configuration"""
    print("=" * 70)
    print("Margin Tolerance Test")
    print("=" * 70)
    
    # Create network
    W1 = np.random.randn(3, 2) * 0.1
    b1 = np.zeros(3)
    W2 = np.random.randn(2, 3) * 0.1
    b2 = np.zeros(2)
    network = ReLUNetwork([W1, W2], [b1, b2])
    
    # Test different tolerance values
    tolerances = [1e-6, 1e-5, 1e-4]
    
    for tol in tolerances:
        verifier = RobustnessVerifier(network, method="smt", margin_tolerance=tol)
        assert verifier.encoder.margin_tolerance == tol
        print(f"  Tolerance {tol}: ✓")
    
    print("\n✓ Margin tolerance correctly configurable")
    print()


def main():
    """Run detailed audit tests"""
    print("\n" + "=" * 70)
    print("DETAILED AUDIT FEATURES TEST")
    print("=" * 70)
    print()
    
    try:
        test_counterexample_replay_detailed()
        test_consistency_scenarios()
        test_input_domain_consistency()
        test_margin_tolerance()
        
        print("=" * 70)
        print("✅ ALL DETAILED TESTS PASSED!")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ Counterexample replay validates domain and spec")
        print("  ✓ Consistency check handles all scenarios correctly")
        print("  ✓ Input domain is consistently defined and clipped")
        print("  ✓ Margin tolerance is configurable")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

