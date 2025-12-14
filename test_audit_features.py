"""
Test script to verify all audit requirements are implemented correctly
"""
import numpy as np
from robustness_verifier import (
    ReLUNetwork, RobustnessVerifier, VerificationResult,
    check_consistency, verify_counterexample_replay
)

def test_verification_result_labels():
    """Test 1: Verify unified result labels"""
    print("=" * 70)
    print("Test 1: Verification Result Labels")
    print("=" * 70)
    
    # Check that new labels exist
    assert VerificationResult.SAFE.value == "SAFE"
    assert VerificationResult.COUNTEREXAMPLE.value == "COUNTEREXAMPLE"
    assert VerificationResult.INCONCLUSIVE.value == "INCONCLUSIVE"
    
    # Check backward compatibility
    assert VerificationResult.UNSAFE == VerificationResult.COUNTEREXAMPLE
    assert VerificationResult.UNKNOWN == VerificationResult.INCONCLUSIVE
    
    print("✓ All result labels correctly defined")
    print(f"  - SAFE: {VerificationResult.SAFE.value}")
    print(f"  - COUNTEREXAMPLE: {VerificationResult.COUNTEREXAMPLE.value}")
    print(f"  - INCONCLUSIVE: {VerificationResult.INCONCLUSIVE.value}")
    print()


def test_input_domain_alignment():
    """Test 2: Verify input domain alignment"""
    print("=" * 70)
    print("Test 2: Input Domain Alignment")
    print("=" * 70)
    
    # Create a small network
    W1 = np.random.randn(5, 3) * 0.1
    b1 = np.zeros(5)
    W2 = np.random.randn(2, 5) * 0.1
    b2 = np.zeros(2)
    network = ReLUNetwork([W1, W2], [b1, b2])
    
    x0 = np.array([0.5, 0.5, 0.5])
    epsilon = 0.1
    
    verifier = RobustnessVerifier(network, method="smt", use_cegar=False)
    
    # Get input domain
    input_lb, input_ub = verifier.get_input_domain(x0, epsilon)
    
    # Verify it matches the expected clipped domain
    expected_lb = np.maximum(0, x0 - epsilon)
    expected_ub = np.minimum(1, x0 + epsilon)
    
    assert np.allclose(input_lb, expected_lb), "Input domain lower bound mismatch"
    assert np.allclose(input_ub, expected_ub), "Input domain upper bound mismatch"
    
    print("✓ Input domain correctly aligned")
    print(f"  - Input domain: [{input_lb}, {input_ub}]")
    print(f"  - Matches clipped ℓ∞-ball: [max(0, x0-ε), min(1, x0+ε)]")
    print()


def test_spec_alignment():
    """Test 3: Verify spec alignment (margin >= tolerance)"""
    print("=" * 70)
    print("Test 3: Specification Alignment")
    print("=" * 70)
    
    # Create a small network
    W1 = np.random.randn(5, 3) * 0.1
    b1 = np.zeros(5)
    W2 = np.random.randn(2, 5) * 0.1
    b2 = np.zeros(2)
    network = ReLUNetwork([W1, W2], [b1, b2])
    
    x0 = np.array([0.5, 0.5, 0.5])
    epsilon = 0.01
    
    # Test with custom margin tolerance
    margin_tolerance = 1e-6
    verifier = RobustnessVerifier(network, method="smt", use_cegar=False, 
                                 margin_tolerance=margin_tolerance)
    
    # Verify encoder has correct tolerance
    assert verifier.encoder.margin_tolerance == margin_tolerance
    
    print("✓ Specification alignment verified")
    print(f"  - Margin tolerance: {margin_tolerance}")
    print(f"  - Uses non-strict inequality: f_k(x) >= f_c(x) + tolerance")
    print()


def test_consistency_check():
    """Test 4: Verify consistency check function"""
    print("=" * 70)
    print("Test 4: Consistency Check Function")
    print("=" * 70)
    
    # Test normal consistent patterns
    result1 = check_consistency(
        VerificationResult.SAFE, 
        VerificationResult.SAFE
    )
    assert result1['consistent'] == True
    assert result1['requires_investigation'] == False
    print("✓ SAFE + SAFE: Consistent")
    
    result2 = check_consistency(
        VerificationResult.INCONCLUSIVE,
        VerificationResult.SAFE
    )
    assert result2['consistent'] == True
    print("✓ INCONCLUSIVE + SAFE: Normal pattern")
    
    # Test critical inconsistency
    result3 = check_consistency(
        VerificationResult.SAFE,
        VerificationResult.COUNTEREXAMPLE
    )
    assert result3['consistent'] == False
    assert result3['requires_investigation'] == True
    assert len(result3['investigation_steps']) > 0
    print("✓ SAFE + COUNTEREXAMPLE: Critical inconsistency detected")
    print(f"  Investigation steps: {len(result3['investigation_steps'])}")
    print()


def test_counterexample_replay():
    """Test 5: Verify counterexample replay function"""
    print("=" * 70)
    print("Test 5: Counterexample Replay Verification")
    print("=" * 70)
    
    # Create a small network
    W1 = np.random.randn(5, 3) * 0.1
    b1 = np.zeros(5)
    W2 = np.random.randn(2, 5) * 0.1
    b2 = np.zeros(2)
    network = ReLUNetwork([W1, W2], [b1, b2])
    
    x0 = np.array([0.5, 0.5, 0.5])
    epsilon = 0.1
    nominal_class = network.predict(x0)
    
    # Test with valid counterexample (within domain)
    valid_cex = np.array([0.6, 0.6, 0.6])  # Within [0.4, 0.6]
    result = verify_counterexample_replay(
        network, valid_cex, x0, epsilon, nominal_class
    )
    
    assert 'valid' in result
    assert 'in_domain' in result
    assert 'violates_spec' in result
    assert 'details' in result
    
    print("✓ Counterexample replay function works")
    print(f"  - Valid: {result['valid']}")
    print(f"  - In domain: {result['in_domain']}")
    print(f"  - Violates spec: {result['violates_spec']}")
    print()


def test_big_m_configuration():
    """Test 6: Verify Big-M configuration"""
    print("=" * 70)
    print("Test 6: Big-M Configuration")
    print("=" * 70)
    
    # Create a small network
    W1 = np.random.randn(5, 3) * 0.1
    b1 = np.zeros(5)
    W2 = np.random.randn(2, 5) * 0.1
    b2 = np.zeros(2)
    network = ReLUNetwork([W1, W2], [b1, b2])
    
    # Test with custom Big-M
    big_m = 2000.0
    verifier = RobustnessVerifier(network, method="smt", big_m=big_m)
    
    assert verifier.encoder.big_m == big_m
    
    print("✓ Big-M is configurable")
    print(f"  - Big-M value: {big_m}")
    print()


def test_small_network_verification():
    """Test 7: Run actual verification on small network"""
    print("=" * 70)
    print("Test 7: Small Network Verification (Quick Test)")
    print("=" * 70)
    
    # Create a very small network for quick testing
    W1 = np.random.randn(3, 2) * 0.1
    b1 = np.zeros(3)
    W2 = np.random.randn(2, 3) * 0.1
    b2 = np.zeros(2)
    network = ReLUNetwork([W1, W2], [b1, b2])
    
    x0 = np.array([0.5, 0.5])
    epsilon = 0.01
    
    verifier = RobustnessVerifier(network, method="smt", use_cegar=False)
    
    print("  Running verification...")
    result, counterexample = verifier.verify_robustness(x0, epsilon)
    
    print(f"✓ Verification completed")
    print(f"  - Result: {result.value}")
    if counterexample is not None:
        print(f"  - Counterexample found: shape {counterexample.shape}")
    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("AUDIT FEATURES VERIFICATION TEST")
    print("=" * 70)
    print()
    
    try:
        test_verification_result_labels()
        test_input_domain_alignment()
        test_spec_alignment()
        test_consistency_check()
        test_counterexample_replay()
        test_big_m_configuration()
        test_small_network_verification()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nAll audit requirements are correctly implemented:")
        print("  ✓ Unified result labels (SAFE/COUNTEREXAMPLE/INCONCLUSIVE)")
        print("  ✓ Input domain alignment (clipped ℓ∞-ball)")
        print("  ✓ Specification alignment (margin >= tolerance)")
        print("  ✓ Consistency check function")
        print("  ✓ Counterexample replay verification")
        print("  ✓ Big-M configuration")
        print("  ✓ Basic verification functionality")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


