"""
Test script for dual-method verification with result recording
"""
import numpy as np
from robustness_verifier import (
    ReLUNetwork, DualMethodVerifier, VerificationRecord
)

def main():
    """Test dual-method verification with result recording"""
    print("=" * 70)
    print("Dual-Method Verification with Result Recording Test")
    print("=" * 70)
    print()
    
    # Create a small network for quick testing
    print("Creating test network...")
    np.random.seed(42)
    W1 = np.random.randn(5, 3) * 0.1
    b1 = np.zeros(5)
    W2 = np.random.randn(2, 5) * 0.1
    b2 = np.zeros(2)
    network = ReLUNetwork([W1, W2], [b1, b2])
    print(f"✓ Network created: {network.num_layers} layers")
    print(f"  Input dim: {network.weights[0].shape[1]}")
    print(f"  Output dim: {network.weights[-1].shape[0]}")
    print()
    
    # Create dual-method verifier
    print("Creating dual-method verifier...")
    verifier = DualMethodVerifier(
        network,
        margin_tolerance=1e-6,
        big_m=1000.0,
        use_cegar=False
    )
    print("✓ Dual-method verifier created")
    print()
    
    # Test case 1: Single verification
    print("=" * 70)
    print("Test Case 1: Single Verification")
    print("=" * 70)
    x0 = np.array([0.5, 0.5, 0.5])
    epsilon = 0.1
    
    record = verifier.verify(x0, epsilon, test_id="test_001")
    
    print("\n" + "-" * 70)
    print("Verification Record Summary:")
    print("-" * 70)
    print(f"Test ID: {record.test_id}")
    print(f"Nominal Class: {record.nominal_class}")
    print(f"Epsilon: {record.epsilon}")
    print()
    print("Star Reachability (NNV):")
    print(f"  Result: {record.star_result.value if record.star_result else 'None'}")
    print(f"  Counterexample: {'Found' if record.star_counterexample is not None else 'None'}")
    if record.star_details:
        print(f"  Min Margin: {record.star_details.get('min_margin', 'N/A')}")
    print()
    print("SMT/MILP (Exact):")
    print(f"  Result: {record.smt_result.value if record.smt_result else 'None'}")
    print(f"  Counterexample: {'Found' if record.smt_counterexample is not None else 'None'}")
    print()
    print("Consistency Check:")
    if record.consistency_check:
        print(f"  Status: {record.consistency_check['status']}")
        print(f"  Consistent: {record.consistency_check['consistent']}")
        print(f"  Requires Investigation: {record.consistency_check['requires_investigation']}")
    print()
    
    # Test case 2: Batch verification
    print("=" * 70)
    print("Test Case 2: Batch Verification")
    print("=" * 70)
    test_cases = [
        (np.array([0.3, 0.3, 0.3]), 0.05),
        (np.array([0.7, 0.7, 0.7]), 0.05),
    ]
    test_ids = ["test_002", "test_003"]
    
    records = verifier.verify_batch(test_cases, test_ids)
    print(f"\n✓ Batch verification completed: {len(records)} records")
    print()
    
    # Test case 3: Save records to JSON
    print("=" * 70)
    print("Test Case 3: Save Records to JSON")
    print("=" * 70)
    all_records = [record] + records
    verifier.save_records(all_records, "verification_records.json")
    print()
    
    # Test case 4: Display JSON for one record
    print("=" * 70)
    print("Test Case 4: Record JSON Format")
    print("=" * 70)
    print("Sample record JSON (first 500 chars):")
    json_str = record.to_json()
    print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
    print()
    
    print("=" * 70)
    print("✅ All tests completed successfully!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  - Total records created: {len(all_records)}")
    print(f"  - Records saved to: verification_records.json")
    print(f"  - Each record contains:")
    print(f"    * Star Reachability results")
    print(f"    * SMT/MILP results")
    print(f"    * Consistency check")
    print(f"    * Counterexample replay verification")

if __name__ == "__main__":
    main()

