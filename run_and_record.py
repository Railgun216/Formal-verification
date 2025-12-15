"""
Run dual-method verification and record all results
"""
import numpy as np
from robustness_verifier import ReLUNetwork, DualMethodVerifier

def main():
    """Run verification and save results"""
    print("=" * 70)
    print("Running Dual-Method Verification with Result Recording")
    print("=" * 70)
    print()
    
    # Create network
    print("Creating test network...")
    np.random.seed(42)
    W1 = np.random.randn(10, 5) * 0.1
    b1 = np.zeros(10)
    W2 = np.random.randn(3, 10) * 0.1
    b2 = np.zeros(3)
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
    print("✓ Verifier created")
    print()
    
    # Define test cases
    print("=" * 70)
    print("Running Verification Tests")
    print("=" * 70)
    print()
    
    test_cases = [
        (np.array([0.5, 0.5, 0.5, 0.5, 0.5]), 0.05),
        (np.array([0.3, 0.3, 0.3, 0.3, 0.3]), 0.1),
        (np.array([0.7, 0.7, 0.7, 0.7, 0.7]), 0.1),
    ]
    test_ids = ["test_001", "test_002", "test_003"]
    
    # Run batch verification
    records = verifier.verify_batch(test_cases, test_ids)
    
    print()
    print("=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    for record in records:
        print(f"\n[{record.test_id}]")
        print(f"  Nominal Class: {record.nominal_class}")
        print(f"  Epsilon: {record.epsilon}")
        print(f"  Star Reachability: {record.star_result.value if record.star_result else 'N/A'}")
        print(f"  SMT/MILP: {record.smt_result.value if record.smt_result else 'N/A'}")
        if record.consistency_check:
            print(f"  Consistency: {record.consistency_check['status']}")
    
    # Save results
    print()
    print("=" * 70)
    print("Saving Results")
    print("=" * 70)
    verifier.save_records(records, "verification_results.json")
    
    print()
    print("=" * 70)
    print("✅ Verification completed and results saved!")
    print("=" * 70)
    print(f"\nResults saved to: verification_results.json")
    print(f"Total records: {len(records)}")

if __name__ == "__main__":
    main()



