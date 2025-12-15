"""View verification results summary"""
import json

with open('verification_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 70)
print("VERIFICATION RESULTS SUMMARY")
print("=" * 70)
print()
print("Metadata:")
print(f"  Network: {data['metadata']['network_layers']} layers")
print(f"  Input dim: {data['metadata']['input_dim']}")
print(f"  Output dim: {data['metadata']['output_dim']}")
print(f"  Margin tolerance: {data['metadata']['margin_tolerance']}")
print(f"  Big-M: {data['metadata']['big_m']}")
print(f"  Timestamp: {data['metadata']['timestamp']}")
print()
print(f"Total Records: {len(data['records'])}")
print()
print("=" * 70)
print("DETAILED RECORDS")
print("=" * 70)

for i, record in enumerate(data['records'], 1):
    print(f"\n[{i}] Test ID: {record['test_id']}")
    print(f"    Timestamp: {record['timestamp']}")
    print(f"    Nominal Class: {record['nominal_class']}")
    print(f"    Epsilon: {record['epsilon']}")
    print()
    print(f"    Star Reachability (NNV):")
    print(f"      Result: {record['star_reachability']['result']}")
    print(f"      Counterexample: {'Found' if record['star_reachability']['counterexample'] else 'None'}")
    if record['star_reachability']['details']:
        details = record['star_reachability']['details']
        print(f"      Min Margin: {details.get('min_margin', 'N/A')}")
        print(f"      Input Domain: [{details['input_domain']['lower_bound'][0]:.3f}, {details['input_domain']['upper_bound'][0]:.3f}]")
    print()
    print(f"    SMT/MILP (Exact):")
    print(f"      Result: {record['smt_exact']['result']}")
    print(f"      Counterexample: {'Found' if record['smt_exact']['counterexample'] else 'None'}")
    if record['smt_exact']['details']:
        details = record['smt_exact']['details']
        print(f"      Input Domain: [{details['input_domain']['lower_bound'][0]:.3f}, {details['input_domain']['upper_bound'][0]:.3f}]")
    print()
    print(f"    Consistency Check:")
    if record['consistency']:
        print(f"      Status: {record['consistency']['status']}")
        print(f"      Consistent: {record['consistency']['consistent']}")
        print(f"      Requires Investigation: {record['consistency']['requires_investigation']}")
    print()
    print(f"    Counterexample Replay:")
    print(f"      Star: {'Verified' if record['counterexample_replay']['star'] else 'N/A'}")
    print(f"      SMT: {'Verified' if record['counterexample_replay']['smt'] else 'N/A'}")

print()
print("=" * 70)
print("✅ Results viewing completed")
print("=" * 70)



