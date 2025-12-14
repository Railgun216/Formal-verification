"""快速验证测试 - 检查所有功能是否正常"""
import sys

def test_imports():
    """测试导入"""
    try:
        from robustness_verifier import ReLUNetwork, RobustnessVerifier, StarSet, VerificationResult
        print("✓ 模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_network_creation():
    """测试网络创建"""
    try:
        import numpy as np
        from robustness_verifier import ReLUNetwork
        
        W1 = np.random.randn(5, 3) * 0.1
        b1 = np.zeros(5)
        W2 = np.random.randn(2, 5) * 0.1
        b2 = np.zeros(2)
        
        net = ReLUNetwork([W1, W2], [b1, b2])
        print(f"✓ 网络创建成功: {net.num_layers} 层")
        return True, net
    except Exception as e:
        print(f"✗ 网络创建失败: {e}")
        return False, None

def test_verification(network):
    """测试验证功能"""
    try:
        import numpy as np
        from robustness_verifier import RobustnessVerifier
        
        verifier = RobustnessVerifier(network, method="smt", use_cegar=False)
        x0 = np.array([0.5, 0.5, 0.5])
        result, counterexample = verifier.verify_robustness(x0, 0.01)
        
        print(f"✓ 验证完成: {result.value}")
        return True
    except Exception as e:
        print(f"✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("快速功能验证测试")
    print("=" * 60)
    print()
    
    # 测试 1: 导入
    print("[测试 1] 模块导入...")
    if not test_imports():
        print("\n❌ 测试失败：无法导入模块")
        return False
    print()
    
    # 测试 2: 网络创建
    print("[测试 2] 网络创建...")
    success, network = test_network_creation()
    if not success:
        print("\n❌ 测试失败：无法创建网络")
        return False
    print()
    
    # 测试 3: 验证功能
    print("[测试 3] 鲁棒性验证...")
    if not test_verification(network):
        print("\n❌ 测试失败：验证功能异常")
        return False
    print()
    
    print("=" * 60)
    print("✅ 所有测试通过！代码可以正常使用。")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


