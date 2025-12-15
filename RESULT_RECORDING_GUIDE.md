# 结果记录功能使用指南

## 概述

代码已实现完整的结果记录功能，可以同时记录 **Star Reachability (NNV)** 和 **SMT/MILP (精确方法)** 的验证结果，用于审计和对比分析。

## 主要功能

### 1. DualMethodVerifier 类

`DualMethodVerifier` 类可以同时运行两种验证方法并记录所有结果：

```python
from robustness_verifier import ReLUNetwork, DualMethodVerifier
import numpy as np

# 创建网络
network = ReLUNetwork([W1, W2], [b1, b2])

# 创建双方法验证器
verifier = DualMethodVerifier(
    network,
    margin_tolerance=1e-6,  # 容忍阈值
    big_m=1000.0,           # Big-M常数
    use_cegar=False         # 是否使用CEGAR
)

# 运行验证
x0 = np.array([0.5, 0.5, 0.5])
epsilon = 0.1
record = verifier.verify(x0, epsilon, test_id="test_001")
```

### 2. VerificationRecord 数据结构

每个验证记录包含：

- **测试信息**：
  - `test_id`: 测试标识符
  - `x0`: 输入中心点
  - `epsilon`: ℓ∞球半径
  - `nominal_class`: 名义预测类别
  - `timestamp`: 时间戳

- **Star Reachability (NNV) 结果**：
  - `star_result`: 验证结果 (SAFE/COUNTEREXAMPLE/INCONCLUSIVE)
  - `star_counterexample`: 反例（如果有）
  - `star_details`: 详细信息（输出边界、margins等）

- **SMT/MILP (精确方法) 结果**：
  - `smt_result`: 验证结果
  - `smt_counterexample`: 精确反例（如果有）
  - `smt_details`: 详细信息（输入域、参数等）

- **一致性检查**：
  - `consistency_check`: 两种方法结果的一致性分析

- **反例回放验证**：
  - `star_cex_replay`: Star方法反例的回放验证
  - `smt_cex_replay`: SMT方法反例的回放验证

### 3. 批量验证

```python
# 批量验证多个测试用例
test_cases = [
    (x0_1, epsilon_1),
    (x0_2, epsilon_2),
    (x0_3, epsilon_3),
]
test_ids = ["test_001", "test_002", "test_003"]

records = verifier.verify_batch(test_cases, test_ids)
```

### 4. 保存结果到JSON

```python
# 保存所有记录到JSON文件
verifier.save_records(records, "verification_results.json")
```

JSON文件包含：
- **metadata**: 网络信息、参数配置
- **records**: 所有验证记录的列表

## 记录内容详解

### Star Reachability 记录

```json
"star_reachability": {
  "result": "SAFE",
  "counterexample": null,
  "details": {
    "method": "Star Reachability",
    "input_domain": {
      "center": [0.5, 0.5, 0.5],
      "epsilon": 0.1,
      "lower_bound": [0.4, 0.4, 0.4],
      "upper_bound": [0.6, 0.6, 0.6]
    },
    "output_bounds": {
      "lower_bound": [...],
      "upper_bound": [...]
    },
    "margins": [
      {
        "class": 0,
        "margin_lb": 0.0083,
        "margin_ub": 0.0157
      }
    ],
    "min_margin": 0.0083,
    "nominal_class": 1
  }
}
```

### SMT/MILP 记录

```json
"smt_exact": {
  "result": "SAFE",
  "counterexample": null,
  "details": {
    "method": "SMT",
    "input_domain": {
      "lower_bound": [0.4, 0.4, 0.4],
      "upper_bound": [0.6, 0.6, 0.6]
    },
    "margin_tolerance": 1e-06,
    "big_m": 1000.0
  }
}
```

### 一致性检查记录

```json
"consistency": {
  "consistent": true,
  "status": "Both methods agree: SAFE",
  "requires_investigation": false,
  "investigation_steps": []
}
```

### 反例回放验证记录

```json
"counterexample_replay": {
  "star": null,
  "smt": {
    "valid": true,
    "in_domain": true,
    "violates_spec": true,
    "details": {
      "domain_check": {...},
      "spec_check": {
        "nominal_class": 1,
        "predicted_class": 0,
        "min_margin": -0.001,
        "margins": [...],
        "violation": true
      }
    }
  }
}
```

## 使用示例

完整示例请参考 `test_dual_method_recording.py`：

```bash
python test_dual_method_recording.py
```

## 审计用途

这个结果记录系统支持：

1. **输入域对齐验证**：确保两种方法使用相同的输入域
2. **规格对齐验证**：确保两种方法使用相同的规格（margin >= tolerance）
3. **一致性检查**：自动检测两种方法结果的不一致
4. **反例验证**：通过回放验证反例的有效性
5. **完整审计追踪**：所有结果都记录在JSON中，便于后续分析

## 注意事项

- Star Reachability 是过度近似方法，通常不会产生精确反例
- SMT/MILP 是精确方法，可以找到精确反例
- 两种方法都使用相同的输入域（裁剪后的ℓ∞球）
- 两种方法都使用相同的规格（margin >= tolerance）




