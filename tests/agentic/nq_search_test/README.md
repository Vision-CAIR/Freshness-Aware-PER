# NQ Search 数据集测试

测试转换后的 nq_search 数据集在 SearchEnv 中的交互情况。

## 测试内容

### test_nq_search_env.py

测试 nq_search 数据集与 SearchEnv 的完整交互流程，包括：

1. **环境创建测试** - 验证环境能否正确创建和加载数据
2. **环境重置测试** - 验证环境重置功能
3. **单次交互测试** - 测试搜索 -> 回答的完整流程
4. **多次搜索测试** - 验证多次搜索和最大搜索次数限制
5. **无效动作测试** - 验证无效动作的处理
6. **最大步数测试** - 验证最大步数截断
7. **不同问题测试** - 验证能加载不同问题
8. **奖励计算测试** - 验证奖励机制

## 运行测试

### 方法1: 直接运行
```bash
cd /data1/Chengyang_project/roll_dev/ROLL
python tests/agentic/nq_search_test/test_nq_search_env.py
```

### 方法2: 使用 pytest
```bash
cd /data1/Chengyang_project/roll_dev/ROLL
pytest tests/agentic/nq_search_test/test_nq_search_env.py -v -s
```

### 方法3: 在 docker 中运行
```bash
docker exec roll_vllm /bin/bash -c "cd /data1/Chengyang_project/roll_dev/ROLL && python tests/agentic/nq_search_test/test_nq_search_env.py"
```

## 数据集路径

测试使用以下转换后的数据集：

- 小样本测试: `/data1/Agentic_LLM-search/datasets/nq_search_converted/test_sample_128_searchenv.parquet`
- 完整测试集: `/data1/Agentic_LLM-search/datasets/nq_search_converted/test_searchenv.parquet`
- 训练集: `/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet`

## 预期结果

所有测试应该通过，验证：
- 数据集格式正确（system + user 两个消息）
- 环境能正确解析和使用数据
- 搜索功能正常工作
- 奖励计算正确
- 边界条件处理正确

