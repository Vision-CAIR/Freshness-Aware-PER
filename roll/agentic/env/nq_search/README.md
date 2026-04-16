## NQ Search Environment

基于 Natural Questions (NQ) 数据集的检索增强问答环境，使用专用的检索服务器提供高质量的搜索结果。

## 特性

### 1. 检索服务器集成
- 使用独立的检索服务器 (默认: `http://127.0.0.1:8100/retrieve`)
- 基于 FAISS 索引的高效向量检索
- E5 模型编码，保证检索质量
- 支持批量检索和相似度打分

### 2. NQ 数据集支持
- 专为 Natural Questions 数据集设计
- 支持转换后的 SearchEnv 格式
- 包含系统提示词和用户问题

### 3. 标签系统
与原始 NQ 数据集标签保持一致：
- `<think>content</think>`: 推理过程
- `<search>query</search>`: 搜索查询
- `<information>results</information>`: 搜索结果
- `<answer>answer</answer>`: 最终答案

### 4. EM 评估
- 精确匹配 (Exact Match) 评估
- 支持忽略大小写、标点符号、冠词
- 提供详细的评估信息

## 使用方法

### 基本配置

```python
from roll.agentic.env.nq_search import NQSearchEnv, NQSearchEnvConfig

config = NQSearchEnvConfig(
    dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet",
    max_instances=1000,
    retrieval_server_url="http://127.0.0.1:8100/retrieve",
    retrieval_topk=3,
    max_steps=10,
    max_search_calls=5
)

env = NQSearchEnv(config)
```

### 交互流程

```python
# 重置环境
obs, info = env.reset(seed=42)

# 第一步：搜索
action1 = '''<think>
I need to search for information about this question.
</think>

<search>nobel prize physics first winner</search>'''

obs, reward, terminated, truncated, info = env.step(action1)

# 第二步：回答
action2 = '''<think>
Based on the search results, the answer is Wilhelm Conrad Röntgen.
</think>

<answer>Wilhelm Conrad Röntgen</answer>'''

obs, reward, terminated, truncated, info = env.step(action2)
print(f"Success: {info['success']}, Score: {info['score']}")
```

### 轨迹格式

```
observation0: System prompt + Question
action0: <think>...</think> <search>query</search>
observation1: <information>search results</information>
action1: <think>...</think> <answer>answer</answer>
observation2: Final feedback + reward
```

## 配置参数

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dataset_path` | train_searchenv.parquet | 数据集路径 |
| `max_instances` | 1000 | 最大实例数 |
| `retrieval_server_url` | http://127.0.0.1:8100/retrieve | 检索服务器地址 |
| `retrieval_topk` | 3 | 返回前k个文档 |
| `retrieval_timeout` | 30 | 超时时间（秒） |

### 环境设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_steps` | 10 | 最大步数 |
| `max_search_calls` | 5 | 最大搜索次数 |
| `disable_limiter` | False | 是否禁用限流 |

### 奖励设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_outcome_reward_only` | True | 只使用最终结果奖励 |
| `correct_answer_reward` | 1.0 | 正确答案奖励 |
| `use_em_evaluation` | True | 使用EM评估 |
| `em_ignore_case` | True | EM忽略大小写 |
| `em_ignore_punctuation` | True | EM忽略标点 |
| `em_ignore_articles` | True | EM忽略冠词 |

## 与 Search 环境的区别

| 特性 | Search Env | NQ Search Env |
|------|------------|---------------|
| 数据集 | xhpang_search | nq_search |
| 搜索方式 | Python代码 `web_search()` | 直接标签 `<search>` |
| 检索后端 | HTTP工具服务器 | 专用检索服务器 |
| 结果标签 | `<execution_results>` | `<information>` |
| 评估方式 | xverify/OTC | EM (Exact Match) |

## 运行示例

### 独立测试
```bash
python -m roll.agentic.env.nq_search.env
```

### pytest 测试
```bash
pytest tests/agentic/nq_search_test/ -v
```

## 依赖

- Python 3.8+
- datasets
- requests
- logging

## 注意事项

1. **检索服务器**：使用前确保检索服务器在 `127.0.0.1:8100` 运行
2. **数据集格式**：需要使用转换后的 SearchEnv 格式数据集
3. **限流控制**：在生产环境中建议启用限流器
4. **EM 评估**：默认配置适用于大多数场景，如需调整可修改 `em_*` 参数

## 文件结构

```
roll/agentic/env/nq_search/
├── __init__.py          # 模块导出
├── config.py            # 配置类
├── env.py              # 环境主类
├── utils.py            # 工具函数
└── README.md           # 说明文档
```

## 数据集路径

- 训练集: `/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet`
- 测试集: `/data1/Agentic_LLM-search/datasets/nq_search_converted/test_searchenv.parquet`
- 小样本: `/data1/Agentic_LLM-search/datasets/nq_search_converted/test_sample_128_searchenv.parquet`

## 检索服务器

检索服务器应该运行在 `127.0.0.1:8100`，提供 `/retrieve` 端点：

**请求格式:**
```json
{
  "queries": ["search query"],
  "topk": 3,
  "return_scores": true
}
```

**响应格式:**
```json
{
  "result": [[
    {
      "document": {"contents": "Title\nContent..."},
      "score": 0.8816
    }
  ]]
}
```

