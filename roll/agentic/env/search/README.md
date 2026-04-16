# Search Environment

基于搜索数据集的强化学习环境，支持模型通过搜索引擎进行推理和问答。

## 功能特性

### 1. 数据集支持
- 读取 xhpang_search 数据集 (Parquet 格式)
- 支持问答任务，包含系统提示词和用户问题
- 自动解析数据集中的 prompt 结构

### 2. 搜索工具
支持三种搜索工具：
- `web_search(keywords)`: 网络搜索
- `web_parse(link, query)`: 网页内容解析  
- `parse_img(link, query)`: 图片内容分析

### 3. 执行模式
- **Mock 模式**: 使用模拟搜索结果，适合测试和开发
- **HTTP 模式**: 通过远程 HTTP 服务执行搜索，适合生产环境
- **API 模式**: 直接调用搜索 API（可扩展）

### 4. 限流控制
- 使用 `env_action_limiter.py` 控制并发请求数
- 支持最大搜索次数限制
- 支持最大步数限制

## 使用方法

### 基本配置

```python
from roll.agentic.env.search import SearchEnv, SearchEnvConfig

config = SearchEnvConfig(
    dataset_path="/path/to/xhpang_search/train.parquet",
    max_instances=1000,
    use_mock_api=True,  # 测试模式
    use_remote_service=False,  # 不使用远程服务
    max_steps=10,
    max_search_calls=5
)

env = SearchEnv(config)
```

### 远程服务配置

```python
config = SearchEnvConfig(
    dataset_path="/path/to/xhpang_search/train.parquet",
    use_mock_api=False,
    use_remote_service=True,
    remote_service_url="http://172.26.104.240:30002",
    remote_service_timeout=30
)
```

### Trajectory 格式

环境遵循以下交互模式：

```
observation0: 起始prompt + 问题
action0: <think>推理过程</think> + <code>搜索代码</code>
observation1: <execution_results>搜索结果</execution_results>
action1: <think>分析结果</think> + <answer>最终答案</answer>
observation2: 最终反馈 + 奖励
```

### 示例代码

```python
# 重置环境
obs, info = env.reset(seed=42)

# 第一步：搜索
action = '''<think>
我需要搜索关于帕丁顿附近铁路调车场的信息。
</think>

<code>
result = web_search("Great Western Railway Paddington railway yard slip switch")
print(result)
</code>'''

obs, reward, terminated, truncated, info = env.step(action)

# 第二步：回答
action = '''<think>
根据搜索结果，答案是 Old Oak Common Yard。
</think>

<answer>Old Oak Common Yard</answer>'''

obs, reward, terminated, truncated, info = env.step(action)
```

## 测试

### 运行基本测试
```bash
cd /mnt/chensiheng/weiyu/xueban_v2/ROLL
python -m tests.agentic.env.test_search
```

### 运行 HTTP 工具测试
```bash
python -m tests.agentic.env.test_search_http_tools
```

### 运行环境演示
```bash
python -m roll.agentic.env.search.env
```

## 文件结构

```
roll/agentic/env/search/
├── __init__.py          # 模块导出
├── config.py            # 配置类
├── env.py              # 主环境类
├── utils.py            # 工具函数
└── README.md           # 说明文档

tests/agentic/env/
├── test_search.py              # 基本测试
└── test_search_http_tools.py   # HTTP 工具测试
```

## 配置参数

### SearchEnvConfig 主要参数

- `dataset_path`: 数据集路径
- `max_instances`: 最大实例数
- `max_steps`: 最大步数
- `max_search_calls`: 最大搜索次数
- `use_mock_api`: 是否使用模拟 API
- `use_remote_service`: 是否使用远程服务
- `remote_service_url`: 远程服务 URL
- `correct_answer_reward`: 正确答案奖励
- `step_penalty`: 步数惩罚
- `invalid_action_penalty`: 无效动作惩罚

## 奖励机制

- 正确答案：+1.0
- 每步惩罚：-0.1
- 无效动作：-0.2
- 超时惩罚：额外 -0.2

## 注意事项

1. 确保数据集路径正确
2. 远程服务需要先启动工具服务器
3. Mock 模式适合开发和测试
4. 生产环境建议使用 HTTP 模式
5. 注意配置限流参数避免过载
