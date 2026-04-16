"""NQ Search Environment Utilities

提供检索服务器交互和动作解析功能
"""

import re
import requests
from typing import Dict, Any, Callable, List
from roll.utils.logging import get_logger

logger = get_logger()


def parse_action_content(action: str, config) -> Dict[str, Any]:
    """解析动作内容，提取 think、search 和 answer 组件
    
    Args:
        action: 模型生成的动作字符串
        config: NQSearchEnvConfig 配置对象
        
    Returns:
        包含解析结果的字典
    """
    result = {
        "is_valid": False,
        "think": "",
        "search_query": "",
        "answer": "",
        "raw_action": action
    }
    
    # 提取 think 内容
    think_match = re.search(config.think_pattern, action, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()
    
    # 提取 search 查询
    search_match = re.search(config.search_pattern, action, re.DOTALL)
    if search_match:
        result["search_query"] = search_match.group(1).strip()
    
    # 提取 answer 内容
    answer_match = re.search(config.answer_pattern, action, re.DOTALL)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    
    # 动作有效性：至少包含 think、search 或 answer 之一
    result["is_valid"] = bool(result["think"] or result["search_query"] or result["answer"])
    
    return result


def call_retrieval_server(
    query: str,
    config,
    increment_search_count: Callable
) -> str:
    """调用检索服务器执行搜索
    
    Args:
        query: 搜索查询字符串
        config: NQSearchEnvConfig 配置对象
        increment_search_count: 增加搜索计数的回调函数
        
    Returns:
        格式化的搜索结果字符串
    """
    # 增加搜索计数
    increment_search_count()
    
    try:
        # 准备请求payload
        payload = {
            "queries": [query],
            "topk": config.retrieval_topk,
            "return_scores": config.return_scores
        }
        
        logger.debug(f"Calling retrieval server with query: {query}")
        
        # 发送POST请求到检索服务器
        response = requests.post(
            config.retrieval_server_url,
            json=payload,
            timeout=config.retrieval_timeout
        )
        
        # 检查响应状态
        if response.status_code != 200:
            error_msg = f"Retrieval server error: HTTP {response.status_code}"
            logger.error(error_msg)
            return error_msg
        
        # 解析响应
        result = response.json()
        
        # 检查结果格式
        if "result" not in result or len(result["result"]) == 0:
            return "No search results found."
        
        # 格式化搜索结果
        formatted_result = format_search_results(result["result"][0])
        
        logger.debug(f"Retrieved {len(result['result'][0])} documents")
        
        return formatted_result
        
    except requests.exceptions.Timeout:
        error_msg = "Retrieval server timeout"
        logger.error(error_msg)
        return error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to retrieval server"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Retrieval error: {str(e)}"
        logger.error(error_msg)
        return error_msg


def format_search_results(results: List[Dict]) -> str:
    """格式化搜索结果为可读文本
    
    Args:
        results: 检索服务器返回的结果列表
        
    Returns:
        格式化的结果字符串
    """
    if not results:
        return "No results found."
    
    formatted_lines = []
    
    for idx, doc_item in enumerate(results):
        # 提取文档内容
        if 'document' in doc_item:
            content = doc_item['document'].get('contents', '')
            
            # 分割标题和正文
            lines = content.split('\n')
            title = lines[0] if lines else "Untitled"
            text = '\n'.join(lines[1:]) if len(lines) > 1 else content
            
            # 构建文档条目
            doc_str = f"Doc {idx + 1}(Title: {title}) {text}"
            
            # 添加相似度分数（如果有）
            if 'score' in doc_item:
                doc_str += f" [Score: {doc_item['score']:.4f}]"
            
            formatted_lines.append(doc_str)
    
    return '\n'.join(formatted_lines)


def evaluate_answer_em(
    predicted_answer: str,
    golden_answers: List[str],
    ignore_case: bool = True,
    ignore_punctuation: bool = True,
    ignore_articles: bool = True
) -> tuple[bool, float]:
    """使用精确匹配（Exact Match）评估答案
    
    Args:
        predicted_answer: 模型预测的答案
        golden_answers: 标准答案列表
        ignore_case: 是否忽略大小写
        ignore_punctuation: 是否忽略标点符号
        ignore_articles: 是否忽略冠词
        
    Returns:
        (是否正确, 分数) 元组
    """
    import string
    
    def normalize_text(text: str) -> str:
        """标准化文本"""
        # 转小写
        if ignore_case:
            text = text.lower()
        
        # 移除标点符号
        if ignore_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 移除冠词
        if ignore_articles:
            for article in [' a ', ' an ', ' the ']:
                text = text.replace(article, ' ')
        
        # 标准化空白字符
        text = ' '.join(text.split())
        
        return text.strip()
    
    # 标准化预测答案
    pred_normalized = normalize_text(predicted_answer)
    
    # 检查是否与任一标准答案匹配
    for golden in golden_answers:
        golden_normalized = normalize_text(golden)
        
        # 精确匹配
        if pred_normalized == golden_normalized:
            return True, 1.0
        
        # 部分匹配（包含关系）
        if pred_normalized in golden_normalized or golden_normalized in pred_normalized:
            return True, 1.0
    
    return False, 0.0

