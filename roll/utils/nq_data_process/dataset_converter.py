"""数据集格式转换工具

将不同格式的搜索数据集转换为 SearchEnv 统一格式
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import datasets
import pandas as pd

logger = logging.getLogger(__name__)


class SearchDatasetConverter:
    """搜索数据集格式转换器
    
    将 nq_search 等数据集格式转换为 SearchEnv 期望的格式
    """
    
    # 默认的 system prompt，适配 SearchEnv 的工具使用方式
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant with access to search tools. You can use the following Python functions:

1. web_search(keywords: str) - Search the web with given keywords
2. web_parse(link: str, query: str) - Parse content from a web page
3. parse_img(link: str, query: str) - Analyze image content

Instructions:
- Always wrap your reasoning in <think> and </think> tags
- When you need to search, write Python code inside <code> and </code> tags
- The search results will be returned in <execution_results> and </execution_results> tags
- When you have enough information, provide your final answer in <answer> and </answer> tags
- Be concise and accurate in your answer

Example workflow:
<think>I need to search for information about X.</think>
<code>
result = web_search("search query here")
print(result)
</code>

After seeing results:
<think>Based on the search results, the answer is Y.</think>
<answer>Y</answer>"""
    
    def __init__(
        self,
        source_format: str = "nq_search",
        custom_system_prompt: Optional[str] = None
    ):
        """初始化转换器
        
        Args:
            source_format: 源数据格式，目前支持 "nq_search", "xhpang_search"
            custom_system_prompt: 自定义系统提示词，如果为 None 则使用默认值
        """
        self.source_format = source_format
        self.system_prompt = custom_system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
    def convert_nq_search_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """转换单个 nq_search 数据项
        
        nq_search 格式:
            prompt: [{"role": "user", "content": "system_prompt + Question: question"}]
        
        SearchEnv 期望格式:
            prompt: [
                {"role": "system", "content": "原始system_prompt"},
                {"role": "user", "content": "question"}
            ]
        
        注意：保留原始数据集的系统提示词，只做格式拆分
        
        Args:
            item: 原始数据项
            
        Returns:
            转换后的数据项
        """
        # 获取原始 prompt（单消息）
        original_prompt = item["prompt"][0]["content"]
        
        # 提取系统提示词和问题（在 "Question: " 处分割）
        if "Question: " in original_prompt:
            parts = original_prompt.split("Question: ", 1)
            # 保留原始的系统提示词部分
            original_system = parts[0].strip()
            question = parts[1].strip()
        else:
            # 如果没有 "Question: " 标记，使用默认系统提示词
            original_system = self.system_prompt
            question = original_prompt.strip()
        
        # 构建新的 prompt 格式（使用原始系统提示词）
        new_prompt = [
            {
                "role": "system",
                "content": original_system
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        # 创建新的数据项
        converted_item = {
            "id": item["id"],
            "question": item["question"],
            "golden_answers": item["golden_answers"],
            "data_source": item.get("data_source", "nq"),
            "prompt": new_prompt,
            "ability": item.get("ability", "fact-reasoning"),
            "reward_model": item.get("reward_model", {}),
            "extra_info": item.get("extra_info", {})
        }
        
        return converted_item
    
    def convert_xhpang_search_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """转换 xhpang_search 格式（已经是正确格式，直接返回）
        
        Args:
            item: 原始数据项
            
        Returns:
            数据项（不修改）
        """
        return item
    
    def convert_dataset(
        self,
        input_path: str,
        output_path: str,
        max_samples: Optional[int] = None
    ) -> None:
        """转换整个数据集
        
        Args:
            input_path: 输入数据集路径（parquet 文件）
            output_path: 输出数据集路径（parquet 文件）
            max_samples: 最大样本数，None 表示转换全部
        """
        logger.info(f"正在加载数据集: {input_path}")
        
        # 尝试使用 datasets 库加载，如果失败则使用 pandas
        try:
            df = datasets.load_dataset("parquet", data_files=input_path)["train"]
        except Exception as e:
            logger.warning(f"datasets 库加载失败: {e}")
            logger.info("尝试使用 pandas 加载...")
            
            # 使用 pandas 读取
            import pandas as pd_lib
            pd_df = pd_lib.read_parquet(input_path)
            
            # 转换为 datasets.Dataset
            df = datasets.Dataset.from_pandas(pd_df)
            logger.info("使用 pandas 加载成功")
        
        if max_samples is not None and max_samples > 0:
            df = df.select(range(min(max_samples, len(df))))
            logger.info(f"只转换前 {len(df)} 个样本")
        
        logger.info(f"数据集大小: {len(df)}")
        
        # 转换每个数据项
        converted_items = []
        
        for i, item in enumerate(df):
            if self.source_format == "nq_search":
                converted = self.convert_nq_search_item(item)
            elif self.source_format == "xhpang_search":
                converted = self.convert_xhpang_search_item(item)
            else:
                raise ValueError(f"不支持的源格式: {self.source_format}")
            
            converted_items.append(converted)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已转换 {i + 1}/{len(df)} 个样本")
        
        logger.info(f"转换完成，共 {len(converted_items)} 个样本")
        
        # 创建新的 Dataset
        converted_df = datasets.Dataset.from_list(converted_items)
        
        # 确保输出目录存在
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为 parquet 格式
        logger.info(f"正在保存到: {output_path}")
        converted_df.to_parquet(output_path)
        
        logger.info("保存完成!")
        
        # 验证转换结果
        self._verify_conversion(output_path)
    
    def _verify_conversion(self, output_path: str) -> None:
        """验证转换结果
        
        Args:
            output_path: 输出文件路径
        """
        logger.info("验证转换结果...")
        
        df = datasets.load_dataset("parquet", data_files=output_path)["train"]
        
        # 检查第一个样本
        if len(df) > 0:
            sample = df[0]
            logger.info(f"✓ 样本数量: {len(df)}")
            logger.info(f"✓ Prompt 格式: {len(sample['prompt'])} 个消息")
            
            if len(sample['prompt']) == 2:
                logger.info(f"  - 消息 1: role={sample['prompt'][0]['role']}")
                logger.info(f"  - 消息 2: role={sample['prompt'][1]['role']}")
            
            logger.info(f"✓ 问题: {sample['question'][:50]}...")
            logger.info(f"✓ 答案: {sample['golden_answers']}")
            
        logger.info("验证完成!")


def convert_nq_search_to_searchenv_format(
    input_path: str,
    output_path: str,
    max_samples: Optional[int] = None,
    custom_system_prompt: Optional[str] = None
) -> None:
    """便捷函数：将 nq_search 转换为 SearchEnv 格式
    
    Args:
        input_path: 输入数据集路径
        output_path: 输出数据集路径
        max_samples: 最大样本数
        custom_system_prompt: 自定义系统提示词
    """
    converter = SearchDatasetConverter(
        source_format="nq_search",
        custom_system_prompt=custom_system_prompt
    )
    
    converter.convert_dataset(
        input_path=input_path,
        output_path=output_path,
        max_samples=max_samples
    )


if __name__ == "__main__":
    # 示例用法
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 默认路径
    input_file = "/data1/Agentic_LLM-search/datasets/nq_search/test_sample_128.parquet"
    output_file = "/data1/Agentic_LLM-search/datasets/nq_search_converted/test_sample_128_searchenv.parquet"
    
    # 支持命令行参数
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    
    max_samples = None
    if len(sys.argv) >= 4:
        max_samples = int(sys.argv[3])
    
    logger.info("="*80)
    logger.info("nq_search 数据集转换工具")
    logger.info("="*80)
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"最大样本数: {max_samples if max_samples else '全部'}")
    logger.info("="*80)
    
    convert_nq_search_to_searchenv_format(
        input_path=input_file,
        output_path=output_file,
        max_samples=max_samples
    )
    
    logger.info("="*80)
    logger.info("转换完成!")
    logger.info("="*80)

