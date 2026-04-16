"""批量转换 nq_search 数据集

将 nq_search 数据集的所有分割（train, test 等）转换为 SearchEnv 格式
"""

import sys
from pathlib import Path

# 将项目根目录添加到 Python 解释器的搜索路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import logging
from roll.utils.nq_data_process import convert_nq_search_to_searchenv_format

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_all_nq_datasets(
    input_dir: str = "/data1/Agentic_LLM-search/datasets/nq_search",
    output_dir: str = "/data1/Agentic_LLM-search/datasets/nq_search_converted"
):
    """转换所有 nq_search 数据集文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("批量转换 nq_search 数据集")
    logger.info("="*80)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("="*80)
    
    # 查找所有 parquet 文件
    parquet_files = list(input_path.glob("*.parquet"))
    
    if not parquet_files:
        logger.error(f"在 {input_dir} 中未找到 .parquet 文件")
        return
    
    logger.info(f"\n找到 {len(parquet_files)} 个数据集文件:")
    for f in parquet_files:
        logger.info(f"  - {f.name}")
    
    # 逐个转换
    success_count = 0
    failed_files = []
    
    for input_file in parquet_files:
        output_file = output_path / f"{input_file.stem}_searchenv.parquet"
        
        logger.info("\n" + "="*80)
        logger.info(f"转换: {input_file.name}")
        logger.info("="*80)
        logger.info(f"输入: {input_file}")
        logger.info(f"输出: {output_file}")
        
        try:
            convert_nq_search_to_searchenv_format(
                input_path=str(input_file),
                output_path=str(output_file),
                max_samples=None  # 转换全部
            )
            success_count += 1
            logger.info(f"✓ {input_file.name} 转换成功")
            
        except Exception as e:
            logger.error(f"✗ {input_file.name} 转换失败: {e}")
            failed_files.append(input_file.name)
            import traceback
            logger.error(traceback.format_exc())
    
    # 总结
    logger.info("\n" + "="*80)
    logger.info("转换完成！")
    logger.info("="*80)
    logger.info(f"成功: {success_count}/{len(parquet_files)}")
    
    if failed_files:
        logger.error(f"失败: {len(failed_files)}")
        for f in failed_files:
            logger.error(f"  - {f}")
    else:
        logger.info("✅ 所有文件转换成功！")
    
    logger.info("\n转换后的文件位于:")
    logger.info(f"  {output_dir}")
    
    # 列出转换后的文件
    converted_files = list(output_path.glob("*_searchenv.parquet"))
    if converted_files:
        logger.info("\n转换后的文件:")
        for f in sorted(converted_files):
            logger.info(f"  - {f.name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量转换 nq_search 数据集")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data1/Agentic_LLM-search/datasets/nq_search",
        help="输入目录路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data1/Agentic_LLM-search/datasets/nq_search_converted",
        help="输出目录路径"
    )
    
    args = parser.parse_args()
    
    convert_all_nq_datasets(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

