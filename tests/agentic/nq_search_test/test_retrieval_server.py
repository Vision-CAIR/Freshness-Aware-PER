"""测试检索服务器功能

测试位于 127.0.0.1:8100 的检索服务器是否正常工作
"""

import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
import requests
import json
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrievalServerTester:
    """检索服务器测试类"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8100"):
        """初始化测试器
        
        Args:
            server_url: 检索服务器URL
        """
        self.server_url = server_url
        self.retrieve_endpoint = f"{server_url}/retrieve"
    
    def test_server_alive(self) -> bool:
        """测试服务器是否存活"""
        logger.info("="*80)
        logger.info("测试1: 检查服务器是否存活")
        logger.info("="*80)
        
        try:
            # 尝试连接服务器
            response = requests.get(self.server_url, timeout=5)
            logger.info(f"✓ 服务器响应状态码: {response.status_code}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ 无法连接到服务器: {e}")
            return False
    
    def test_single_query(self) -> bool:
        """测试单个查询"""
        logger.info("\n" + "="*80)
        logger.info("测试2: 单个查询测试")
        logger.info("="*80)
        
        # 准备查询
        query = "who got the first nobel prize in physics?"
        payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
        
        logger.info(f"查询: {query}")
        logger.info(f"请求 topk: {payload['topk']}")
        
        try:
            # 发送请求
            start_time = time.time()
            response = requests.post(
                self.retrieve_endpoint, 
                json=payload, 
                timeout=30
            )
            elapsed_time = time.time() - start_time
            
            # 检查响应状态
            if response.status_code != 200:
                logger.error(f"✗ 请求失败，状态码: {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return False
            
            # 解析结果
            result = response.json()
            logger.info(f"✓ 请求成功，耗时: {elapsed_time:.3f}秒")
            
            # 显示结果
            if "result" in result and len(result["result"]) > 0:
                logger.info(f"✓ 返回结果数量: {len(result['result'])}")
                
                # 显示第一个查询的结果
                query_results = result["result"][0]
                logger.info(f"✓ 检索到的文档数: {len(query_results)}")
                
                # 显示每个文档的详细信息
                for idx, doc_item in enumerate(query_results):
                    logger.info(f"\n  文档 {idx+1}:")
                    if 'document' in doc_item:
                        doc_content = doc_item['document'].get('contents', '')
                        lines = doc_content.split('\n')
                        title = lines[0] if lines else "无标题"
                        logger.info(f"    标题: {title}")
                        logger.info(f"    内容长度: {len(doc_content)} 字符")
                        logger.info(f"    内容预览: {doc_content[:200]}...")
                    
                    if 'score' in doc_item:
                        logger.info(f"    相似度分数: {doc_item['score']:.4f}")
                
                return True
            else:
                logger.error("✗ 返回结果为空")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("✗ 请求超时")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ 请求失败: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"✗ 解析响应JSON失败: {e}")
            logger.error(f"响应内容: {response.text[:500]}")
            return False
    
    def test_batch_queries(self) -> bool:
        """测试批量查询"""
        logger.info("\n" + "="*80)
        logger.info("测试3: 批量查询测试")
        logger.info("="*80)
        
        # 准备多个查询
        queries = [
            "who got the first nobel prize in physics?",
            "when is the next deadpool movie being released?",
            "which mode is used for short wave broadcast service?"
        ]
        
        payload = {
            "queries": queries,
            "topk": 3,
            "return_scores": True
        }
        
        logger.info(f"批量查询数量: {len(queries)}")
        for i, q in enumerate(queries):
            logger.info(f"  查询 {i+1}: {q}")
        
        try:
            # 发送请求
            start_time = time.time()
            response = requests.post(
                self.retrieve_endpoint, 
                json=payload, 
                timeout=30
            )
            elapsed_time = time.time() - start_time
            
            # 检查响应状态
            if response.status_code != 200:
                logger.error(f"✗ 请求失败，状态码: {response.status_code}")
                return False
            
            # 解析结果
            result = response.json()
            logger.info(f"✓ 批量请求成功，耗时: {elapsed_time:.3f}秒")
            
            # 验证结果数量
            if "result" in result:
                logger.info(f"✓ 返回结果集数量: {len(result['result'])}")
                
                # 验证每个查询都有结果
                if len(result["result"]) == len(queries):
                    logger.info("✓ 每个查询都返回了结果")
                    
                    # 统计信息
                    total_docs = sum(len(r) for r in result["result"])
                    avg_docs = total_docs / len(queries)
                    logger.info(f"✓ 总检索文档数: {total_docs}")
                    logger.info(f"✓ 平均每个查询返回: {avg_docs:.1f} 个文档")
                    
                    return True
                else:
                    logger.error(f"✗ 结果数量不匹配: 期望 {len(queries)}, 实际 {len(result['result'])}")
                    return False
            else:
                logger.error("✗ 响应中没有 'result' 字段")
                return False
                
        except Exception as e:
            logger.error(f"✗ 批量查询失败: {e}")
            return False
    
    def test_different_topk(self) -> bool:
        """测试不同的topk参数"""
        logger.info("\n" + "="*80)
        logger.info("测试4: 不同topk参数测试")
        logger.info("="*80)
        
        query = "nobel prize physics"
        topk_values = [1, 3, 5]
        
        for topk in topk_values:
            payload = {
                "queries": [query],
                "topk": topk,
                "return_scores": True
            }
            
            try:
                response = requests.post(
                    self.retrieve_endpoint, 
                    json=payload, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    doc_count = len(result["result"][0])
                    logger.info(f"✓ topk={topk}: 返回 {doc_count} 个文档")
                    
                    if doc_count != topk:
                        logger.warning(f"  ⚠ 实际返回数量与topk不符")
                else:
                    logger.error(f"✗ topk={topk}: 请求失败")
                    return False
                    
            except Exception as e:
                logger.error(f"✗ topk={topk}: 测试失败: {e}")
                return False
        
        return True
    
    def test_search_with_searchenv(self) -> bool:
        """测试与SearchEnv集成使用"""
        logger.info("\n" + "="*80)
        logger.info("测试5: SearchEnv集成测试")
        logger.info("="*80)
        
        try:
            from roll.agentic.env.search import SearchEnv, SearchEnvConfig
            
            # 创建配置，使用远程搜索服务
            config = SearchEnvConfig(
                dataset_path="/data1/Agentic_LLM-search/datasets/nq_search_converted/test_sample_128_searchenv.parquet",
                max_instances=5,
                use_mock_api=False,  # 使用真实搜索
                use_remote_service=True,  # 使用远程服务
                remote_service_url=self.server_url,
                disable_limiter=True,
                max_steps=5,
                max_search_calls=3
            )
            
            logger.info("创建SearchEnv环境...")
            env = SearchEnv(config)
            logger.info(f"✓ 环境创建成功，加载了 {len(env.data)} 个样本")
            
            # 测试一个完整的交互流程
            logger.info("\n执行测试交互...")
            obs, info = env.reset(seed=0)
            logger.info(f"✓ Reset成功")
            
            # 执行搜索动作
            search_action = '''<think>
需要搜索相关信息
</think>

<code>
result = web_search("nobel prize physics winner")
print(result)
</code>'''
            
            obs, reward, terminated, truncated, info = env.step(search_action)
            
            if info["action_is_valid"] and info["search_calls"] > 0:
                logger.info(f"✓ 搜索执行成功")
                logger.info(f"  - 搜索次数: {info['search_calls']}")
                logger.info(f"  - 返回结果长度: {len(obs)}")
                logger.info(f"  - 结果预览: {obs[:200]}...")
                return True
            else:
                logger.error(f"✗ 搜索执行失败")
                return False
                
        except Exception as e:
            logger.error(f"✗ SearchEnv集成测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("\n" + "="*80)
        logger.info("开始检索服务器完整测试")
        logger.info("="*80)
        logger.info(f"服务器URL: {self.server_url}")
        logger.info("="*80 + "\n")
        
        tests = [
            ("服务器存活检查", self.test_server_alive),
            ("单个查询测试", self.test_single_query),
            ("批量查询测试", self.test_batch_queries),
            ("不同topk参数测试", self.test_different_topk),
            ("SearchEnv集成测试", self.test_search_with_searchenv),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                logger.error(f"测试 '{test_name}' 执行时出错: {e}")
                results.append((test_name, False))
        
        # 输出总结
        logger.info("\n" + "="*80)
        logger.info("测试总结")
        logger.info("="*80)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"{status} - {test_name}")
        
        logger.info("="*80)
        logger.info(f"测试结果: {passed}/{total} 通过")
        logger.info("="*80)
        
        return passed == total


def main():
    """主函数"""
    # 创建测试器
    tester = RetrievalServerTester(server_url="http://127.0.0.1:8100")
    
    # 运行所有测试
    success = tester.run_all_tests()
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

