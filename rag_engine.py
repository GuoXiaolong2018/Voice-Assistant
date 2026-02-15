"""
RAG接口抽象模块
支持多种RAG实现：Bailian RAG、Dify RAG等
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RAGInterface(ABC):
    """RAG接口抽象基类"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化RAG服务
        
        参数:
            config: RAG配置字典
            
        返回:
            是否初始化成功
        """
        pass
    
    @abstractmethod
    def query(self, message: str, return_timing: bool = False) -> Any:
        """
        查询RAG服务
        
        参数:
            message: 用户消息
            return_timing: 是否返回耗时信息
            
        返回:
            如果 return_timing=False: 返回答案字符串
            如果 return_timing=True: 返回字典 {'answer': str, 'timing': dict}
        """
        pass


class BailianRAG(RAGInterface):
    """Bailian RAG实现"""
    
    def __init__(self):
        self._assistant_instance = None
        self._thread_instance = None
        self._initialized = False
        self._index_id = None
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化Bailian RAG"""
        if self._initialized and self._assistant_instance and self._thread_instance:
            logger.info("Bailian RAG 已初始化，跳过重复初始化")
            return True
        
        try:
            from dashscope import Assistants, Threads
            
            index_id = config.get("index_id")
            if not index_id or not index_id.strip():
                raise ValueError("知识库索引 ID 不能为空")
            
            self._index_id = index_id
            logger.info(f"正在初始化 Bailian RAG Assistant，使用知识库索引 ID: {index_id}")
            
            assistant_id = self._create_assistant(config)
            self._assistant_instance = Assistants.get(assistant_id)
            self._thread_instance = Threads.create()
            self._initialized = True
            
            logger.info(f"✓ Bailian RAG 初始化成功！Assistant ID: {assistant_id}, Thread ID: {self._thread_instance.id}")
            return True
        except Exception as e:
            logger.error(f"❌ Bailian RAG 初始化失败：{str(e)}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            return False
    
    def _create_assistant(self, config: Dict[str, Any]) -> str:
        """创建Assistant"""
        from dashscope import Assistants
        
        assistant = Assistants.create(
            model=config.get("model", "qwen-plus"),
            name=config.get("name", "中石化资料库"),
            description=config.get("description", "一个包含中石化常见知识的资料库助手"),
            instructions=config.get("instructions", ""),
            tools=[
                {
                    "type": "rag",
                    "prompt_ra": {
                        "pipeline_id": [self._index_id],
                        "multiknowledge_rerank_top_n": config.get("multiknowledge_rerank_top_n", 10),
                        "rerank_top_n": config.get("rerank_top_n", 5),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query_word": {
                                    "type": "str",
                                    "value": "${document1}"
                                }
                            }
                        }
                    }
                },
            ]
        )
        return assistant.id
    
    def query(self, message: str, return_timing: bool = False) -> Any:
        """查询Bailian RAG"""
        if not self._initialized or not self._assistant_instance or not self._thread_instance:
            error_msg = "❌ Bailian RAG 未初始化，无法回答问题"
            if return_timing:
                return {
                    'answer': error_msg,
                    'timing': {
                        'total_time': 0,
                        'time_to_first_token': 0,
                        'generation_time': 0,
                        'first_token_time': False,
                        'error': True
                    }
                }
            return error_msg
        
        import time
        from dashscope import Messages, Runs
        
        start_time = time.time()
        
        try:
            Messages.create(thread_id=self._thread_instance.id, content=message)
            message_sent_time = time.time()
            
            run_iterator = Runs.create(
                thread_id=self._thread_instance.id,
                assistant_id=self._assistant_instance.id,
                stream=True
            )
            
            full_response = ""
            first_token_time = None
            
            try:
                for event, data in run_iterator:
                    if first_token_time is None and event == 'thread.message.delta':
                        first_token_time = time.time()
                    
                    if event == 'thread.message.delta':
                        try:
                            full_response += data.delta.content.text.value
                        except AttributeError:
                            pass
                    
                    if event == 'thread.run.step.completed':
                        break
            except KeyboardInterrupt:
                run_iterator.close()
                raise
            except Exception as e:
                run_iterator.close()
                raise
            
            end_time = time.time()
            total_time = end_time - start_time
            time_to_first_token = (first_token_time - message_sent_time) if first_token_time else 0
            generation_time = end_time - (first_token_time if first_token_time else message_sent_time)
            
            timing_info = {
                'total_time': total_time,
                'time_to_first_token': time_to_first_token,
                'generation_time': generation_time,
                'first_token_time': first_token_time is not None
            }
            
            if return_timing:
                return {
                    'answer': full_response,
                    'timing': timing_info
                }
            else:
                return full_response
                
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            error_msg = f"❌ 发生错误：{str(e)}"
            
            if return_timing:
                return {
                    'answer': error_msg,
                    'timing': {
                        'total_time': total_time,
                        'time_to_first_token': 0,
                        'generation_time': 0,
                        'first_token_time': False,
                        'error': True
                    }
                }
            else:
                return error_msg


class DifyRAG(RAGInterface):
    """Dify RAG实现"""
    
    def __init__(self):
        self._initialized = False
        self._config = None
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化Dify RAG"""
        try:
            self._config = config
            self._initialized = True
            logger.info("✓ Dify RAG 初始化成功")
            return True
        except Exception as e:
            logger.error(f"❌ Dify RAG 初始化失败：{str(e)}")
            self._initialized = False
            return False
    
    def query(self, message: str, return_timing: bool = False) -> Any:
        """查询Dify RAG"""
        if not self._initialized:
            error_msg = "❌ Dify RAG 未初始化，无法回答问题"
            if return_timing:
                return {
                    'answer': error_msg,
                    'timing': {
                        'total_time': 0,
                        'time_to_first_token': 0,
                        'generation_time': 0,
                        'first_token_time': False,
                        'error': True
                    }
                }
            return error_msg
        
        import time
        import requests
        import json
        import os
        
        start_time = time.time()
        
        try:
            # 保存并临时清除代理环境变量
            saved_proxy_vars = {}
            proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 
                          'all_proxy', 'ALL_PROXY', 'socks_proxy', 'SOCKS_PROXY']
            for var in proxy_vars:
                if var in os.environ:
                    saved_proxy_vars[var] = os.environ[var]
                    del os.environ[var]
            
            try:
                payload = {
                    "query": message,
                    "inputs": {},
                    "response_mode": self._config.get("response_mode", "streaming"),
                    "user": self._config.get("user", "abc-123")
                }
                
                headers = {
                    "Authorization": f"Bearer {self._config.get('api_key')}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                }
                
                stream_mode = payload["response_mode"] == "streaming"
                response = requests.post(
                    self._config.get("api_url"),
                    json=payload,
                    headers=headers,
                    timeout=30,
                    stream=stream_mode,
                    proxies={'http': None, 'https': None}
                )
                
                if response.status_code == 200:
                    if stream_mode:
                        result_text = ""
                        first_token_time = None
                        
                        for line in response.iter_lines(decode_unicode=True):
                            if not line:
                                continue
                            
                            if not line.startswith("data: "):
                                continue
                            
                            data_str = line[6:].strip()
                            if not data_str or data_str == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                event_type = data.get("event", "")
                                
                                if first_token_time is None and event_type == "message":
                                    first_token_time = time.time()
                                
                                if event_type == "message" and "answer" in data:
                                    answer = data["answer"]
                                    if isinstance(answer, str):
                                        result_text += answer
                                    elif isinstance(answer, dict):
                                        if "text" in answer:
                                            result_text += answer["text"]
                                        elif "content" in answer:
                                            result_text += answer["content"]
                                
                                elif event_type == "workflow_finished":
                                    workflow_data = data.get("data", {})
                                    if workflow_data:
                                        outputs = workflow_data.get("outputs", {})
                                        if outputs:
                                            for key in ["answer", "text", "content", "output", "result"]:
                                                if key in outputs:
                                                    result_text = str(outputs[key])
                                                    break
                            except json.JSONDecodeError:
                                continue
                        
                        end_time = time.time()
                        total_time = end_time - start_time
                        time_to_first_token = (first_token_time - start_time) if first_token_time else 0
                        generation_time = end_time - (first_token_time if first_token_time else start_time)
                        
                        if return_timing:
                            return {
                                'answer': result_text.strip(),
                                'timing': {
                                    'total_time': total_time,
                                    'time_to_first_token': time_to_first_token,
                                    'generation_time': generation_time,
                                    'first_token_time': first_token_time is not None
                                }
                            }
                        else:
                            return result_text.strip()
                    else:
                        result_data = response.json() or {}
                        result_text = str(result_data.get("answer", ""))
                        end_time = time.time()
                        total_time = end_time - start_time
                        
                        if return_timing:
                            return {
                                'answer': result_text.strip(),
                                'timing': {
                                    'total_time': total_time,
                                    'time_to_first_token': 0,
                                    'generation_time': total_time,
                                    'first_token_time': False
                                }
                            }
                        else:
                            return result_text.strip()
                else:
                    error_msg = f"Dify API 调用失败，状态码: {response.status_code}"
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    if return_timing:
                        return {
                            'answer': error_msg,
                            'timing': {
                                'total_time': total_time,
                                'time_to_first_token': 0,
                                'generation_time': 0,
                                'first_token_time': False,
                                'error': True
                            }
                        }
                    else:
                        return error_msg
            finally:
                # 恢复代理环境变量
                for var, value in saved_proxy_vars.items():
                    os.environ[var] = value
                    
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            error_msg = f"❌ 发生错误：{str(e)}"
            
            if return_timing:
                return {
                    'answer': error_msg,
                    'timing': {
                        'total_time': total_time,
                        'time_to_first_token': 0,
                        'generation_time': 0,
                        'first_token_time': False,
                        'error': True
                    }
                }
            else:
                return error_msg


def create_rag_instance(rag_type: str) -> Optional[RAGInterface]:
    """
    创建RAG实例
    
    参数:
        rag_type: RAG类型 ("bailian" 或 "dify")
        
    返回:
        RAG实例，如果类型不支持则返回None
    """
    if rag_type.lower() == "bailian":
        return BailianRAG()
    elif rag_type.lower() == "dify":
        return DifyRAG()
    else:
        logger.error(f"不支持的RAG类型: {rag_type}")
        return None

