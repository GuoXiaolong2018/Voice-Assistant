#!/usr/bin/env python3
"""
重构后的语音助手主程序
所有配置通过YAML文件进行管理
"""

import sys
import os
import time
import logging
import signal
import re
import threading
import json
import tempfile
import io
import wave
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from queue import Queue
from collections import deque
from datetime import datetime

# 在导入其他模块之前设置环境变量，屏蔽 ALSA 错误
os.environ.update({
    "LIBASOUND_DEBUG": "0",
    "JACK_NO_START_SERVER": "1",
    "PIPEWIRE_DISABLE": "1",
    "PULSE_SERVER": "unix:/dev/null",
    "CT2_LOG_LEVEL": "error",
    "ALSA_CARD": "0",
})

# 禁用代理
def disable_proxy():
    """禁用所有代理设置"""
    proxy_vars = [
        'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
        'ALL_PROXY', 'all_proxy',
        'SOCKS_PROXY', 'socks_proxy', 'SOCKS5_PROXY', 'socks5_proxy',
        'SOCKS4_PROXY', 'socks4_proxy',
        'NO_PROXY', 'no_proxy'
    ]
    for var in proxy_vars:
        if var in ['NO_PROXY', 'no_proxy']:
            os.environ[var] = '*'
        else:
            os.environ.pop(var, None)
    env_keys_to_remove = []
    for key in os.environ.keys():
        if 'proxy' in key.lower() and key not in ['NO_PROXY', 'no_proxy']:
            env_keys_to_remove.append(key)
    for key in env_keys_to_remove:
        os.environ.pop(key, None)

disable_proxy()

# 添加路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入配置和接口模块
from config import load_config, get_config_value
from rag_engine import create_rag_instance
from communication import create_communication_instance
from intent_detector import IntentHandler
from intent_manager import IntentManager

# 导入各个接口
import importlib.util

# 导入 wakeup_detector
listener_spec = importlib.util.spec_from_file_location(
    "wakeup_detector", current_dir / "wakeup_detector.py"
)
listener_module = importlib.util.module_from_spec(listener_spec)
listener_spec.loader.exec_module(listener_module)

# 导入 asr_engine
stt_spec = importlib.util.spec_from_file_location(
    "asr_engine", current_dir / "asr_engine.py"
)
stt_module = importlib.util.module_from_spec(stt_spec)
stt_spec.loader.exec_module(stt_module)
transcribe_audio_funasr = stt_module.transcribe_audio_funasr
transcribe_audio_funasr_preloaded = stt_module.transcribe_audio_funasr_preloaded
preload_funasr_model = stt_module.preload_funasr_model
RealtimeASREngine = stt_module.RealtimeASREngine

# 导入其他模块
from tts_engine import TTSPlayer
from openai import OpenAI
import pyaudio

# 导入 FunASR（用于模型预加载，现在主要在 asr_engine 中使用）
try:
    from funasr import AutoModel
    HAS_FUNASR = True
except ImportError:
    HAS_FUNASR = False

# 导入 dashscope 并设置 API key
import dashscope
import requests
requests.Session().proxies = {}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceAssistant:
    """语音助手主类"""
    
    def __init__(self, config_path: str = None):
        """初始化语音助手"""
        # 加载配置
        self.config = load_config(config_path)
        
        # 设置日志级别
        log_level = get_config_value(self.config, "logging.level", "INFO")
        logging.getLogger().setLevel(getattr(logging, log_level))
        
        # 设置环境变量
        self._setup_environment()
        
        # 初始化组件
        self._init_components()
        
        # 状态管理
        self.enable_audio_playback = get_config_value(self.config, "audio.enable_playback", True)
        self.enable_audio_lock = threading.Lock()
        self.wozai_audio_thread = None
        self.wozai_audio_lock = threading.Lock()
        
        # 实时识别相关（现在由 RealtimeASREngine 管理）
        
        logger.info("语音助手初始化完成")
    
    def _setup_environment(self):
        """设置环境变量"""
        api_keys = self.config.get("api_keys", {})
        
        # 设置 Bailian 相关环境变量
        required_vars = {
            'ALIBABA_CLOUD_ACCESS_KEY_ID': api_keys.get("alibaba_cloud_access_key_id", ""),
            'ALIBABA_CLOUD_ACCESS_KEY_SECRET': api_keys.get("alibaba_cloud_access_key_secret", ""),
            'DASHSCOPE_API_KEY': api_keys.get("dashscope_api_key", ""),
            'WORKSPACE_ID': api_keys.get("workspace_id", "")
        }
        for var, value in required_vars.items():
            if value and not os.environ.get(var):
                os.environ[var] = value
        
        # 设置 dashscope API key
        dashscope_api_key = api_keys.get("dashscope_api_key", "")
        if dashscope_api_key:
            dashscope.api_key = dashscope_api_key
    
    def _init_components(self):
        """初始化各个组件"""
        # 初始化千问客户端
        qwen_config = self.config.get("qwen", {})
        api_keys = self.config.get("api_keys", {})
        self.qwen_client = OpenAI(
            api_key=api_keys.get("dashscope_api_key", ""),
            base_url=qwen_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            http_client=None
        )
        
        # 初始化TTS播放器
        tts_config = self.config.get("tts", {})
        self.tts_player = TTSPlayer(
            engine=tts_config.get("engine", "dashscope"),
            dashscope_voice=tts_config.get("dashscope_voice", "zhitian"),
            keep_files=tts_config.get("keep_files", False),
            auto_play=tts_config.get("auto_play", True),
            dashscope_api_key=api_keys.get("dashscope_api_key", "")
        )
        
        # 初始化RAG
        rag_config = self.config.get("rag", {})
        rag_type = rag_config.get("type", "bailian")
        self.rag_instance = create_rag_instance(rag_type)
        if self.rag_instance:
            if rag_type == "bailian":
                rag_type_config = rag_config.get("bailian", {})
            elif rag_type == "dify":
                rag_type_config = rag_config.get("dify", {})
            else:
                rag_type_config = {}
            
            if not self.rag_instance.initialize(rag_type_config):
                logger.warning("RAG初始化失败，知识库问答功能将不可用")
        
        # 初始化通信接口
        comm_config = self.config.get("communication", {})
        comm_type = comm_config.get("type", "websocket")
        if comm_type == "websocket":
            ws_config = comm_config.get("websocket", {})
            host = ws_config.get("host", "localhost")
            port = ws_config.get("port", 2626)
        else:
            tcp_config = comm_config.get("tcp", {})
            host = tcp_config.get("host", "127.0.0.1")
            port = tcp_config.get("port", 8888)
        
        self.comm_interface = create_communication_instance(comm_type, host, port)
        if self.comm_interface:
            self.comm_interface.set_control_callback(self._on_audio_control)
        
        # 初始化意图管理器（从主配置文件加载意图配置）
        intent_tts_config = {
            "dashscope_voice": tts_config.get("dashscope_voice", "zhitian"),
            "dashscope_api_key": api_keys.get("dashscope_api_key", "")
        }
        intent_manager = None
        
        try:
            # 从主配置文件加载意图管理器
            logger.info("从主配置文件加载意图配置")
            intent_manager = IntentManager(
                main_config_path=str(current_dir / "voice_assistant_config.yaml"),
                tts_config=intent_tts_config
            )
            logger.info("✓ 意图管理器初始化成功")
            
            # 自动为所有有文本的意图生成音频文件
            logger.info("正在检查并生成音频文件...")
            for role_name, role in intent_manager.roles.items():
                for tag, intent in role.intents.items():
                    if intent.action.text and intent.action.type == "tts":
                        audio_file = intent_manager._ensure_audio_file(role_name, intent)
                        if audio_file:
                            intent_manager._update_map_file(role_name, intent, audio_file)
                            logger.debug(f"角色 {role_name} 意图 {tag} 音频文件: {audio_file}")
            
            # 更新主配置文件（移除intent_dict和audio_files）
            for role_name in intent_manager.roles.keys():
                intent_manager._update_main_config_role(role_name)
            
            # 保存意图配置到主配置文件
            intent_manager.save_intents()
            
            logger.info("✓ 音频文件检查完成，map.jsonl已更新")
        except Exception as e:
            logger.error(f"意图管理器初始化失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 初始化意图处理器（传入意图管理器）
        self.intent_handler = IntentHandler(self.config, self.qwen_client, intent_manager=intent_manager)
        
        # 获取当前场景的唤醒词（从意图管理器或配置中）
        current_scenario = self.config.get("intent", {}).get("scenario", "aipaoge")
        wakeup_word = "爱跑哥"  # 默认值
        
        if intent_manager and current_scenario in intent_manager.roles:
            # 从意图管理器获取当前场景的唤醒词
            wakeup_word = intent_manager.roles[current_scenario].wakeup_word
        else:
            # 如果没有意图管理器，从basic配置中获取（向后兼容）
            basic_config = self.config.get("basic", {})
            wakeup_word = basic_config.get("wakeup_word", "爱跑哥")
        
        # 初始化唤醒管理器
        basic_config = self.config.get("basic", {})
        asr_config = self.config.get("asr", {})
        model_paths = self.config.get("paths", {}).get("funasr_model_paths", [])
        model_path = None
        for path in model_paths:
            full_path = Path(current_dir) / path
            if full_path.exists():
                model_path = str(full_path)
                break
        
        if not model_path:
            model_path = asr_config.get("model_path", "")
            if model_path:
                model_path = str(Path(current_dir) / model_path)
        
        self.wakeup_manager = ParallelWakeupManager(
            wakeup_word=wakeup_word,
            threshold=basic_config.get("wakeup_threshold", 1.0),
            record_duration=basic_config.get("wakeup_record_duration", 1.4),
            overlap_duration=basic_config.get("wakeup_overlap_duration", 0.7),
            model_path=model_path,
            config=self.config
        )
        
        # 初始化实时识别
        self._init_realtime_recognition()
    
    def _init_realtime_recognition(self):
        """初始化实时语音识别"""
        self.realtime_asr_engine = RealtimeASREngine(self.config)
        if not self.realtime_asr_engine.initialize():
            logger.error("实时识别引擎初始化失败")
            self.realtime_asr_engine = None
    
    def _on_audio_control(self, enable: bool):
        """音频控制回调"""
        with self.enable_audio_lock:
            self.enable_audio_playback = enable
        status = "启用" if enable else "禁用"
        logger.info(f"音频播放已{status}")
    
    def preload_models(self):
        """预加载模型"""
        # 预加载FunASR模型
        asr_config = self.config.get("asr", {})
        model_paths = self.config.get("paths", {}).get("funasr_model_paths", [])
        model_path = None
        for path in model_paths:
            full_path = Path(current_dir) / path
            if full_path.exists():
                model_path = str(full_path)
                break
        
        if not model_path:
            model_path = asr_config.get("model_path", "")
            if model_path:
                model_path = str(Path(current_dir) / model_path)
        
        if model_path and HAS_FUNASR:
            logger.info("正在预加载 FunASR 模型...")
            if preload_funasr_model(model_path, self.config):
                logger.info("✓ FunASR 模型预加载成功")
            else:
                logger.warning("⚠ FunASR 模型预加载失败")
        
        # 初始化RAG
        if self.rag_instance:
            logger.info("正在初始化 RAG...")
            rag_config = self.config.get("rag", {})
            rag_type = rag_config.get("type", "bailian")
            if rag_type == "bailian":
                rag_type_config = rag_config.get("bailian", {})
            elif rag_type == "dify":
                rag_type_config = rag_config.get("dify", {})
            else:
                rag_type_config = {}
            
            if self.rag_instance.initialize(rag_type_config):
                logger.info("✓ RAG 初始化成功")
            else:
                logger.warning("⚠ RAG 初始化失败")
    
    
    def run(self):
        """运行主循环"""
        logger.info("=" * 60)
        logger.info("语音助手启动")
        logger.info(f"唤醒词: {self.config.get('basic', {}).get('wakeup_word', '')}")
        logger.info(f"通信方式: {self.config.get('communication', {}).get('type', '')}")
        logger.info(f"RAG类型: {self.config.get('rag', {}).get('type', '')}")
        logger.info("=" * 60)
        
        # 预加载模型
        self.preload_models()
        
        # 启动通信接口
        if self.comm_interface:
            self.comm_interface.start()
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            while True:
                # 检查是否允许唤醒词检测
                with self.enable_audio_lock:
                    can_detect_wakeup = self.enable_audio_playback
                
                if not can_detect_wakeup:
                    time.sleep(0.1)
                    continue
                
                logger.info("启动滑动窗口并行唤醒检测...")
                
                # 启动并行唤醒检测
                if not self.wakeup_manager.start():
                    logger.error("启动并行唤醒检测失败")
                    time.sleep(0.5)
                    continue
                
                # 等待唤醒词被检测到
                try:
                    while True:
                        has_wakeup, recognized_text = self.wakeup_manager.wait_for_wakeup(timeout=0.2)
                        
                        with self.enable_audio_lock:
                            if not self.enable_audio_playback:
                                logger.info("音频播放被禁用，停止并行唤醒检测")
                                self.wakeup_manager.stop()
                                break
                        
                        if has_wakeup:
                            logger.info(f"✓ 滑动窗口检测到唤醒词: {recognized_text}")
                            break
                except KeyboardInterrupt:
                    self.wakeup_manager.stop()
                    raise
                except Exception as e:
                    logger.error(f"并行唤醒检测出错: {e}")
                    self.wakeup_manager.stop()
                    time.sleep(0.5)
                    continue
                
                # 停止并行唤醒检测
                self.wakeup_manager.stop()
                
                if not has_wakeup:
                    continue
                
                # 唤醒后的处理
                self._handle_wakeup()
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("\n程序被用户中断，正在退出...")
        except Exception as e:
            logger.error(f"程序运行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def _handle_wakeup(self):
        """处理唤醒后的逻辑"""
        logger.info("✓ 检测到唤醒词，开始录制指令...")
        
        # 播放"我在"音频
        with self.enable_audio_lock:
            can_play = self.enable_audio_playback
        
        if can_play:
            wozai_audio = self.intent_handler.wozai_audio
            if wozai_audio:
                self._play_wozai_audio(wozai_audio)
                # 等待音频播放完成，避免回声干扰
                time.sleep(0.5)
        
        # 使用实时语音识别录制和识别指令
        instruction = self._record_instruction()
        
        if instruction:
            self._process_instruction(instruction)
    
    def _play_wozai_audio(self, audio_path: str):
        """播放'我在'音频"""
        with self.wozai_audio_lock:
            if self.wozai_audio_thread is not None and self.wozai_audio_thread.is_alive():
                logger.info("'我在'音频正在播放中，跳过重复播放")
                return
            
            def play_audio():
                try:
                    self.tts_player.play_audio(audio_path)
                except Exception as e:
                    logger.error(f"播放'我在'音频失败: {e}")
                finally:
                    with self.wozai_audio_lock:
                        self.wozai_audio_thread = None
            
            self.wozai_audio_thread = threading.Thread(target=play_audio, daemon=True)
            self.wozai_audio_thread.start()
            logger.info("已在单独线程中启动'我在'音频播放")
    
    def _record_instruction(self) -> Optional[str]:
        """录制指令"""
        if not self.realtime_asr_engine:
            logger.error("实时识别引擎未初始化")
            return None
        
        try:
            instruction = self.realtime_asr_engine.record_instruction()
            return instruction
        except Exception as e:
            logger.error(f"录制指令失败: {e}")
            return None
    
    def _process_instruction(self, instruction: str):
        """处理指令"""
        try:
            # 识别意图
            intent_tag, intent_data = self.intent_handler.recognize_intent(instruction)
            
            # 根据意图处理
            response_text, is_predefined, audio_file, need_stream = self._handle_intent(
                intent_tag, intent_data, instruction
            )
            
            logger.info(f"意图tag: {response_text if response_text else '流式播报'}")
            
            # 检查是否允许播放音频
            with self.enable_audio_lock:
                can_play = self.enable_audio_playback
            
            if can_play:
                if is_predefined and audio_file:
                    logger.info(f"播放预生成音频: {audio_file}")
                    self.tts_player._play_audio_direct(audio_file)
                    if self.comm_interface:
                        self.comm_interface.send_tag(response_text)
                else:
                    if response_text:
                        self.tts_player.speak(response_text)
                        if self.comm_interface:
                            self.comm_interface.send_tag(response_text)
            else:
                logger.info("音频播放已禁用，跳过播报")
        except Exception as e:
            logger.error(f"处理指令或播报失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_intent(self, intent_tag: str, intent_data: Dict[str, Any], instruction: str) -> Tuple[Optional[str], bool, Optional[str], bool]:
        """处理意图"""
        if intent_tag == "A":
            # 地图查询
            address = intent_data.get("address", "北京市崇文门")
            check_audio = self.intent_handler.check_audio
            if check_audio:
                self.tts_player.play_audio(check_audio)
            return self._handle_map_query(instruction, address)
        elif self.intent_handler.is_predefined_intent(intent_tag):
            # 预定义意图
            audio_file = self.intent_handler.get_audio_file(intent_tag)
            return intent_tag, True, audio_file, False
        else:
            # 知识库问答
            check_audio = self.intent_handler.check_audio
            if check_audio:
                self.tts_player.play_audio(check_audio)
            return self._handle_knowledge_query(instruction)
    
    def _handle_map_query(self, instruction: str, address: str) -> Tuple[Optional[str], bool, Optional[str], bool]:
        """处理地图查询"""
        api_keys = self.config.get("api_keys", {})
        amap_api_key = api_keys.get("amap_api_key", "")
        map_config = self.config.get("map_query", {})
        
        address = address.replace(" ", "")
        if not address:
            return "抱歉，我没听清您想查询哪个位置附近的加油站。", False, None, False
        
        # 获取地址的经纬度坐标
        geocode_url = "https://restapi.amap.com/v3/geocode/geo"
        try:
            response = requests.get(geocode_url, params={"key": amap_api_key, "address": address}, timeout=5)
            data = response.json()
            if data.get("status") == "1" and data.get("geocodes"):
                location = data["geocodes"][0]["location"]
            else:
                return f"抱歉，我找不到'{address}'这个地方，请换个更详细的地址试试。", False, None, False
        except Exception as e:
            logger.error(f"请求高德地理编码API时出错: {e}")
            return "抱歉，网络有点问题，暂时无法查询地图信息。", False, None, False
        
        # 搜索附近的加油站
        poi_url = "https://restapi.amap.com/v3/place/around"
        try:
            response = requests.get(poi_url, params={
                "key": amap_api_key,
                "location": location,
                "types": map_config.get("poi_type", "010100"),
                "radius": map_config.get("radius", 5000),
                "sortrule": map_config.get("sort_rule", "distance"),
                "offset": map_config.get("offset", 5),
                "output": "json"
            }, timeout=5)
            poi_data = response.json()
            if poi_data.get("status") != "1" or int(poi_data.get("count", 0)) == 0:
                return f"很抱歉，在'{address}'附近5公里内没有找到加油站。", False, None, False
        except Exception as e:
            logger.error(f"请求高德POI API时出错: {e}")
            return "抱歉，网络有点问题，暂时无法查询加油站信息。", False, None, False
        
        # 格式化POI数据并调用Qwen生成最终回答
        stations = []
        for poi in poi_data.get("pois", []):
            stations.append({
                "name": poi.get("name", "未知"),
                "address": poi.get("address", "未知"),
                "distance": f"{int(poi.get('distance', 0))}米"
            })
        
        map_prompt = (
            f"你是一个智能助理，请根据以下信息，用自然语言回答用户关于附近加油站的问题。\n"
            f"用户问题：{instruction}\n"
            f"查询中心点：{address}\n"
            f"附近的加油站数据：{json.dumps(stations, ensure_ascii=False)}\n"
            "回答应包含加油站名称和距离，并以友好、简洁的语气呈现。字数不超过150字"
        )
        
        try:
            completion = self.qwen_client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": map_prompt}],
                stream=True,
            )
            content_parts = [chunk.choices[0].delta.content for chunk in completion if chunk.choices and chunk.choices[0].delta.content]
            answer = "".join(content_parts).strip()
            logger.info(f"地图问答生成的答案: {answer}")
            return answer, False, None, True
        except Exception as e:
            logger.error(f"为地图问答生成答案失败: {e}")
            return "抱歉，生成地图回答时遇到了点麻烦。", False, None, False
    
    def _handle_knowledge_query(self, instruction: str) -> Tuple[Optional[str], bool, Optional[str], bool]:
        """处理知识库问答"""
        if not self.rag_instance:
            return "抱歉，我的知识库暂时无法访问，请稍后再试。", False, None, False
        
        try:
            instruction = instruction + "。回答字数不要超过150字。"
            logger.info(f"向 RAG 知识库查询: {instruction}")
            
            result = self.rag_instance.query(instruction, return_timing=True)
            
            answer = result["answer"]
            timing = result.get("timing", {})
            
            if timing.get("error", False):
                logger.error(f"RAG 查询失败: {answer}")
                return "抱歉，我的知识库暂时无法访问，请稍后再试。", False, None, False
            
            logger.info(f"RAG 返回答案 (总耗时: {timing.get('total_time', 0):.2f}秒): {answer}")
            return answer, False, None, True
        except Exception as e:
            logger.error(f"调用 RAG 服务时出错: {e}")
            import traceback
            traceback.print_exc()
            return "抱歉，与我的知识库连接时出现问题，请稍后再试。", False, None, False
    
    def _signal_handler(self, sig, frame):
        """信号处理函数"""
        logger.info('收到中断信号，正在停止识别...')
        if hasattr(self, 'realtime_asr_engine') and self.realtime_asr_engine:
            try:
                self.realtime_asr_engine.stop()
                logger.info('实时识别已停止')
            except:
                pass
        logger.info('程序退出')
        sys.exit(0)
    
    def cleanup(self):
        """清理资源"""
        self.wakeup_manager.stop()
        
        if hasattr(self, 'realtime_asr_engine') and self.realtime_asr_engine:
            try:
                self.realtime_asr_engine.cleanup()
            except:
                pass
        
        if self.comm_interface:
            self.comm_interface.stop()
        
        logger.info("程序已退出")




class ParallelWakeupManager:
    """并行唤醒管理器"""
    
    def __init__(self, wakeup_word: str, threshold: float, record_duration: float,
                 overlap_duration: float, model_path: str, config: Dict[str, Any]):
        self.wakeup_word = wakeup_word
        self.threshold = threshold
        self.record_duration = record_duration
        self.overlap_duration = overlap_duration
        self.model_path = model_path
        self.config = config
        
        self.is_running = False
        self.wakeup_detected = False
        self.detected_text = ""
        
        self.recording_thread = None
        self.processing_thread = None
        self.audio_queue = Queue(maxsize=2)
        
        self.pa = None
        self.audio_stream = None
        self.audio_stream_lock = threading.Lock()
        
        audio_config = config.get("audio", {})
        self.sample_rate = audio_config.get("sample_rate", 16000)
        self.channels = audio_config.get("channels", 1)
        self.chunk = audio_config.get("chunk", 1280)
        
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.wakeup_event = threading.Event()
        
        paths_config = config.get("paths", {})
        tmp_wav_dir = paths_config.get("tmp_wav_dir", "./tmp_wav_file")
        self.tmp_wav_dir = Path(current_dir) / tmp_wav_dir
        self.tmp_wav_dir.mkdir(exist_ok=True)
        
        logger.info(f"初始化滑动窗口并行唤醒管理器，唤醒词: {wakeup_word}, 阈值: {threshold}")
    
    def start(self) -> bool:
        """启动并行唤醒检测"""
        with self.state_lock:
            if self.is_running:
                return False
            
            old_recording_thread = self.recording_thread
            old_processing_thread = self.processing_thread
            has_old_threads = (old_recording_thread and old_recording_thread.is_alive()) or \
                              (old_processing_thread and old_processing_thread.is_alive())
        
        if has_old_threads:
            self.is_running = False
            self.stop_event.set()
            
            if old_recording_thread and old_recording_thread.is_alive():
                old_recording_thread.join(timeout=2.0)
            if old_processing_thread and old_processing_thread.is_alive():
                old_processing_thread.join(timeout=2.0)
            
            self._cleanup_audio_stream()
            time.sleep(0.2)
        
        with self.state_lock:
            self.recording_thread = None
            self.processing_thread = None
            self.is_running = True
            self.wakeup_detected = False
            self.detected_text = ""
            self.stop_event.clear()
            self.wakeup_event.clear()
            
            while not self.audio_queue.empty():
                try:
                    tmp_file_path = self.audio_queue.get_nowait()
                    self._cleanup_temp_file(tmp_file_path)
                except:
                    pass
            
            if not self._init_audio_stream():
                self.is_running = False
                return False
            
            self.recording_thread = threading.Thread(
                target=self._recording_loop,
                daemon=True,
                name="RecordingThread"
            )
            self.recording_thread.start()
            
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="ProcessingThread"
            )
            self.processing_thread.start()
            
            return True
    
    def stop(self):
        """停止并行唤醒检测"""
        with self.state_lock:
            if not self.is_running:
                return
            
            self.is_running = False
            self.stop_event.set()
            self.wakeup_detected = False
            self.detected_text = ""
            self.wakeup_event.clear()
        
        max_wait_time = 3.0
        
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=max_wait_time)
            self.recording_thread = None
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=max_wait_time)
            self.processing_thread = None
        
        self._cleanup_audio_stream()
        
        while not self.audio_queue.empty():
            try:
                tmp_file_path = self.audio_queue.get_nowait()
                self._cleanup_temp_file(tmp_file_path)
            except:
                pass
    
    def wait_for_wakeup(self, timeout: float = None) -> Tuple[bool, str]:
        """等待唤醒词被检测到"""
        if self.wakeup_event.wait(timeout=timeout):
            with self.state_lock:
                return self.wakeup_detected, self.detected_text
        return False, ""
    
    def _init_audio_stream(self) -> bool:
        """初始化持久音频流"""
        try:
            from audio_device import AudioStream
            
            self._cleanup_audio_stream()
            time.sleep(0.1)
            
            device_index = AudioStream.select_best_device(target_sample_rate=self.sample_rate)
            if device_index is None:
                logger.error("无法找到可用的音频输入设备")
                return False
            
            self.pa = pyaudio.PyAudio()
            
            try:
                is_supported = self.pa.is_format_supported(
                    self.sample_rate,
                    input_device=device_index,
                    input_channels=self.channels,
                    input_format=pyaudio.paInt16
                )
                if not is_supported:
                    device_info = self.pa.get_device_info_by_index(device_index)
                    default_rate = int(device_info.get('defaultSampleRate', 16000))
                    self.sample_rate = default_rate
            except Exception as e:
                logger.warning(f"无法验证设备采样率支持: {e}")
            
            self.audio_stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=device_index,
            )
            
            return True
        except Exception as e:
            logger.error(f"初始化音频流失败: {e}")
            self._cleanup_audio_stream()
            return False
    
    def _cleanup_audio_stream(self):
        """清理音频流"""
        with self.audio_stream_lock:
            try:
                if self.audio_stream:
                    try:
                        if self.audio_stream.is_active():
                            self.audio_stream.stop_stream()
                    except:
                        pass
                    try:
                        self.audio_stream.close()
                    except:
                        pass
                    self.audio_stream = None
                
                if self.pa:
                    try:
                        self.pa.terminate()
                    except:
                        pass
                    self.pa = None
                
                # 等待一小段时间确保资源释放
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"清理音频流时出错: {e}")
    
    def _frames_to_wav_bytes(self, frames: List[bytes]) -> bytes:
        """将音频帧列表转换为WAV格式的字节数据"""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        return wav_buffer.getvalue()
    
    def _create_temp_file(self, audio_bytes: bytes) -> Optional[str]:
        """创建临时文件并写入音频数据"""
        try:
            tmp_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False,
                dir=current_dir
            )
            tmp_file.write(audio_bytes)
            tmp_file.flush()
            tmp_file.close()
            return tmp_file.name
        except Exception as e:
            logger.error(f"创建临时文件失败: {e}")
            return None
    
    def _cleanup_temp_file(self, tmp_file_path: str):
        """清理临时文件"""
        try:
            if tmp_file_path and Path(tmp_file_path).exists():
                os.unlink(tmp_file_path)
        except Exception as e:
            logger.warning(f"清理临时文件失败 {tmp_file_path}: {e}")
    
    def _recording_loop(self):
        """录音循环"""
        logger.info("录音线程启动（滑动窗口模式）")
        
        chunks_per_window = int(self.sample_rate / self.chunk * self.record_duration)
        chunks_per_overlap = int(self.sample_rate / self.chunk * self.overlap_duration)
        
        audio_buffer = deque(maxlen=chunks_per_window + chunks_per_overlap)
        
        while not self.stop_event.is_set():
            try:
                if not self.audio_stream:
                    break
                
                data = self.audio_stream.read(self.chunk, exception_on_overflow=False)
                audio_buffer.append(data)
                
                if len(audio_buffer) >= chunks_per_window:
                    with self.state_lock:
                        if self.wakeup_detected:
                            break
                    
                    window_frames = list(audio_buffer)[-chunks_per_window:]
                    audio_bytes = self._frames_to_wav_bytes(window_frames)
                    
                    tmp_file_path = self._create_temp_file(audio_bytes)
                    if not tmp_file_path:
                        for _ in range(chunks_per_window - chunks_per_overlap):
                            if audio_buffer:
                                audio_buffer.popleft()
                        continue
                    
                    try:
                        timestamp = time.time()
                        dt = datetime.fromtimestamp(timestamp)
                        formatted = dt.strftime("%Y%m%d%H%M%f")[:-3]
                        filename = f"{formatted}.wav"
                        saved_file_path = self.tmp_wav_dir / filename
                        with open(saved_file_path, 'wb') as f:
                            f.write(audio_bytes)
                    except Exception as e:
                        logger.warning(f"保存临时文件失败: {e}")
                    
                    if not self.stop_event.is_set():
                        try:
                            self.audio_queue.put(tmp_file_path, timeout=0.5)
                        except:
                            self._cleanup_temp_file(tmp_file_path)
                            try:
                                old_file = self.audio_queue.get_nowait()
                                self._cleanup_temp_file(old_file)
                                try:
                                    self.audio_queue.put(tmp_file_path, timeout=0.1)
                                except:
                                    self._cleanup_temp_file(tmp_file_path)
                            except:
                                self._cleanup_temp_file(tmp_file_path)
                    
                    for _ in range(chunks_per_window - chunks_per_overlap):
                        if audio_buffer:
                            audio_buffer.popleft()
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"录音线程出错: {e}")
        
        logger.info("录音线程结束")
    
    def _processing_loop(self):
        """处理循环"""
        logger.info("处理线程启动")
        
        while not self.stop_event.is_set():
            try:
                try:
                    tmp_file_path = self.audio_queue.get(timeout=0.5)
                except:
                    continue
                
                with self.state_lock:
                    if self.wakeup_detected:
                        self._cleanup_temp_file(tmp_file_path)
                        break
                
                logger.debug(f"处理音频窗口: {tmp_file_path}")
                has_wakeup, recognized_text = self._check_wakeup_word_parallel(tmp_file_path)
                
                self._cleanup_temp_file(tmp_file_path)
                
                if has_wakeup:
                    with self.state_lock:
                        self.wakeup_detected = True
                        self.detected_text = recognized_text
                    
                    self.wakeup_event.set()
                    logger.info(f"处理线程检测到唤醒词: {recognized_text}")
                    break
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"处理线程出错: {e}")
                break
        
        logger.info("处理线程结束")
    
    def _check_wakeup_word_parallel(self, audio_file: str) -> Tuple[bool, str]:
        """并行版本：检查音频中是否包含唤醒词"""
        try:
            text = transcribe_audio_funasr_preloaded(
                audio_file,
                model_path=self.model_path,
                config=self.config,
                wakeup_word=self.wakeup_word
            )
            if text is None or not text.strip():
                return False, ""
            
            logger.debug(f"并行识别文本: {text}")
            match_score = 0
            text = text.replace(" ", "")
            if self.wakeup_word in text:
                match_score = 1
            
            if match_score >= self.threshold:
                logger.debug(f"并行检测到唤醒词: {self.wakeup_word} (匹配分数: {match_score:.2f})")
                return True, text
            
            return False, text
        except Exception as e:
            logger.error(f"并行检查唤醒词失败: {e}")
            return False, ""


def main():
    """主函数"""
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()

