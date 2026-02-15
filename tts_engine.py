"""
TTS接口模块 - 支持EdgeTTS和DashScope TTS
"""

import os
import time
import threading
import logging
import subprocess
import platform
from pathlib import Path
import re
import asyncio
from typing import Optional, Generator, Tuple, Dict, List

# 尝试导入edge-tts
try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False
    print("警告: 未安装edge-tts，请运行: pip install edge-tts")

# 尝试导入dashscope
try:
    import dashscope
    from dashscope.audio.tts import SpeechSynthesizer
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False
    print("警告: 未安装dashscope，请运行: pip install dashscope")

logger = logging.getLogger(__name__)


class TTSPlayer:
    """
    TTS播放器，支持多种TTS引擎，使用系统命令播放音频
    """
    
    # 支持的音频播放工具（按优先级排序）
    PLAYERS = [
        (['mpv', '--no-terminal', '--really-quiet'], 'mpv'),
        (['mpg123', '-q'], 'mpg123'),
        (['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet'], 'ffplay'),
        (['aplay'], 'aplay'),  # Linux ALSA
        (['paplay'], 'paplay'),  # Linux PulseAudio
        (['afplay'], 'afplay'),  # macOS
    ]
    
    # Windows特定播放工具
    if platform.system() == 'Windows':
        PLAYERS.append((['powershell', '-c', '(New-Object Media.SoundPlayer "{}").PlaySync();'], 'windows_sound'))
    
    def __init__(
        self, 
        engine: str = "dashscope",  # 引擎选择：edge 或 dashscope
        voice: str = "zh-CN-XiaoxiaoNeural",  # EdgeTTS音色
        dashscope_model: str = "sambert-zhihao-v1",  # DashScope模型
        dashscope_voice: str = "zhizhe",  # DashScope音色：zhiyan（知言）、zhitian（知甜）、zhizhe（知哲）
        sample_rate: int = 48000,  # DashScope采样率
        keep_files: bool = False,
        auto_play: bool = True,
        dashscope_api_key: str = None,  # DashScope API密钥
        output_dir: str = ".",  # 输出目录
        config_file: Optional[str] = None  # 配置文件路径
    ):
        """
        初始化TTS播放器
        
        参数:
            engine: TTS引擎，可选 'edge' 或 'dashscope'
            voice: EdgeTTS音色名称
            dashscope_model: DashScope模型名称
            dashscope_voice: DashScope音色名称
            sample_rate: 音频采样率
            keep_files: 是否保留生成的音频文件
            auto_play: 是否自动播放
            dashscope_api_key: DashScope API密钥，如果为None则从环境变量读取
            output_dir: 语音文件输出目录
            config_file: YAML配置文件路径，如果提供则从配置文件加载设置
        """
        self.engine = engine
        self.voice = voice
        self.dashscope_model = dashscope_model
        self.dashscope_voice = dashscope_voice
        self.sample_rate = sample_rate
        self.keep_files = keep_files
        self.auto_play = auto_play
        self.dashscope_api_key = dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找可用的播放工具
        self._player_tool = None
        self._find_player()
        
        # 如果提供了配置文件，从配置文件加载
        if config_file:
            self._load_config(config_file)
        
        # 初始化DashScope
        if engine == "dashscope" and HAS_DASHSCOPE and self.dashscope_api_key:
            dashscope.api_key = self.dashscope_api_key
            logger.info(f"DashScope TTS已初始化，使用模型: {dashscope_model}, 音色: {dashscope_voice}")
        elif engine == "dashscope" and not self.dashscope_api_key:
            logger.warning("未设置DashScope API密钥，请设置环境变量DASHSCOPE_API_KEY")
        
        logger.info(f"TTS播放器初始化完成，引擎: {engine}")
    
    def _load_config(self, config_file: str) -> None:
        """
        从配置文件加载配置
        
        参数:
            config_file: 配置文件路径
        """
        try:
            import yaml
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    
                # 更新配置
                if config:
                    for key, value in config.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                            logger.info(f"从配置文件加载 {key} = {value}")
        except ImportError:
            logger.warning("YAML未安装，跳过配置文件加载，请运行: pip install pyyaml")
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}")
    
    def _find_player(self) -> Optional[Tuple[List[str], str]]:
        """
        查找可用的音频播放工具
        
        返回:
            可用的播放工具命令和名称，如果未找到返回None
        """
        if self._player_tool is not None:
            return self._player_tool
        
        for player_cmd, player_name in self.PLAYERS:
            try:
                if player_name == 'windows_sound':
                    # Windows特殊处理
                    self._player_tool = (player_cmd, player_name)
                    return self._player_tool
                
                # 检查命令是否存在
                result = subprocess.run(
                    ['which', player_cmd[0]],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    self._player_tool = (player_cmd, player_name)
                    logger.info(f"找到音频播放工具: {player_name} ({player_cmd[0]})")
                    return self._player_tool
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        logger.warning("未找到可用的音频播放工具")
        return None
    
    def _get_system_players(self) -> List[str]:
        """
        获取系统推荐的播放器安装命令
        
        返回:
            系统推荐的安装命令列表
        """
        system = platform.system()
        if system == "Linux":
            return [
                "sudo apt-get install mpv        # 推荐",
                "sudo apt-get install mpg123     # 或",
                "sudo apt-get install ffmpeg     # 包含ffplay",
                "sudo apt-get install alsa-utils # 包含aplay",
            ]
        elif system == "Darwin":  # macOS
            return [
                "brew install mpv               # 推荐",
                "brew install mpg123            # 或",
                "brew install ffmpeg            # 包含ffplay",
            ]
        elif system == "Windows":
            return [
                "下载并安装 MPV: https://mpv.io/installation/",
                "或安装 FFmpeg: https://ffmpeg.org/download.html",
                "或安装 mpg123: https://www.mpg123.de/download.shtml",
            ]
        return []
    
    def speak(self, text: str, save_path: Optional[str] = None) -> Optional[str]:
        """
        将文本转换为语音并播放
        
        参数:
            text: 要转换为语音的文本
            save_path: 保存音频文件的路径（可选）
            
        返回:
            音频文件路径（如果保存）或None
        """
        if not text:
            return None
        
        # 清理文本
        text = self._clean_text(text)
        
        if self.engine == "edge":
            return self._speak_edge(text, save_path)
        elif self.engine == "dashscope":
            return self._speak_dashscope(text, save_path)
        else:
            logger.error(f"不支持的TTS引擎: {self.engine}")
            return None
    
    def speak_stream(self, text_generator: Generator[str, None, None]) -> None:
        """
        流式播报文本
        
        参数:
            text_generator: 生成文本的生成器
        """
        full_text = ""
        for text_chunk in text_generator:
            if text_chunk:
                full_text += text_chunk + " "
        
        # 清理完整文本
        full_text = self._clean_text(full_text.strip())
        
        if full_text:
            self.speak(full_text)
    
    def _speak_edge(self, text: str, save_path: Optional[str] = None) -> Optional[str]:
        """使用EdgeTTS合成语音"""
        if not HAS_EDGE_TTS:
            logger.error("EdgeTTS未安装")
            return None
        
        try:
            # 创建临时文件
            if save_path is None:
                # 使用输出目录
                safe_text = "".join(c for c in text[:20] if c.isalnum() or c in (' ', '-', '_'))
                safe_text = safe_text.replace(' ', '_')
                output_file = f"tts_edge_{int(time.time())}_{safe_text}.mp3"
                audio_file = str(self.output_dir / output_file)
            else:
                audio_file = save_path
            
            # 异步合成语音
            async def _async_synthesize():
                tts = edge_tts.Communicate(text=text, voice=self.voice)
                await tts.save(audio_file)
            
            # 运行异步任务
            asyncio.run(_async_synthesize())
            
            logger.info(f"EdgeTTS合成成功，文件: {audio_file}")
            
            # 播放音频
            if self.auto_play:
                self.play_audio(audio_file)
            
            # 如果不保留文件且是临时文件，则删除
            if not self.keep_files and save_path is None:
                threading.Thread(
                    target=self._delayed_delete,
                    args=(audio_file, 10),  # 10秒后删除
                    daemon=True
                ).start()
                return None
            else:
                return audio_file
                
        except Exception as e:
            logger.error(f"EdgeTTS合成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _speak_dashscope(self, text: str, save_path: Optional[str] = None) -> Optional[str]:
        """使用DashScope TTS合成语音"""
        if not HAS_DASHSCOPE:
            logger.error("DashScope未安装")
            return None
        
        if not self.dashscope_api_key:
            logger.error("未设置DashScope API密钥")
            return None
        
        try:
            # 创建临时文件
            if save_path is None:
                # 使用输出目录
                safe_text = "".join(c for c in text[:20] if c.isalnum() or c in (' ', '-', '_'))
                safe_text = safe_text.replace(' ', '_')
                output_file = f"tts_dashscope_{int(time.time())}_{safe_text}.wav"
                audio_file = str(self.output_dir / output_file)
            else:
                audio_file = save_path
            
            # 调用DashScope TTS API
            result = SpeechSynthesizer.call(
                model=self.dashscope_model,
                text=text,
                voice=self.dashscope_voice,
                sample_rate=self.sample_rate,
                format='wav',
                volume=100
            )
            
            if result.get_audio_data() is not None:
                # 保存音频数据到文件
                with open(audio_file, 'wb') as f:
                    f.write(result.get_audio_data())
                
                logger.info(f"DashScope TTS合成成功，文件: {audio_file}")
                
                # 播放音频
                if self.auto_play:
                    self._play_audio_direct(audio_file)
                
                # 如果不保留文件且是临时文件，则删除
                if not self.keep_files and save_path is None:
                    threading.Thread(
                        target=self._delayed_delete,
                        args=(audio_file, 10),  # 10秒后删除
                        daemon=True
                    ).start()
                    return None
                else:
                    return audio_file
            else:
                logger.error(f"DashScope TTS合成失败: {result}")
                return None
                
        except Exception as e:
            logger.error(f"DashScope TTS合成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def play_audio(self, audio_file: str) -> bool:
        """
        播放音频文件
        
        参数:
            audio_file: 音频文件路径
            
        返回:
            是否播放成功
        """
        if not Path(audio_file).exists():
            logger.error(f"音频文件不存在: {audio_file}")
            return False
        
        # 在新线程中播放音频，避免阻塞主线程
        def _play_thread():
            try:
                return self._play_audio_direct(audio_file)
            except Exception as e:
                logger.error(f"播放音频失败: {e}")
                return False
        
        thread = threading.Thread(target=_play_thread, daemon=True)
        thread.start()
        return True
    
    def _play_audio_direct(self, audio_file: str) -> bool:
        """直接播放音频文件（使用系统命令）"""
        player_info = self._find_player()
        if player_info is None:
            # 提示用户安装播放工具
            logger.error("未找到可用的音频播放工具")
            print("\n请安装以下工具之一：")
            for cmd in self._get_system_players():
                print(f"  {cmd}")
            return False
        
        player_cmd, player_name = player_info
        
        try:
            if player_name == 'windows_sound':
                # Windows特殊处理：使用powershell播放
                cmd_str = player_cmd[2].format(audio_file)
                cmd = ['powershell', '-c', cmd_str]
                subprocess.run(cmd, check=True, timeout=30)
            else:
                # 普通系统命令播放
                subprocess.run(
                    player_cmd + [audio_file],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30
                )
            
            logger.info(f"音频播放成功: {audio_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"播放命令执行失败: {e}")
        except subprocess.TimeoutExpired:
            logger.error(f"播放超时: {audio_file}")
        except Exception as e:
            logger.error(f"播放音频失败: {e}")
        
        return False
    
    def _clean_text(self, text: str) -> str:
        """清理文本，移除不需要的字符"""
        if not text:
            return ""
        
        # 移除markdown格式
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **粗体**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # *斜体*
        text = re.sub(r'`([^`]+)`', r'\1', text)  # `代码`
        text = re.sub(r'#+\s*', '', text)  # # 标题
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [链接](url)
        
        # 移除URL
        text = re.sub(r'https?://\S+', '', text)
        
        # 移除emoji
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # 表情符号
            "\U0001F300-\U0001F5FF"  # 符号和象形文字
            "\U0001F680-\U0001F6FF"  # 交通和地图符号
            "\U0001F1E0-\U0001F1FF"  # 旗帜
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # 移除多余空格和换行
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _delayed_delete(self, file_path: str, delay_seconds: int = 10):
        """延迟删除文件"""
        time.sleep(delay_seconds)
        try:
            if Path(file_path).exists():
                os.unlink(file_path)
                logger.debug(f"已删除临时音频文件: {file_path}")
        except Exception as e:
            logger.warning(f"删除文件失败 {file_path}: {e}")
    
    def get_available_players(self) -> List[Dict[str, str]]:
        """
        获取所有可用的音频播放工具
        
        返回:
            播放工具信息列表
        """
        available = []
        for player_cmd, player_name in self.PLAYERS:
            try:
                if player_name == 'windows_sound':
                    available.append({
                        'name': player_name,
                        'command': 'powershell sound player',
                        'status': 'available'
                    })
                    continue
                
                result = subprocess.run(
                    ['which', player_cmd[0]],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    available.append({
                        'name': player_name,
                        'command': ' '.join(player_cmd),
                        'status': 'available'
                    })
                else:
                    available.append({
                        'name': player_name,
                        'command': ' '.join(player_cmd),
                        'status': 'not installed'
                    })
            except Exception:
                available.append({
                    'name': player_name,
                    'command': ' '.join(player_cmd),
                    'status': 'check failed'
                })
        
        return available