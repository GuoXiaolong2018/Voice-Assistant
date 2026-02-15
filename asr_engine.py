#!/usr/bin/env python3
"""
语音识别引擎 - 使用 FunASR 和 DashScope 进行语音识别
包含离线识别、实时识别和模型预加载功能
"""

import os
import sys
import argparse
import logging
import threading
import time
import re
import wave
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import yaml
import ctypes

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入实时语音识别相关模块
try:
    import pyaudio
    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
    HAS_DASHSCOPE_ASR = True
except ImportError:
    HAS_DASHSCOPE_ASR = False
    logger.warning("DashScope ASR 模块未安装，实时识别功能将不可用")

# ============================================================
# 环境初始化（从原文件提取）
# ============================================================
os.environ.update({
    "LIBASOUND_DEBUG": "0",
    "JACK_NO_START_SERVER": "1",
    "PIPEWIRE_DISABLE": "1",
    "PULSE_SERVER": "unix:/dev/null",
    "CT2_LOG_LEVEL": "error"
})

# ============================================================
# 预加载 cuDNN（从原文件提取）
# ============================================================
conda_env = os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV")
if conda_env:
    for pyver in ["python3.10", "python3.11", "python3.12"]:
        cudnn_path = os.path.join(conda_env, "lib", pyver, "site-packages", "nvidia", "cudnn", "lib")
        if os.path.isdir(cudnn_path):
            os.environ["LD_LIBRARY_PATH"] = f"{cudnn_path}:{os.environ.get('LD_LIBRARY_PATH','')}".rstrip(":")
            for lib in ["libcudnn_ops.so.9", "libcudnn.so.9"]:
                lp = os.path.join(cudnn_path, lib)
                if os.path.exists(lp):
                    try:
                        ctypes.CDLL(lp, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
            break

# 导入 FunASR
try:
    from funasr import AutoModel
    HAS_FUNASR = True
except ImportError:
    HAS_FUNASR = False
    logger.warning("未安装 FunASR，离线识别功能将不可用")


def load_model_path_from_config(config_path: str = None) -> Optional[str]:
    """
    从配置文件加载模型路径
    
    参数:
        config_path: 配置文件路径（yaml格式）
    
    返回:
        模型路径，如果未找到则返回 None
    """
    if config_path is None:
        # 尝试查找默认配置文件
        current_dir = Path(__file__).parent
        possible_configs = [
            current_dir / "g1_electrical_cabinet_demo_v1-main" / "g1_electrical_cabinet_demo_v1" / "config.yaml",
            current_dir / "config.yaml",
        ]
        
        for config_file in possible_configs:
            if config_file.exists():
                config_path = str(config_file)
                break
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                model_path = config.get("voice2txt_model_path")
                if model_path:
                    # 如果是相对路径，转换为绝对路径
                    if not os.path.isabs(model_path):
                        config_dir = Path(config_path).parent
                        model_path = str(config_dir / model_path)
                    return model_path
        except Exception as e:
            logger.warning(f"读取配置文件失败: {e}")
    
    return None


def transcribe_audio_funasr(
    audio_path: str,
    model_path: str = None,
    config_path: str = None,
    batch_size_s: int = 300,
    show_tqdm: bool = False,
    vad_model: str = None,
    punc_model: str = None,
) -> Optional[str]:
    """
    使用 FunASR 将音频文件转换为文字
    
    参数:
        audio_path: 音频文件路径
        model_path: FunASR 模型路径（如果为 None，则从配置文件读取）
        config_path: 配置文件路径（可选）
        batch_size_s: 批处理大小（秒）
        show_tqdm: 是否显示进度条
        vad_model: VAD 模型（可选，None 表示不使用）
        punc_model: 标点模型（可选，None 表示不使用）
    
    返回:
        识别的文字内容，如果失败则返回 None
    """
    # 检查音频文件是否存在
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    # 获取模型路径
    if model_path is None:
        model_path = load_model_path_from_config(config_path)
        if model_path is None:
            raise ValueError(
                "未指定模型路径，且无法从配置文件读取。\n"
                "请使用 --model-path 参数指定模型路径，或确保配置文件存在且包含 voice2txt_model_path 字段。"
            )
    
    # 检查模型路径是否存在
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    logger.info(f"正在加载 FunASR 模型: {model_path}")
    
    # 初始化模型（从原文件提取的配置）
    try:
        asr_model = AutoModel(
            model=model_path,
            vad_kwargs={"max_end_silence_time": 0.5},
            vad_model=vad_model,
            punc_model=punc_model,
            disable_update=True,
        )
        logger.info("模型加载完成")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise
    
    # 进行识别（从原文件提取的逻辑）
    try:
        logger.info(f"正在识别音频文件: {audio_path}")
        res = asr_model.generate(
            input=audio_path,
            batch_size_s=batch_size_s,
            show_tqdm=show_tqdm
        )
        
        # 处理返回结果（从原文件提取的逻辑）
        if isinstance(res, dict):
            text = res.get("text", "")
        elif isinstance(res, list) and len(res) > 0:
            # 兼容旧版（如果以后换旧模型）
            text = res[0].get("text", "")
        else:
            text = ""
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"FunASR 识别失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# 全局变量：预加载的 FunASR 模型
# ============================================================
_preloaded_asr_model = None
_asr_model_lock = threading.Lock()


# ============================================================
# FunASR 模型预加载功能
# ============================================================
def preload_funasr_model(model_path: str, config: Dict[str, Any] = None) -> bool:
    """
    预加载 FunASR 模型到全局变量
    
    参数:
        model_path: 模型路径
        config: 配置字典，包含 asr 相关配置
    
    返回:
        是否预加载成功
    """
    global _preloaded_asr_model, _asr_model_lock
    
    if not HAS_FUNASR:
        logger.warning("FunASR 未安装，无法预加载模型")
        return False
    
    if not Path(model_path).exists():
        logger.error(f"模型路径不存在: {model_path}")
        return False
    
    try:
        with _asr_model_lock:
            if _preloaded_asr_model is not None:
                logger.info("FunASR 模型已预加载")
                return True
            
            asr_config = config.get("asr", {}) if config else {}
            _preloaded_asr_model = AutoModel(
                model=model_path,
                vad_kwargs={"max_end_silence_time": asr_config.get("vad_max_end_silence_time", 0.3)},
                vad_model=None,
                punc_model=None,
                disable_update=True,
                device=asr_config.get("device", "cuda:0")
            )
            logger.info("FunASR 模型预加载成功")
            return True
    except Exception as e:
        logger.error(f"预加载 FunASR 模型失败: {e}")
        return False


def transcribe_audio_funasr_preloaded(
    audio_path: str,
    model_path: str = None,
    config: Dict[str, Any] = None,
    wakeup_word: str = None
) -> Optional[str]:
    """
    使用预加载的 FunASR 模型进行语音识别
    
    参数:
        audio_path: 音频文件路径
        model_path: 模型路径（如果预加载模型不存在，则使用此路径）
        config: 配置字典
        wakeup_word: 唤醒词（用于热词识别）
    
    返回:
        识别的文字内容，如果失败则返回 None
    """
    global _preloaded_asr_model, _asr_model_lock
    
    if not Path(audio_path).exists():
        logger.error(f"音频文件不存在: {audio_path}")
        return None
    
    with _asr_model_lock:
        if _preloaded_asr_model is None:
            # 如果没有预加载模型，则使用 transcribe_audio_funasr
            if model_path:
                return transcribe_audio_funasr(audio_path, model_path=model_path)
            else:
                logger.error("预加载模型不存在，且未提供模型路径")
                return None
        
        asr_model = _preloaded_asr_model
    
    try:
        asr_config = config.get("asr", {}) if config else {}
        res = asr_model.generate(
            input=audio_path,
            batch_size_s=asr_config.get("batch_size_s", 5),
            batch_size=asr_config.get("batch_size", 16),
            hotword=wakeup_word,
            use_itn=asr_config.get("use_itn", True),
            beam_size=asr_config.get("beam_size", 1),
            show_tqdm=False,
            normalize=asr_config.get("normalize", True),
        )
        
        if isinstance(res, dict):
            text = res.get("text", "")
        elif isinstance(res, list) and len(res) > 0:
            text = res[0].get("text", "")
        else:
            text = ""
        
        return text.strip()
    except Exception as e:
        logger.error(f"FunASR 识别失败: {e}")
        return None


# ============================================================
# 实时语音识别功能
# ============================================================
class RealtimeRecognitionCallback(RecognitionCallback):
    """实时语音识别回调类"""
    
    def __init__(self, on_text_callback: Callable[[str], None] = None,
                 on_sentence_end_callback: Callable[[str], None] = None):
        """
        初始化回调
        
        参数:
            on_text_callback: 当识别到文本时的回调函数，参数为识别文本
            on_sentence_end_callback: 当检测到句子结束时的回调函数，参数为完整句子
        """
        self.on_text_callback = on_text_callback
        self.on_sentence_end_callback = on_sentence_end_callback
        self.realtime_stream = None
        self.realtime_mic = None
        self.cur_asr_recognized_text = None
        self.full_sentence = None
        self.is_processing = False
    
    def on_open(self) -> None:
        """识别服务打开时的回调"""
        logger.info('实时识别回调：服务已打开')
        try:
            realtime_mic = pyaudio.PyAudio()
            realtime_stream = realtime_mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True
            )
            self.realtime_stream = realtime_stream
            self.realtime_mic = realtime_mic
        except Exception as e:
            logger.error(f"初始化音频流失败: {e}")
    
    def on_close(self) -> None:
        """识别服务关闭时的回调"""
        logger.info('实时识别回调：服务已关闭')
        if self.realtime_stream:
            try:
                if self.realtime_stream.is_active():
                    self.realtime_stream.stop_stream()
                self.realtime_stream.close()
            except:
                pass
            self.realtime_stream = None
        
        if self.realtime_mic:
            try:
                self.realtime_mic.terminate()
            except:
                pass
            self.realtime_mic = None
    
    def on_complete(self) -> None:
        """识别完成时的回调"""
        logger.info('实时识别回调：识别已完成')
    
    def on_error(self, message) -> None:
        """识别出错时的回调"""
        logger.error(f'实时识别回调错误 - task_id: {message.request_id}, error: {message.message}')
        if self.realtime_stream and self.realtime_stream.is_active():
            try:
                self.realtime_stream.stop_stream()
                self.realtime_stream.close()
            except:
                pass
        logger.error("实时识别服务出错，但继续运行")
    
    def on_event(self, result: RecognitionResult) -> None:
        """识别事件回调"""
        sentence = result.get_sentence()
        if 'text' in sentence:
            self.cur_asr_recognized_text = sentence["text"]
            
            if len(self.cur_asr_recognized_text) < 2:
                return
            
            # 过滤无意义字符
            meaningless_chars = '嗯啊哦呃。'
            if sum(1 for c in self.cur_asr_recognized_text if c in meaningless_chars) / len(self.cur_asr_recognized_text) > 0.5:
                return
            
            # 清理文本
            self.cur_asr_recognized_text = self.cur_asr_recognized_text.replace(" ", "").replace("我在，", "").replace("我在", "")
            self.cur_asr_recognized_text = re.sub(r'[。.?!！？]+$', '', self.cur_asr_recognized_text)
            
            # 调用文本回调
            if self.on_text_callback:
                self.on_text_callback(self.cur_asr_recognized_text)
            
            # 检测句子结束
            if len(self.cur_asr_recognized_text) >= 2 and RecognitionResult.is_sentence_end(sentence):
                logger.info(f'检测到整句: {self.cur_asr_recognized_text}')
                
                if self.full_sentence is not None:
                    self.full_sentence = self.full_sentence + "。" + self.cur_asr_recognized_text
                else:
                    self.full_sentence = self.cur_asr_recognized_text
                    self.cur_asr_recognized_text = ""
                
                self.is_processing = True
                
                # 调用句子结束回调
                if self.on_sentence_end_callback and self.full_sentence:
                    self.on_sentence_end_callback(self.full_sentence)


class RealtimeASREngine:
    """实时语音识别引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化实时识别引擎
        
        参数:
            config: 配置字典，包含 asr.realtime 相关配置
        """
        self.config = config
        self.realtime_config = config.get("asr", {}).get("realtime", {})
        self.realtime_recognition = None
        self.callback = None
        self.full_sentence = ""
        self.cur_asr_recognized_text = None
        self.is_processing = False
        self.realtime_audio_frames = []
        self.realtime_audio_frames_lock = threading.Lock()
        self.G_RASR_CUR_START = time.time()
        self.G_RASR_CUR_END = time.time()
    
    def initialize(self):
        """初始化实时识别"""
        if not HAS_DASHSCOPE_ASR:
            logger.error("DashScope ASR 模块未安装，无法初始化实时识别")
            return False
        
        def on_text(text: str):
            self.cur_asr_recognized_text = text
        
        def on_sentence_end(sentence: str):
            self.full_sentence = sentence
            self.is_processing = True
        
        self.callback = RealtimeRecognitionCallback(
            on_text_callback=on_text,
            on_sentence_end_callback=on_sentence_end
        )
        
        try:
            self.realtime_recognition = Recognition(
                model=self.realtime_config.get("model", "fun-asr-realtime"),
                format=self.realtime_config.get("format", "pcm"),
                sample_rate=self.realtime_config.get("sample_rate", 16000),
                semantic_punctuation_enabled=self.realtime_config.get("semantic_punctuation_enabled", False),
                max_sentence_silence=self.realtime_config.get("max_sentence_silence", 800),
                multi_threshold_mode_enabled=self.realtime_config.get("multi_threshold_mode_enabled", True),
                vocabulary_id=None,
                callback=self.callback
            )
            return True
        except Exception as e:
            logger.error(f"初始化实时识别失败: {e}")
            return False
    
    def record_instruction(self) -> Optional[str]:
        """
        录制指令并返回识别结果
        
        返回:
            识别的指令文本，如果失败则返回 None
        """
        if not self.realtime_recognition:
            logger.error("实时识别未初始化")
            return None
        
        try:
            logger.info("请说指令...")
            
            with self.realtime_audio_frames_lock:
                self.realtime_audio_frames = []
            
            # 清空所有识别状态
            self.full_sentence = ""
            self.cur_asr_recognized_text = None
            self.is_processing = False
            
            # 清空回调中的状态
            if self.callback:
                self.callback.full_sentence = None
                self.callback.cur_asr_recognized_text = None
                self.callback.is_processing = False
            
            self.realtime_recognition.start()
            self.G_RASR_CUR_START = time.time()
            instruction = None
            
            while True:
                self.is_processing = False
                logger.info("请说话，我正在聆听...")
                local_time_duration = False
                
                block_size = self.realtime_config.get("block_size", 3200)
                
                # 从回调中获取stream
                while not self.is_processing:
                    if self.callback and self.callback.realtime_stream:
                        try:
                            data = self.callback.realtime_stream.read(block_size, exception_on_overflow=False)
                            
                            # 累积保存音频数据
                            try:
                                with self.realtime_audio_frames_lock:
                                    self.realtime_audio_frames.append(data)
                            except Exception as e:
                                logger.warning(f"累积音频块失败: {e}")
                            
                            self.realtime_recognition.send_audio_frame(data)
                            time.sleep(0.01)
                            
                            self.G_RASR_CUR_END = time.time()
                            local_time_duration = (self.G_RASR_CUR_END - self.G_RASR_CUR_START) > 6.0
                            if local_time_duration:
                                break
                        except Exception as e:
                            logger.error(f"读取音频流失败: {e}")
                            break
                    else:
                        logger.error("音频流未初始化")
                        break
                
                if local_time_duration and self.cur_asr_recognized_text and len(self.cur_asr_recognized_text) >= 2:
                    self.full_sentence = self.cur_asr_recognized_text
                
                if self.full_sentence:
                    self.realtime_recognition.stop()
                    
                    # 保存累积的音频数据
                    try:
                        with self.realtime_audio_frames_lock:
                            if self.realtime_audio_frames:
                                save_dir = self.config.get("paths", {}).get("realtime_audio_save_dir", "./instruction_temp_file")
                                self._save_audio_frames(self.realtime_audio_frames, save_dir)
                                self.realtime_audio_frames = []
                    except Exception as e:
                        logger.warning(f"保存累积音频失败: {e}")
                    
                    instruction = self.full_sentence
                    self.full_sentence = ""
                    logger.info(f"实时识别到的指令: {instruction}")
                    break
                else:
                    self.realtime_recognition.stop()
                    break
        except Exception as e:
            logger.error(f"实时识别指令失败: {e}")
            try:
                if self.realtime_recognition:
                    self.realtime_recognition.stop()
            except:
                pass
        
        return instruction
    
    def _save_audio_frames(self, audio_frames: list, output_dir: str):
        """保存音频帧"""
        current_dir = Path(__file__).parent
        output_dir = Path(current_dir) / output_dir
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S_%f")[:-3]
        filename = f"audio_{timestamp}.wav"
        filepath = output_dir / filename
        
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b''.join(audio_frames))
        
        logger.info(f"已保存完整音频: {filepath}")
    
    def stop(self):
        """停止实时识别"""
        if self.realtime_recognition:
            try:
                self.realtime_recognition.stop()
            except:
                pass
    
    def cleanup(self):
        """清理资源"""
        self.stop()
        if self.callback:
            self.callback.on_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="语音识别 - 使用 FunASR 将音频文件转换为文字",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法（使用配置文件中的模型路径）
  python speech_to_text_funasr.py test.wav
  
  # 指定模型路径
  python speech_to_text_funasr.py test.wav --model-path ./weights/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
  
  # 指定配置文件
  python speech_to_text_funasr.py test.wav --config config.yaml
  
  # 保存结果到文件
  python speech_to_text_funasr.py test.wav -o output.txt
  
  # 显示进度条
  python speech_to_text_funasr.py test.wav --show-progress
        """
    )
    
    parser.add_argument(
        "audio_file",
        type=str,
        help="输入的音频文件路径（如 test.wav）"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出文件路径（如果不指定，则输出到控制台）"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="FunASR 模型路径（如果不指定，则从配置文件读取）"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="配置文件路径（yaml格式，包含 voice2txt_model_path 字段）"
    )
    
    parser.add_argument(
        "--batch-size-s",
        type=int,
        default=300,
        help="批处理大小（秒），默认 300"
    )
    
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="显示识别进度条"
    )
    
    parser.add_argument(
        "--vad-model",
        type=str,
        default=None,
        help="VAD 模型路径（可选，None 表示不使用）"
    )
    
    parser.add_argument(
        "--punc-model",
        type=str,
        default=None,
        help="标点模型路径（可选，None 表示不使用）"
    )
    
    args = parser.parse_args()
    
    try:
        # 进行识别
        text = transcribe_audio_funasr(
            audio_path=args.audio_file,
            model_path=args.model_path,
            config_path=args.config,
            batch_size_s=args.batch_size_s,
            show_tqdm=args.show_progress,
            vad_model=args.vad_model,
            punc_model=args.punc_model,
        )
        
        if text is None:
            logger.error("识别失败")
            sys.exit(1)
        
        # 输出结果
        print("\n=== 识别结果 ===")
        print(text)
        
        # 保存到文件
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"\n结果已保存到: {args.output}")
        
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"配置错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"识别失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

