#!/usr/bin/env python3
"""
使用 USB 麦克风录制音频并保存的示例脚本
"""

import sys
import os
from pathlib import Path

# 添加 voice_control 目录到路径
current_dir = Path(__file__).parent
voice_control_path = current_dir / "voice_control"
sys.path.insert(0, str(voice_control_path))

from audio_device import AudioStream, AudioConfig
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_audio_devices(verbose: bool = True):
    """列出所有可用的音频输入设备"""
    devices = AudioStream.list_input_devices()
    if verbose:
        print("\n=== 可用的音频输入设备 ===")
        for dev in devices:
            print(f"  [{dev['index']}] {dev['name']} (采样率: {dev['sample_rate']} Hz)")
        print()
    return devices


def record_audio(output_path: str = None, duration: float = None, verbose: bool = True):
    """
    录制音频并保存
    
    参数:
        output_path: 输出文件路径（如果为 None，将自动生成）
        duration: 录制时长（秒），如果为 None，将录制到检测到静音为止
        verbose: 是否显示详细信息（默认 True）
    """
    # 列出设备
    devices = list_audio_devices(verbose=verbose)
    
    # 自动选择最佳设备（优先选择 USB 麦克风）
    device_index = AudioStream.select_best_device(target_sample_rate=16000)
    if device_index is not None and verbose:
        device_info = next((d for d in devices if d['index'] == device_index), None)
        if device_info:
            print(f"已选择设备: [{device_index}] {device_info['name']}")
    elif device_index is None and verbose:
        print("使用默认设备")
    
    # # 配置音频参数
    config = AudioConfig(
        sample_rate=16000,
        frame_length=1280,
        device_index=device_index,
        silence_threshold=1000.0,  # 静音阈值 #500 
        max_phrase_duration=6.0 if duration is None else duration + 5.0,  # 最大录制时长
        silence_duration=2,  # 静音持续时间（秒）
        wait_for_silence_timeout=1.0,
        skip_initial_frames=1,
        required_silent_frames=1,
        no_speech_timeout=6.0,  # 无语音超时
        adaptive_silence_threshold=True,
    )

    # 创建音频流
    audio_stream = AudioStream(config)
    
    try:
        # 打开音频流
        if verbose:
            print("\n正在打开音频流...")
        audio_stream.open()
        if verbose:
            print("音频流已打开，准备录制...")
        
        if duration is None:
            if verbose:
                print("\n请开始说话（检测到静音后自动停止）...")
                print("提示: 说话后保持静音 1 秒以上将自动停止录制")
        else:
            if verbose:
                print(f"\n开始录制 {duration} 秒...")
        
        # 录制音频
        audio_path = audio_stream.record_phrase(wait_for_silence=True)
        
        if audio_path:
            # 如果指定了输出路径，复制文件
            if output_path:
                import shutil
                shutil.copy(audio_path, output_path)
                if verbose:
                    print(f"\n✓ 音频已保存到: {output_path}")
                # 删除临时文件
                try:
                    os.unlink(audio_path)
                except:
                    pass
            else:
                if verbose:
                    print(f"\n✓ 音频已保存到: {audio_path}")
                return audio_path
        else:
            if verbose:
                print("\n✗ 录制失败：未检测到音频")
            return None
            
    except KeyboardInterrupt:
        print("\n\n录制被用户中断")
        return None
    except Exception as e:
        print(f"\n✗ 录制出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 关闭音频流
        audio_stream.terminate()
        print("音频流已关闭")


def record_with_duration(output_path: str, duration: float, verbose: bool = True):
    """
    录制指定时长的音频
    
    参数:
        output_path: 输出文件路径
        duration: 录制时长（秒）
        verbose: 是否显示详细信息（默认 False）
    """
    import time
    import wave
    import pyaudio
    
    # 列出设备（不打印信息）
    devices = list_audio_devices(verbose=verbose)
    
    # 自动选择最佳设备
    device_index = AudioStream.select_best_device(target_sample_rate=16000)
    if device_index is not None and verbose:
        device_info = next((d for d in devices if d['index'] == device_index), None)
        if device_info:
            print(f"已选择设备: [{device_index}] {device_info['name']}")
    
    # 音频参数
    sample_rate = 16000
    channels = 1
    chunk = 1280
    format = pyaudio.paInt16
    
    # 初始化 PyAudio
    pa = pyaudio.PyAudio()
    stream = None
    
    try:
        # 打开音频流
        stream = pa.open(
            format=format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk,
            input_device_index=device_index,
        )
        
        if verbose:
            print(f"\n开始录制 {duration} 秒...")
            print("录制中...")
        
        frames = []
        num_frames = int(sample_rate / chunk * duration)
        
        for i in range(num_frames):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            # 显示进度
            if verbose and (i + 1) % 10 == 0:
                progress = (i + 1) / num_frames * 100
                print(f"进度: {progress:.1f}%", end='\r')
        
        if verbose:
            print("\n录制完成，正在保存...")
        
        # 保存为 WAV 文件
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(pa.get_sample_size(format))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
        
        if verbose:
            print(f"✓ 音频已保存到: {output_path}")
        
    except KeyboardInterrupt:
        print("\n\n录制被用户中断")
    except Exception as e:
        print(f"\n✗ 录制出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        pa.terminate()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用 USB 麦克风录制音频")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出文件路径（默认：自动生成）"
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=None,
        help="录制时长（秒），如果不指定则录制到检测到静音为止"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="仅列出可用的音频设备"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_audio_devices()
    elif args.duration:
        # 使用固定时长录制
        output_path = args.output or f"recorded_audio_{int(args.duration)}s.wav"
        record_with_duration(output_path, args.duration)
    else:
        # 使用自动检测静音的方式录制
        output_path = args.output
        record_audio(output_path)

