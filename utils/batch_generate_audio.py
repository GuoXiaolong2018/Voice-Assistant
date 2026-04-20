#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量生成音频文件脚本
基于 04_talker_qwen_flow.py 的实现方式
"""

import os
import sys
from pathlib import Path

# 添加父目录到路径，以便导入 tts_engine
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# 尝试从 old_version 导入，如果不存在则从当前目录导入
try:
    from old_version.utils.tts_interface_qwen import TTSPlayer
except ImportError:
    try:
        from tts_engine import TTSPlayer
    except ImportError:
        print("错误: 无法导入 TTSPlayer，请检查文件路径")
        sys.exit(1)


def parse_map_file(map_file_path):
    """
    解析 map.txt 文件
    
    参数:
        map_file_path: map.txt 文件路径
        
    返回:
        list: [(文件名, 文本内容), ...]
    """
    audio_list = []
    map_path = Path(map_file_path)
    
    if not map_path.exists():
        print(f"错误: map.txt 文件不存在: {map_file_path}")
        return audio_list
    
    with open(map_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 解析格式: 文件名->文本内容
            if '->' in line:
                parts = line.split('->', 1)
                if len(parts) == 2:
                    filename = parts[0].strip()
                    text = parts[1].strip()
                    if filename and text:
                        audio_list.append((filename, text))
                    else:
                        print(f"警告: 第 {line_num} 行格式不正确，跳过: {line}")
                else:
                    print(f"警告: 第 {line_num} 行格式不正确，跳过: {line}")
            else:
                print(f"警告: 第 {line_num} 行缺少 '->' 分隔符，跳过: {line}")
    
    return audio_list


def generate_audio_for_directory(output_dir, map_file_path, dashscope_voice='zhitian', dashscope_api_key=None):
    """
    为指定目录生成音频文件
    
    参数:
        output_dir: 输出目录
        map_file_path: map.txt 文件路径
        dashscope_voice: DashScope 音色
        dashscope_api_key: DashScope API 密钥
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 解析 map.txt
    audio_list = parse_map_file(map_file_path)
    
    if not audio_list:
        print(f"警告: {map_file_path} 中没有找到有效的音频条目")
        return
    
    print(f"\n开始为 {output_dir} 生成音频文件...")
    print(f"共 {len(audio_list)} 个音频文件需要生成\n")
    
    # 创建 TTSPlayer 实例
    # 注意：auto_play=False 表示不自动播放，keep_files=True 表示保留文件
    tts_player = TTSPlayer(
        engine='dashscope',
        dashscope_voice=dashscope_voice,
        keep_files=True,  # 保留生成的文件
        auto_play=False,  # 批量生成时不播放
        dashscope_api_key=dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY"),
        output_dir=str(output_path)
    )
    
    success_count = 0
    fail_count = 0
    
    for idx, (filename, text) in enumerate(audio_list, 1):
        print(f"[{idx}/{len(audio_list)}] 生成: {filename}")
        print(f"  文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # 构建完整的输出文件路径
        output_file = output_path / filename
        
        # 生成音频文件
        try:
            result = tts_player.speak(text, save_path=str(output_file))
            
            if result and Path(result).exists():
                print(f"  ✓ 成功生成: {output_file}")
                success_count += 1
            else:
                print(f"  ✗ 生成失败: {filename}")
                fail_count += 1
        except Exception as e:
            print(f"  ✗ 生成异常: {filename}, 错误: {e}")
            fail_count += 1
        
        print()  # 空行分隔
    
    print(f"\n生成完成!")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")


def main():
    """主函数"""
    # 项目根目录（utils 的父目录）
    base_dir = Path(__file__).parent.parent
    
    # 两个目录的配置
    configs = [
        {
            'output_dir': base_dir / 'gen_voice_dashscope_aipaoge',
            'map_file': base_dir / 'gen_voice_dashscope_aipaoge' / 'map.txt',
            'voice': 'zhitian'  # 可以根据需要修改音色
        },
        {
            'output_dir': base_dir / 'gen_voice_dashscope_jiayouxia',
            'map_file': base_dir / 'gen_voice_dashscope_jiayouxia' / 'map.txt',
            'voice': 'zhitian'  # 可以根据需要修改音色
        }
    ]
    
    # API 密钥：请设置环境变量 DASHSCOPE_API_KEY
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)
    
    # 为每个目录生成音频
    for config in configs:
        generate_audio_for_directory(
            output_dir=str(config['output_dir']),
            map_file_path=str(config['map_file']),
            dashscope_voice=config['voice'],
            dashscope_api_key=api_key
        )
        print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()

