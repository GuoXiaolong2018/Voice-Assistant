import os

from tts_interface_qwen import TTSPlayer

# 创建播放器并播报（密钥：环境变量 DASHSCOPE_API_KEY）
ts_player = TTSPlayer(
    engine='dashscope',  # 明确指定使用千问TTS
    dashscope_voice='zhitian',  # 可选：设置音色，zhiyan/zhitian/zhizhe
    keep_files=False,
    auto_play=True,
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "")
)
ts_player.speak("谢谢您对易思汀啤酒的喜爱。")



