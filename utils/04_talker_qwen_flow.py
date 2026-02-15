from tts_interface_qwen import TTSPlayer

# 创建播放器并播报
ts_player = TTSPlayer(
    engine='dashscope',  # 明确指定使用千问TTS
    dashscope_voice='zhitian',  # 可选：设置音色，zhiyan/zhitian/zhizhe
    keep_files=False,
    auto_play=True,
    dashscope_api_key="sk-3ecf3fdfbf734c06a536bbe7d841054d"
)
ts_player.speak("谢谢您对易思汀啤酒的喜爱。")



