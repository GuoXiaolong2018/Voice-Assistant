#!/usr/bin/env python3
"""
集成意图管理器的语音助手示例
展示如何在现有语音助手中集成意图管理器
"""

# 这个文件展示了如何修改 voice_assistant.py 来集成意图管理器
# 实际使用时，可以将这些修改应用到 voice_assistant.py

# 在 voice_assistant.py 的 _init_components 方法中添加以下代码：

"""
from intent_manager import IntentManager

def _init_components(self):
    # ... 现有代码 ...
    
    # ========== 集成意图管理器（可选） ==========
    intent_config_path = current_dir / "intent_config.yaml"
    intent_manager = None
    
    if intent_config_path.exists():
        logger.info(f"发现意图配置文件: {intent_config_path}，启用动态意图管理")
        tts_config = {
            "dashscope_voice": self.config.get("tts", {}).get("dashscope_voice", "zhitian"),
            "dashscope_api_key": self.config.get("api_keys", {}).get("dashscope_api_key", "")
        }
        try:
            intent_manager = IntentManager(
                config_path=str(intent_config_path),
                main_config_path=str(current_dir / "voice_assistant_config.yaml"),
                scenario=self.config.get("intent", {}).get("scenario", "aipaoge"),
                tts_config=tts_config
            )
            logger.info("✓ 意图管理器初始化成功")
        except Exception as e:
            logger.warning(f"意图管理器初始化失败: {e}，将使用配置文件中的意图")
    else:
        logger.info("未发现意图配置文件，使用默认意图配置")
    # ========== 意图管理器集成结束 ==========
    
    # 初始化意图处理器（传入意图管理器）
    self.intent_handler = IntentHandler(self.config, self.qwen_client, intent_manager=intent_manager)
    
    # ... 其他现有代码 ...
"""

print("这是一个示例文件，展示了如何集成意图管理器到 voice_assistant.py")
print("请参考 INTENT_MANAGER_README.md 了解详细使用方法")

