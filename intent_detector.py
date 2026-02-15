"""
意图识别和处理模块
"""

import logging
import json
from typing import Dict, Any, Tuple, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class IntentHandler:
    """意图识别处理器"""
    
    def __init__(self, config: Dict[str, Any], qwen_client: OpenAI, intent_manager=None):
        """
        初始化意图识别处理器
        
        参数:
            config: 配置字典
            qwen_client: 千问客户端
            intent_manager: 意图管理器（可选），如果提供则使用动态意图管理
        """
        self.config = config
        self.qwen_client = qwen_client
        self.intent_manager = intent_manager
        self.scenario = config.get("intent", {}).get("scenario", "aipaoge")
        self.intent_config = config.get("intent", {}).get(self.scenario, {})
        
        # 如果提供了意图管理器，使用动态加载的意图
        if self.intent_manager:
            # 获取当前场景的意图字典和音频文件
            self.intent_dict = self.intent_manager.get_intent_dict(self.scenario)
            self.audio_files = self.intent_manager.get_audio_files(self.scenario)
        else:
            # 否则使用配置文件中的意图
            self.intent_dict = self.intent_config.get("intent_dict", {})
            self.audio_files = self.intent_config.get("audio_files", {})
        
        # 如果使用意图管理器，从角色的配置中获取wozai_audio和check_audio
        if self.intent_manager:
            # 从主配置文件的intent.roles中获取
            main_intent_config = config.get("intent", {})
            roles_config = main_intent_config.get("roles", {})
            current_role_config = roles_config.get(self.scenario, {})
            self.wozai_audio = current_role_config.get("wozai_audio", "")
            self.check_audio = current_role_config.get("check_audio", "")
        else:
            # 否则从旧的配置结构获取
            self.wozai_audio = self.intent_config.get("wozai_audio", "")
            self.check_audio = self.intent_config.get("check_audio", "")
        self.model = config.get("intent", {}).get("model", "tongyi-intent-detect-v3")
        
        # 如果使用意图管理器，需要重新加载意图字典（因为可能被更新）
        if self.intent_manager:
            self._refresh_intent_dict()
    
    def _refresh_intent_dict(self):
        """刷新意图字典（从意图管理器重新加载）"""
        if self.intent_manager:
            self.intent_dict = self.intent_manager.get_intent_dict(self.scenario)
            self.audio_files = self.intent_manager.get_audio_files(self.scenario)
    
    def get_system_prompt(self) -> str:
        """生成意图识别的系统提示词"""
        intent_dict_str = json.dumps(self.intent_dict, indent=4, ensure_ascii=False)
        
        system_prompt = f"""
你是一个强大的意图识别机器人，你需要根据用户的输入，精准地判断用户的意图。

你需要从以下意图列表中，选择一个主意图(primary_intent)和一个子意图(sub_intent)。

# 意图列表
{intent_dict_str}

# 注意事项
1.  **精准判断**: 用户的意图只有在明确提到"天气"或"油价"时，才应被分类为"今日天气情况(E)"或"今日油价情况(F)"。对于其他比较模糊的、与出行或日常信息相关的问题，应优先考虑"知识库问答(B)"。
2.  **知识库优先**: 对于无法明确归类到其他特定意图（如地图、打招呼、机器人拿东西）的日常问题、信息查询、建议请求等，都应归类到"知识库问答(B)"。
3.  **严格遵守格式**: 你的输出必须是一个可以被Python的json.loads()函数解析的JSON字符串，格式如下：
    {{
        "primary_intent": "主意图代码",
        "sub_intent": "子意图代码",
        "unsupported_item": "不支持的物品名称（如果有的话）"
    }}

# 示例
- 用户输入: "你好啊" -> 输出: {{"primary_intent": "C", "sub_intent": "C", "unsupported_item": ""}}
- 用户输入: "你叫什么名字" -> 输出: {{"primary_intent": "H", "sub_intent": "H", "unsupported_item": ""}}
- 用户输入: "今天天气怎么样？" -> 输出: {{"primary_intent": "E", "sub_intent": "E", "unsupported_item": ""}}
- 用户输入: "92号汽油多少钱？" -> 输出: {{"primary_intent": "F", "sub_intent": "F", "unsupported_item": ""}}
- 用户输入: "帮我查一下距离天安门广场最近的加油站" -> 输出: {{"primary_intent": "A", "sub_intent": "A", "unsupported_item": "", "address": "天安门广场"}}
- 用户输入: "今天出门开车有什么要注意的吗？" -> 输出: {{"primary_intent": "B", "sub_intent": "B", "unsupported_item": ""}}
"""
        
        # 根据场景添加特定示例
        if self.scenario == "aipaoge":
            system_prompt += """
- 用户输入: "给我拿一瓶水" -> 输出: {"primary_intent": "D1", "sub_intent": "D1", "unsupported_item": ""}
- 用户输入: "帮我拿个啤酒" -> 输出: {"primary_intent": "D_unsupported", "sub_intent": "D_unsupported", "unsupported_item": "啤酒"}
- 用户输入: "中国石化的董事长是谁？" -> 输出: {"primary_intent": "B", "sub_intent": "B", "unsupported_item": ""}
"""
        elif self.scenario == "jiayouxia":
            system_prompt += """
- 用户输入: "去给大家拿咖啡" -> 输出: {"primary_intent": "D1", "sub_intent": "D1", "unsupported_item": ""}
- 用户输入: "给我一杯啤酒" -> 输出: {"primary_intent": "D2", "sub_intent": "D2", "unsupported_item": ""}
- 用户输入: "谢谢你的咖啡" -> 输出: {"primary_intent": "K1", "sub_intent": "K1", "unsupported_item": ""}
- 用户输入: "谢谢你的啤酒" -> 输出: {"primary_intent": "K2", "sub_intent": "K2", "unsupported_item": ""}
- 用户输入: "帮我拿一瓶水" -> 输出: {"primary_intent": "D_unsupported", "sub_intent": "D_unsupported", "unsupported_item": "水"}
- 用户输入: "中国石化的董事长是谁？" -> 输出: {"primary_intent": "B", "sub_intent": "B", "unsupported_item": ""}
"""
        
        return system_prompt
    
    def recognize_intent(self, instruction: str) -> Tuple[str, Dict[str, Any]]:
        """
        识别用户意图
        
        参数:
            instruction: 用户指令文本
            
        返回:
            Tuple[意图tag, 意图数据字典]
        """
        system_prompt = self.get_system_prompt()
        
        try:
            completion = self.qwen_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.0,
                stream=False,
            )
            
            if completion.choices:
                intent_tag_str = completion.choices[0].message.content.strip()
                logger.info(f"意图识别模型({self.model})返回: {intent_tag_str}")
                
                # 解析返回的JSON字符串
                if intent_tag_str[-1] != "}":
                    intent_tag_str = intent_tag_str + "}"
                intent_data = json.loads(intent_tag_str)
                intent_tag = intent_data.get("primary_intent", "")
                
                return intent_tag, intent_data
        except Exception as e:
            logger.error(f"调用意图识别模型失败: {e}. 将默认作为知识库问答处理。")
            return "B", {"primary_intent": "B", "sub_intent": "B", "unsupported_item": ""}
    
    def get_audio_file(self, intent_tag: str) -> Optional[str]:
        """
        获取意图对应的音频文件路径
        
        参数:
            intent_tag: 意图tag
            
        返回:
            音频文件路径，如果不存在则返回None
        """
        return self.audio_files.get(intent_tag)
    
    def is_predefined_intent(self, intent_tag: str) -> bool:
        """
        判断是否是预定义意图（有对应的音频文件）
        
        参数:
            intent_tag: 意图tag
            
        返回:
            是否是预定义意图
        """
        return intent_tag in self.audio_files

