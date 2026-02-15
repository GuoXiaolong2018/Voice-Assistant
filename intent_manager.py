"""
意图管理模块（多角色版本）
支持通过配置文件动态管理多个角色（场景）的意图
每个角色有独立的意图配置和唤醒词
"""

import os
import json
import yaml
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class IntentAction:
    """意图动作定义"""
    type: str  # 动作类型: "tts" (文本转语音播报), "custom" (自定义动作，未来扩展)
    text: str  # 播报文本（当type为tts时使用）
    audio_file: Optional[str] = None  # 音频文件路径（可选，如果提供则直接使用）
    custom_handler: Optional[str] = None  # 自定义处理器（未来扩展）


@dataclass
class Intent:
    """意图定义"""
    tag: str  # 意图tag（唯一标识）
    instruction: str  # 用户指令示例（用于意图识别prompt）
    description: str  # 意图描述（用于意图识别prompt）
    action: IntentAction  # 意图动作


@dataclass
class Role:
    """角色定义"""
    name: str  # 角色名称（场景名称，如 aipaoge, jiayouxia）
    wakeup_word: str  # 唤醒词
    audio_dir: Optional[str] = None  # 音频存储路径（可选，如果为None则使用默认路径）
    intents: Dict[str, Intent] = None  # 意图字典 {tag: Intent}
    
    def __post_init__(self):
        """初始化后处理"""
        if self.intents is None:
            self.intents = {}


class IntentManager:
    """意图管理器（支持多角色）"""
    
    # 角色和唤醒词的默认映射
    DEFAULT_ROLE_WAKEUP_MAP = {
        "aipaoge": "爱跑哥",
        "jiayouxia": "加油侠"
    }
    
    def __init__(
        self,
        main_config_path: str,
        tts_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化意图管理器
        
        参数:
            main_config_path: 主配置文件路径（voice_assistant_config.yaml），所有配置都在这里
            tts_config: TTS配置（用于生成音频）
        """
        self.main_config_path = Path(main_config_path)
        self.tts_config = tts_config or {}
        
        # 存储角色字典 {role_name: Role}
        self.roles: Dict[str, Role] = {}
        
        # 音频文件基础目录
        self.audio_base_dir = Path(main_config_path).parent
        
        # 加载主配置文件以获取角色信息
        self._load_main_config()
        
        # 从主配置文件加载意图配置
        self.load_intents()
    
    def _load_main_config(self):
        """从主配置文件加载角色信息"""
        try:
            with open(self.main_config_path, 'r', encoding='utf-8') as f:
                main_config = yaml.safe_load(f) or {}
            
            # 获取所有场景（角色）
            intent_config = main_config.get("intent", {})
            roles_config = intent_config.get("roles", {})
            
            # 从intent.roles中提取所有角色
            for role_name, role_config in roles_config.items():
                wakeup_word = role_config.get('wakeup_word', self.DEFAULT_ROLE_WAKEUP_MAP.get(role_name, role_name))
                
                # 获取音频目录
                audio_dir = role_config.get('audio_dir')
                if audio_dir:
                    # 如果是相对路径，转换为绝对路径
                    audio_dir_path = Path(audio_dir)
                    if not audio_dir_path.is_absolute():
                        audio_dir_path = self.audio_base_dir / audio_dir_path
                    audio_dir = str(audio_dir_path.resolve())
                else:
                    # 使用默认音频目录
                    audio_dir = str(self.audio_base_dir / f"gen_voice_dashscope_{role_name}")
                
                # 创建角色
                role = Role(
                    name=role_name,
                    wakeup_word=wakeup_word,
                    audio_dir=audio_dir,
                    intents={}
                )
                self.roles[role_name] = role
                    
        except Exception as e:
            logger.warning(f"加载主配置文件失败: {e}，将使用默认角色配置")
            # 创建默认角色
            for role_name, wakeup_word in self.DEFAULT_ROLE_WAKEUP_MAP.items():
                default_audio_dir = str(self.audio_base_dir / f"gen_voice_dashscope_{role_name}")
                self.roles[role_name] = Role(
                    name=role_name,
                    wakeup_word=wakeup_word,
                    audio_dir=default_audio_dir,
                    intents={}
                )
    
    def load_intents(self) -> bool:
        """
        从主配置文件加载所有角色的意图
        
        返回:
            是否加载成功
        """
        try:
            # 从主配置文件加载
            with open(self.main_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            intent_config = config.get('intent', {})
            roles_config = intent_config.get('roles', {})
            
            if not roles_config:
                logger.warning("主配置文件中未找到intent.roles配置，将使用已加载的角色")
                return True
            
            for role_name, role_config in roles_config.items():
                # 确保角色存在
                if role_name not in self.roles:
                    wakeup_word = role_config.get('wakeup_word', self.DEFAULT_ROLE_WAKEUP_MAP.get(role_name, role_name))
                    # 获取音频目录（如果配置中有则使用，否则为None，使用默认路径）
                    audio_dir = role_config.get('audio_dir')  # 可能是None
                    self.roles[role_name] = Role(
                        name=role_name,
                        wakeup_word=wakeup_word,
                        audio_dir=audio_dir,  # 可能是None，表示使用默认路径
                        intents={}
                    )
                
                role = self.roles[role_name]
                
                # 更新唤醒词（如果配置中有）
                if 'wakeup_word' in role_config:
                    role.wakeup_word = role_config['wakeup_word']
                
                # 更新音频目录（如果配置中有）
                if 'audio_dir' in role_config:
                    audio_dir = role_config['audio_dir']
                    if audio_dir:
                        # 如果是相对路径，转换为绝对路径
                        audio_dir_path = Path(audio_dir)
                        if not audio_dir_path.is_absolute():
                            audio_dir_path = self.audio_base_dir / audio_dir_path
                        role.audio_dir = str(audio_dir_path.resolve())
                
                # 加载意图
                intents_config = role_config.get('intents', [])
                role.intents = {}
                
                for intent_data in intents_config:
                    try:
                        intent = self._dict_to_intent(intent_data)
                        # 从map.jsonl中恢复audio_file（如果存在）
                        self._restore_audio_file_from_map(role_name, intent)
                        role.intents[intent.tag] = intent
                    except Exception as e:
                        logger.error(f"加载角色 {role_name} 的意图失败: {intent_data}, 错误: {e}")
                        continue
            
            total_intents = sum(len(role.intents) for role in self.roles.values())
            logger.info(f"成功加载 {len(self.roles)} 个角色的 {total_intents} 个意图")
            return True
        except Exception as e:
            logger.error(f"加载意图配置文件失败: {e}")
            return False
    
    def save_intents(self) -> bool:
        """
        保存所有角色的意图到配置文件
        保存到主配置文件的intent.roles部分
        
        返回:
            是否保存成功
        """
        try:
            # 读取主配置文件
            with open(self.main_config_path, 'r', encoding='utf-8') as f:
                main_config = yaml.safe_load(f) or {}
            
            # 确保intent配置存在
            if 'intent' not in main_config:
                main_config['intent'] = {}
            
            if 'roles' not in main_config['intent']:
                main_config['intent']['roles'] = {}
            
            roles_config = main_config['intent']['roles']
            
            for role_name, role in self.roles.items():
                role_config = {
                    'wakeup_word': role.wakeup_word,
                    'intents': [self._intent_to_dict(intent) for intent in role.intents.values()]
                }
                
                # 保留wozai_audio和check_audio（如果存在）
                if role_name in roles_config:
                    if 'wozai_audio' in roles_config[role_name]:
                        role_config['wozai_audio'] = roles_config[role_name]['wozai_audio']
                    if 'check_audio' in roles_config[role_name]:
                        role_config['check_audio'] = roles_config[role_name]['check_audio']
                
                # 保存音频目录（总是保存，转换为相对路径）
                # 默认路径格式：./gen_voice_dashscope_{role_name}
                default_audio_dir = f"./gen_voice_dashscope_{role_name}"
                if role.audio_dir:
                    audio_dir_path = Path(role.audio_dir)
                    if audio_dir_path.is_absolute():
                        # 转换为相对路径
                        try:
                            relative_path = audio_dir_path.relative_to(self.audio_base_dir)
                            relative_str = f"./{relative_path}"
                            role_config['audio_dir'] = relative_str
                        except ValueError:
                            # 如果无法转换为相对路径，保存绝对路径
                            role_config['audio_dir'] = role.audio_dir
                    else:
                        # 相对路径，直接保存
                        role_config['audio_dir'] = role.audio_dir
                else:
                    # 如果没有设置audio_dir，使用默认路径
                    role_config['audio_dir'] = default_audio_dir
                
                roles_config[role_name] = role_config
            
            # 保存到主配置文件
            with open(self.main_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(main_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            total_intents = sum(len(role.intents) for role in self.roles.values())
            logger.info(f"成功保存 {len(self.roles)} 个角色的 {total_intents} 个意图到配置文件")
            return True
        except Exception as e:
            logger.error(f"保存意图配置文件失败: {e}")
            return False
    
    def add_intent(
        self,
        role_name: str,
        tag: str,
        instruction: str,
        description: str,
        action_text: str,
        action_type: str = "tts",
        audio_file: Optional[str] = None
    ) -> bool:
        """
        为指定角色添加新意图
        
        参数:
            role_name: 角色名称（场景名称，如 "aipaoge", "jiayouxia"）
            tag: 意图tag
            instruction: 用户指令示例
            description: 意图描述
            action_text: 动作文本（播报内容）
            action_type: 动作类型（默认"tts"）
            audio_file: 音频文件路径（可选）
            
        返回:
            是否添加成功
        """
        # 确保角色存在
        if role_name not in self.roles:
            wakeup_word = self.DEFAULT_ROLE_WAKEUP_MAP.get(role_name, role_name)
            # 使用默认音频目录
            default_audio_dir = str(self.audio_base_dir / f"gen_voice_dashscope_{role_name}")
            self.roles[role_name] = Role(
                name=role_name,
                wakeup_word=wakeup_word,
                audio_dir=default_audio_dir,
                intents={}
            )
        
        role = self.roles[role_name]
        
        if tag in role.intents:
            logger.warning(f"角色 {role_name} 的意图 {tag} 已存在，将更新")
            return self.update_intent(role_name, tag, instruction, description, action_text, action_type, audio_file)
        
        action = IntentAction(
            type=action_type,
            text=action_text,
            audio_file=audio_file
        )
        
        intent = Intent(
            tag=tag,
            instruction=instruction,
            description=description,
            action=action
        )
        
        role.intents[tag] = intent
        
        # 确保音频文件存在（会自动从map.jsonl中查找或生成新的）
        if action_type == "tts":
            audio_file_path = self._ensure_audio_file(role_name, intent)
            if audio_file_path:
                # 更新map.jsonl文件（如果是从map.jsonl恢复的，也会更新以确保一致性）
                self._update_map_file(role_name, intent, audio_file_path)
        
        # 保存配置
        if self.save_intents():
            # 更新主配置文件
            self._update_main_config_role(role_name)
            return True
        
        return False
    
    def update_intent(
        self,
        role_name: str,
        tag: str,
        instruction: Optional[str] = None,
        description: Optional[str] = None,
        action_text: Optional[str] = None,
        action_type: Optional[str] = None,
        audio_file: Optional[str] = None
    ) -> bool:
        """
        更新指定角色的现有意图
        
        参数:
            role_name: 角色名称
            tag: 意图tag
            instruction: 用户指令示例（可选）
            description: 意图描述（可选）
            action_text: 动作文本（可选）
            action_type: 动作类型（可选）
            audio_file: 音频文件路径（可选）
            
        返回:
            是否更新成功
        """
        if role_name not in self.roles:
            logger.error(f"角色 {role_name} 不存在")
            return False
        
        role = self.roles[role_name]
        
        if tag not in role.intents:
            logger.error(f"角色 {role_name} 的意图 {tag} 不存在，无法更新")
            return False
        
        intent = role.intents[tag]
        
        # 更新字段
        if instruction is not None:
            intent.instruction = instruction
        if description is not None:
            intent.description = description
        if action_text is not None:
            intent.action.text = action_text
        if action_type is not None:
            intent.action.type = action_type
        if audio_file is not None:
            intent.action.audio_file = audio_file
        
        # 确保音频文件存在（会自动从map.jsonl中查找或生成新的）
        if intent.action.type == "tts":
            audio_file_path = self._ensure_audio_file(role_name, intent)
            if audio_file_path:
                # 更新map.jsonl文件（如果是从map.jsonl恢复的，也会更新以确保一致性）
                self._update_map_file(role_name, intent, audio_file_path)
        
        # 保存配置
        if self.save_intents():
            # 更新主配置文件
            self._update_main_config_role(role_name)
            return True
        
        return False
    
    def delete_intent(self, role_name: str, tag: str) -> bool:
        """
        删除指定角色的意图
        
        参数:
            role_name: 角色名称
            tag: 意图tag
            
        返回:
            是否删除成功
        """
        if role_name not in self.roles:
            logger.warning(f"角色 {role_name} 不存在")
            return False
        
        role = self.roles[role_name]
        
        if tag not in role.intents:
            logger.warning(f"角色 {role_name} 的意图 {tag} 不存在")
            return False
        
        # 删除map.jsonl中的记录
        self._remove_from_map_file(role_name, tag)
        
        del role.intents[tag]
        
        # 保存配置
        if self.save_intents():
            # 更新主配置文件
            self._update_main_config_role(role_name)
            return True
        
        return False
    
    def get_intent(self, role_name: str, tag: str) -> Optional[Intent]:
        """
        获取指定角色的意图
        
        参数:
            role_name: 角色名称
            tag: 意图tag
            
        返回:
            意图对象，如果不存在返回None
        """
        if role_name not in self.roles:
            return None
        return self.roles[role_name].intents.get(tag)
    
    def list_intents(self, role_name: Optional[str] = None) -> List[Tuple[str, Intent]]:
        """
        列出意图
        
        参数:
            role_name: 角色名称，如果为None则列出所有角色的意图
            
        返回:
            [(role_name, intent), ...] 列表
        """
        result = []
        roles_to_list = [role_name] if role_name else self.roles.keys()
        
        for rn in roles_to_list:
            if rn in self.roles:
                role = self.roles[rn]
                for intent in role.intents.values():
                    result.append((rn, intent))
        
        return result
    
    def get_role(self, role_name: str) -> Optional[Role]:
        """
        获取角色
        
        参数:
            role_name: 角色名称
            
        返回:
            角色对象，如果不存在返回None
        """
        return self.roles.get(role_name)
    
    def list_roles(self) -> List[Role]:
        """
        列出所有角色
        
        返回:
            角色列表
        """
        return list(self.roles.values())
    
    def get_intent_dict(self, role_name: str) -> Dict[str, str]:
        """
        获取指定角色的意图字典（用于生成prompt）
        
        参数:
            role_name: 角色名称
            
        返回:
            {tag: description} 字典
        """
        if role_name not in self.roles:
            return {}
        return {intent.tag: intent.description for intent in self.roles[role_name].intents.values()}
    
    def get_audio_files(self, role_name: str) -> Dict[str, str]:
        """
        获取指定角色的音频文件映射（用于主配置文件）
        
        参数:
            role_name: 角色名称
            
        返回:
            {tag: audio_file_path} 字典
        """
        if role_name not in self.roles:
            return {}
        
        audio_files = {}
        role = self.roles[role_name]
        
        for tag, intent in role.intents.items():
            if intent.action.type == "tts" and intent.action.audio_file:
                # 使用相对路径（相对于主配置文件所在目录）
                audio_path = Path(intent.action.audio_file)
                if audio_path.is_absolute():
                    # 转换为相对路径（相对于主配置文件所在目录）
                    try:
                        # 使用主配置文件所在目录作为基准
                        main_config_dir = self.main_config_path.parent.resolve()
                        audio_path_resolved = audio_path.resolve()
                        relative_path = audio_path_resolved.relative_to(main_config_dir)
                        audio_files[tag] = f"./{relative_path}"
                    except ValueError:
                        # 如果无法转换为相对路径，保持绝对路径
                        audio_files[tag] = intent.action.audio_file
                else:
                    # 已经是相对路径，直接使用
                    audio_files[tag] = intent.action.audio_file
        
        return audio_files
    
    def _get_audio_dir(self, role_name: str) -> Path:
        """
        获取角色的音频目录
        
        参数:
            role_name: 角色名称
            
        返回:
            音频目录路径
        """
        role = self.roles.get(role_name)
        if role and role.audio_dir:
            # 如果配置了自定义路径，使用自定义路径
            audio_dir = Path(role.audio_dir)
            if not audio_dir.is_absolute():
                # 相对路径，相对于主配置文件所在目录
                audio_dir = self.audio_base_dir / audio_dir
        else:
            # 使用默认路径
            audio_dir = self.audio_base_dir / f"gen_voice_dashscope_{role_name}"
        
        audio_dir.mkdir(parents=True, exist_ok=True)
        return audio_dir
    
    def _restore_audio_file_from_map(self, role_name: str, intent: Intent) -> bool:
        """
        从map.jsonl中恢复audio_file路径
        
        参数:
            role_name: 角色名称
            intent: 意图对象
            
        返回:
            是否成功恢复
        """
        # 如果已经指定了audio_file且文件存在，不需要恢复
        if intent.action.audio_file and Path(intent.action.audio_file).exists():
            return True
        
        # 从map.jsonl中查找
        map_data = self._load_map_file(role_name)
        if intent.tag in map_data:
            entry = map_data[intent.tag]
            audio_dir = self._get_audio_dir(role_name)
            # filename可能是相对路径或文件名
            filename = entry['filename']
            if Path(filename).is_absolute():
                audio_file_path = Path(filename)
            else:
                audio_file_path = audio_dir / filename
            
            # 检查文件是否存在
            if audio_file_path.exists():
                # 确保使用绝对路径
                intent.action.audio_file = str(audio_file_path.resolve())
                logger.debug(f"从map.jsonl恢复音频文件: {intent.tag} -> {audio_file_path}")
                return True
            else:
                logger.warning(f"map.jsonl中记录的音频文件不存在: {audio_file_path}")
        
        return False
    
    def _ensure_audio_file(self, role_name: str, intent: Intent) -> Optional[str]:
        """
        确保音频文件存在，如果不存在则生成
        优先从map.jsonl中查找，如果不存在则生成新的
        
        参数:
            role_name: 角色名称
            intent: 意图对象
            
        返回:
            音频文件路径，如果生成失败返回None
        """
        # 如果已指定音频文件且存在，直接返回
        if intent.action.audio_file and Path(intent.action.audio_file).exists():
            return intent.action.audio_file
        
        # 尝试从map.jsonl中恢复
        if self._restore_audio_file_from_map(role_name, intent):
            if intent.action.audio_file and Path(intent.action.audio_file).exists():
                return intent.action.audio_file
        
        # 获取音频目录
        audio_dir = self._get_audio_dir(role_name)
        
        # 生成音频文件名（基于tag和文本内容的hash）
        text_hash = hashlib.md5(intent.action.text.encode('utf-8')).hexdigest()[:8]
        audio_filename = f"tts_{intent.tag}_{text_hash}.wav"
        audio_file_path = audio_dir / audio_filename
        
        # 如果文件已存在，直接使用
        if audio_file_path.exists():
            intent.action.audio_file = str(audio_file_path)
            return str(audio_file_path)
        
        # 生成音频文件
        logger.info(f"为角色 {role_name} 的意图 {intent.tag} 生成音频文件: {audio_file_path}")
        
        try:
            from tts_engine import TTSPlayer
            
            # 从配置获取TTS参数
            dashscope_voice = self.tts_config.get('dashscope_voice', 'zhitian')
            dashscope_api_key = self.tts_config.get('dashscope_api_key')
            
            if not dashscope_api_key:
                logger.error("未提供DashScope API密钥，无法生成音频")
                return None
            
            tts_player = TTSPlayer(
                engine='dashscope',
                dashscope_voice=dashscope_voice,
                keep_files=True,
                auto_play=False,
                dashscope_api_key=dashscope_api_key,
                output_dir=str(audio_dir)
            )
            
            # 注意：不在这里设置role.audio_dir，保持用户配置或默认值
            
            result = tts_player.speak(intent.action.text, save_path=str(audio_file_path))
            
            if result and Path(result).exists():
                intent.action.audio_file = str(audio_file_path)
                logger.info(f"✓ 成功生成音频文件: {audio_file_path}")
                return str(audio_file_path)
            else:
                logger.error(f"✗ 生成音频文件失败: {audio_file_path}")
                return None
                
        except Exception as e:
            logger.error(f"生成音频文件时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_map_file_path(self, role_name: str) -> Path:
        """
        获取角色的map.jsonl文件路径
        
        参数:
            role_name: 角色名称
            
        返回:
            map.jsonl文件路径
        """
        audio_dir = self._get_audio_dir(role_name)
        return audio_dir / "map.jsonl"
    
    def _load_map_file(self, role_name: str) -> Dict[str, Dict[str, Any]]:
        """
        加载角色的map.jsonl文件
        
        参数:
            role_name: 角色名称
            
        返回:
            {tag: map_entry} 字典
        """
        map_file = self._get_map_file_path(role_name)
        result = {}
        
        if not map_file.exists():
            return result
        
        try:
            with open(map_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        tag = entry.get('tag')
                        if tag:
                            result[tag] = entry
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析map.jsonl行失败: {line}, 错误: {e}")
                        continue
        except Exception as e:
            logger.error(f"加载map.jsonl文件失败: {e}")
        
        return result
    
    def _save_map_file(self, role_name: str, map_data: Dict[str, Dict[str, Any]]) -> bool:
        """
        保存角色的map.jsonl文件
        
        参数:
            role_name: 角色名称
            map_data: {tag: map_entry} 字典
            
        返回:
            是否保存成功
        """
        map_file = self._get_map_file_path(role_name)
        
        try:
            with open(map_file, 'w', encoding='utf-8') as f:
                for entry in map_data.values():
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            return True
        except Exception as e:
            logger.error(f"保存map.jsonl文件失败: {e}")
            return False
    
    def _update_map_file(self, role_name: str, intent: Intent, audio_file_path: str) -> bool:
        """
        更新角色的map.jsonl文件（添加或更新条目）
        
        参数:
            role_name: 角色名称
            intent: 意图对象
            audio_file_path: 音频文件路径
            
        返回:
            是否更新成功
        """
        # 加载现有map数据
        map_data = self._load_map_file(role_name)
        
        # 获取音频文件名（相对路径或文件名）
        audio_path = Path(audio_file_path)
        audio_dir = self._get_audio_dir(role_name)
        try:
            # 尝试转换为相对路径
            if audio_path.is_absolute():
                audio_filename = audio_path.relative_to(audio_dir)
            else:
                audio_filename = audio_path.name
        except ValueError:
            audio_filename = audio_path.name
        
        # 创建或更新条目
        map_entry = {
            'tag': intent.tag,
            'filename': str(audio_filename),
            'text': intent.action.text,
            'instruction': intent.instruction,
            'description': intent.description
        }
        
        map_data[intent.tag] = map_entry
        
        # 保存map文件
        return self._save_map_file(role_name, map_data)
    
    def _remove_from_map_file(self, role_name: str, tag: str) -> bool:
        """
        从角色的map.jsonl文件中删除条目
        
        参数:
            role_name: 角色名称
            tag: 意图tag
            
        返回:
            是否删除成功
        """
        # 加载现有map数据
        map_data = self._load_map_file(role_name)
        
        # 删除条目
        if tag in map_data:
            del map_data[tag]
            # 保存map文件
            return self._save_map_file(role_name, map_data)
        
        return True
    
    def _update_main_config_role(self, role_name: str) -> bool:
        """
        更新主配置文件中指定角色的意图配置
        
        参数:
            role_name: 角色名称
            
        返回:
            是否更新成功
        """
        try:
            # 读取主配置文件
            with open(self.main_config_path, 'r', encoding='utf-8') as f:
                main_config = yaml.safe_load(f) or {}
            
            # 确保intent配置存在
            if 'intent' not in main_config:
                main_config['intent'] = {}
            
            # 删除旧的空条目（如果存在）
            # 这些是旧的配置结构，现在所有配置都在intent.roles下
            if 'aipaoge' in main_config['intent'] and main_config['intent']['aipaoge'] == {}:
                del main_config['intent']['aipaoge']
            if 'jiayouxia' in main_config['intent'] and main_config['intent']['jiayouxia'] == {}:
                del main_config['intent']['jiayouxia']
            
            # 不再创建或更新 intent.{role_name}，因为所有配置都在 intent.roles 下
            # 这个方法现在只负责清理旧的空条目
            
            # 保存主配置文件
            with open(self.main_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(main_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            logger.info(f"✓ 成功更新主配置文件中角色 {role_name} 的配置")
            return True
            
        except Exception as e:
            logger.error(f"更新主配置文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _intent_to_dict(self, intent: Intent) -> Dict[str, Any]:
        """
        将Intent对象转换为字典
        注意：audio_file字段不保存，因为可以从map.jsonl中自动恢复
        """
        action_dict = {
            'type': intent.action.type,
            'text': intent.action.text
        }
        # 只有当audio_file不是从map.jsonl自动恢复的，或者是用户明确指定的自定义路径时才保存
        # 为了简化，我们不再保存audio_file，因为可以从map.jsonl中自动恢复
        # 如果用户有特殊需求，可以在配置文件中手动指定
        
        return {
            'tag': intent.tag,
            'instruction': intent.instruction,
            'description': intent.description,
            'action': action_dict
        }
    
    def _dict_to_intent(self, data: Dict[str, Any]) -> Intent:
        """将字典转换为Intent对象"""
        action_data = data.get('action', {})
        action = IntentAction(
            type=action_data.get('type', 'tts'),
            text=action_data.get('text', ''),
            audio_file=action_data.get('audio_file')
        )
        
        return Intent(
            tag=data['tag'],
            instruction=data['instruction'],
            description=data['description'],
            action=action
        )
