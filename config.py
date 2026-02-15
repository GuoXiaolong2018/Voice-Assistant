"""
配置加载模块
从YAML文件加载配置
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    参数:
        config_path: 配置文件路径，如果为None则使用默认路径
        
    返回:
        配置字典
    """
    if config_path is None:
        # 默认配置文件路径（相对于脚本所在目录）
        current_dir = Path(__file__).parent
        config_path = current_dir / "voice_assistant_config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 设置当前目录路径（如果配置中使用相对路径）
        if 'paths' in config:
            config['paths']['current_dir'] = str(Path(__file__).parent)
        
        logger.info(f"✓ 配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    从配置字典中获取值（支持点号分隔的路径）
    
    参数:
        config: 配置字典
        key_path: 键路径，例如 "basic.wakeup_word"
        default: 默认值
        
    返回:
        配置值
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

