import os
import json
from typing import List, Tuple, Optional
from agentverse.message import Message

def get_translation_output(setting: Optional[str] = None, messages: Optional[List[Message]] = None, agent_nums: Optional[int] = None) -> Tuple[List[dict], List[dict]]:
    """
    获取翻译输出结果
    
    Args:
        setting: 设置类型
        messages: 消息列表
        agent_nums: agent数量
        
    Returns:
        Tuple[List[dict], List[dict]]: (chat_history, translations)
        - chat_history: 整理好的历史消息
        - translations: 每个agent最后一次发言的翻译结果
    """
    
    chat_history = []
    translations = []
    
    agent_type_mapping = {
        "LP translator": "LP",
        "FOL translator": "FOL", 
        "SAT translator": "SAT"
    }
    
    if setting == "every_agent" and messages is not None and agent_nums is not None:
        for message in messages:
            chat_history.append({
                "role": message.sender,
                "content": message.content
            })
        
        # 获取每个agent的最后一次发言（最后agent_nums条消息）
        last_messages = messages[-agent_nums:]
        
        # 创建一个合并的对象包含所有三个字段
        single_translation = {}
        for i, message in enumerate(last_messages):
            agent_type = agent_type_mapping.get(message.sender, f"translation_{i}")
            single_translation[agent_type] = message.content.strip()
        
        translations.append(single_translation)

    return chat_history, translations 