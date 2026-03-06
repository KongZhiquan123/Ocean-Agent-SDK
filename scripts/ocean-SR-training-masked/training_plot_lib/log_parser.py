"""
@file log_parser.py

@description Parse structured training logs and extract event data.
@author kongzhiquan
@date 2026-02-09
@version 1.0.0

@changelog
    - 2026-02-09 kongzhiquan: v1.0.0 extracted from generate_training_plots.py
"""

import os
import re
import json
from typing import Dict, Any


def parse_structured_log(log_path: str) -> Dict[str, Any]:
    """解析结构化日志并提取训练数据"""
    result = {
        'training_start': None,
        'training_end': None,
        'epoch_train': [],
        'epoch_valid': [],
        'final_valid': None,
        'final_test': None,
    }

    if not os.path.exists(log_path):
        return result

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return result

    event_pattern = re.compile(r'__event__(\{.*?\})__event__')
    matches = event_pattern.findall(content)

    for match in matches:
        try:
            event = json.loads(match)
            event_type = event.get('event')

            if event_type == 'training_start':
                result['training_start'] = event
            elif event_type == 'training_end':
                result['training_end'] = event
            elif event_type == 'epoch_train':
                result['epoch_train'].append(event)
            elif event_type == 'epoch_valid':
                result['epoch_valid'].append(event)
            elif event_type == 'final_valid':
                result['final_valid'] = event
            elif event_type == 'final_test':
                result['final_test'] = event
        except json.JSONDecodeError:
            continue

    return result
