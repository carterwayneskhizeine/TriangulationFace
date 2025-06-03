#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON输出格式示例
展示人脸landmarks的JSON数据结构
"""

import json
import os

def create_sample_json():
    """创建一个示例JSON文件展示数据格式"""
    
    # 示例landmarks数据（这里只展示前5个点作为示例）
    sample_landmarks = []
    for i in range(5):  # 实际会有468个点
        sample_landmarks.append({
            "id": i,
            "x": round(0.5 + i * 0.001, 6),  # 归一化坐标 (0-1)
            "y": round(0.5 + i * 0.001, 6),  # 归一化坐标 (0-1)  
            "z": round(-0.02 + i * 0.001, 6)  # 深度坐标
        })
    
    # JSON数据结构
    json_data = {
        "frame": 1,
        "timestamp_ms": 33,
        "total_landmarks": len(sample_landmarks),
        "landmarks": sample_landmarks
    }
    
    # 创建jsonfile文件夹（如果不存在）
    json_dir = "jsonfile"
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    
    # 保存示例JSON文件
    sample_file = os.path.join(json_dir, "example_1.json")
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"示例JSON文件已创建: {sample_file}")
    print("JSON数据格式:")
    print(json.dumps(json_data, ensure_ascii=False, indent=2))
    
    return json_data

if __name__ == "__main__":
    create_sample_json() 