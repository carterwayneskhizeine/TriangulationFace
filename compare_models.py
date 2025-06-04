#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型比较工具
比较对称对齐后的模型与canonical_face_model.obj的差异
"""

import numpy as np
import os

def load_vertices_from_obj(obj_path):
    """从OBJ文件加载顶点"""
    vertices = []
    with open(obj_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return np.array(vertices)

def analyze_model_properties(vertices, model_name):
    """分析模型属性"""
    print(f"\n=== {model_name} 分析 ===")
    
    # 基本统计
    print(f"顶点数量: {len(vertices)}")
    print(f"X坐标范围: [{vertices[:, 0].min():.6f}, {vertices[:, 0].max():.6f}]")
    print(f"Y坐标范围: [{vertices[:, 1].min():.6f}, {vertices[:, 1].max():.6f}]")
    print(f"Z坐标范围: [{vertices[:, 2].min():.6f}, {vertices[:, 2].max():.6f}]")
    
    # 中心点
    center = np.mean(vertices, axis=0)
    print(f"几何中心: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
    
    # X坐标为0的点数量
    zero_x_count = np.sum(np.abs(vertices[:, 0]) < 1e-6)
    print(f"X坐标接近0的点数量: {zero_x_count}")
    
    # 对称性分析（检查是否有对称点）
    symmetric_pairs = 0
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            v1, v2 = vertices[i], vertices[j]
            # 检查是否X坐标相反，Y和Z坐标相近
            if (abs(v1[0] + v2[0]) < 0.01 and 
                abs(v1[1] - v2[1]) < 0.01 and 
                abs(v1[2] - v2[2]) < 0.01 and
                abs(v1[0]) > 0.01):  # 不是中线点
                symmetric_pairs += 1
                if symmetric_pairs <= 5:  # 只显示前5对
                    print(f"对称点对 {symmetric_pairs}: ({v1[0]:.3f}, {v1[1]:.3f}, {v1[2]:.3f}) ↔ ({v2[0]:.3f}, {v2[1]:.3f}, {v2[2]:.3f})")
    
    print(f"找到对称点对数量: {symmetric_pairs}")
    
    return {
        'center': center,
        'zero_x_count': zero_x_count,
        'symmetric_pairs': symmetric_pairs,
        'x_range': (vertices[:, 0].min(), vertices[:, 0].max()),
        'y_range': (vertices[:, 1].min(), vertices[:, 1].max()),
        'z_range': (vertices[:, 2].min(), vertices[:, 2].max())
    }

def compare_models(vertices1, vertices2, name1, name2):
    """比较两个模型"""
    print(f"\n=== {name1} vs {name2} 比较 ===")
    
    if len(vertices1) != len(vertices2):
        print(f"警告：顶点数量不同 ({len(vertices1)} vs {len(vertices2)})")
        min_len = min(len(vertices1), len(vertices2))
        vertices1 = vertices1[:min_len]
        vertices2 = vertices2[:min_len]
    
    # 计算点对点距离
    distances = np.linalg.norm(vertices1 - vertices2, axis=1)
    
    print(f"平均点距离: {np.mean(distances):.6f}")
    print(f"最大点距离: {np.max(distances):.6f}")
    print(f"最小点距离: {np.min(distances):.6f}")
    print(f"距离标准差: {np.std(distances):.6f}")
    
    # 找到差异最大的点
    max_diff_idx = np.argmax(distances)
    print(f"差异最大的点 (索引{max_diff_idx}):")
    print(f"  {name1}: ({vertices1[max_diff_idx, 0]:.6f}, {vertices1[max_diff_idx, 1]:.6f}, {vertices1[max_diff_idx, 2]:.6f})")
    print(f"  {name2}: ({vertices2[max_diff_idx, 0]:.6f}, {vertices2[max_diff_idx, 1]:.6f}, {vertices2[max_diff_idx, 2]:.6f})")
    print(f"  距离: {distances[max_diff_idx]:.6f}")
    
    # 分析前10个点的差异（通常是重要的landmark点）
    print(f"\n前10个关键点比较:")
    for i in range(min(10, len(vertices1))):
        v1, v2 = vertices1[i], vertices2[i]
        dist = distances[i]
        print(f"点{i+1:2d}: 距离={dist:.6f}, {name1}=({v1[0]:7.3f},{v1[1]:7.3f},{v1[2]:7.3f}), {name2}=({v2[0]:7.3f},{v2[1]:7.3f},{v2[2]:7.3f})")

def main():
    """主函数"""
    print("=== 模型对比分析工具 ===")
    
    # 文件路径
    canonical_path = "obj/canonical_face_model.obj"
    aligned_path = "result_file/averaged_landmarks_1749043292_face_model_precise_aligned.obj"
    
    if not os.path.exists(canonical_path):
        print(f"错误：找不到文件 {canonical_path}")
        return
        
    if not os.path.exists(aligned_path):
        print(f"错误：找不到文件 {aligned_path}")
        return
    
    # 加载模型
    print("正在加载模型...")
    canonical_vertices = load_vertices_from_obj(canonical_path)
    aligned_vertices = load_vertices_from_obj(aligned_path)
    
    # 分析各个模型
    canonical_props = analyze_model_properties(canonical_vertices, "Canonical Face Model")
    aligned_props = analyze_model_properties(aligned_vertices, "对称对齐后的模型")
    
    # 比较模型
    compare_models(canonical_vertices, aligned_vertices, "Canonical", "对齐后")
    
    # 总结
    print(f"\n=== 总结 ===")
    print(f"对称对齐是否改善了对称性：")
    print(f"  Canonical模型对称点对: {canonical_props['symmetric_pairs']}")
    print(f"  对齐后模型对称点对: {aligned_props['symmetric_pairs']}")
    
    print(f"中线点分布：")
    print(f"  Canonical模型X=0点数: {canonical_props['zero_x_count']}")
    print(f"  对齐后模型X=0点数: {aligned_props['zero_x_count']}")
    
    # 判断对齐质量
    center_distance = np.linalg.norm(canonical_props['center'] - aligned_props['center'])
    print(f"几何中心距离: {center_distance:.6f}")
    
    if center_distance < 1.0 and aligned_props['zero_x_count'] >= canonical_props['zero_x_count']:
        print("✅ 对称对齐效果良好！")
    else:
        print("⚠️  对称对齐可能需要进一步优化")

if __name__ == "__main__":
    main() 