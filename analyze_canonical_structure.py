#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入分析canonical_face_model.obj的结构
找出对称规律、中线点和几何特征，为正确的居中算法提供依据
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def find_exact_symmetric_pairs(vertices, tolerance=0.01):
    """找出精确的对称点对"""
    symmetric_pairs = []
    used_indices = set()
    
    for i in range(len(vertices)):
        if i in used_indices:
            continue
            
        v1 = vertices[i]
        
        # 跳过X坐标接近0的点（这些是中线点）
        if abs(v1[0]) < tolerance:
            continue
            
        # 寻找对称点：X坐标相反，Y和Z坐标相同
        for j in range(i+1, len(vertices)):
            if j in used_indices:
                continue
                
            v2 = vertices[j]
            
            # 检查是否为对称点对
            if (abs(v1[0] + v2[0]) < tolerance and 
                abs(v1[1] - v2[1]) < tolerance and 
                abs(v1[2] - v2[2]) < tolerance):
                
                symmetric_pairs.append((i, j, v1, v2))
                used_indices.add(i)
                used_indices.add(j)
                break
    
    return symmetric_pairs

def find_centerline_points(vertices, tolerance=1e-5):
    """找出中线点（X坐标接近0的点）"""
    centerline_points = []
    
    for i, vertex in enumerate(vertices):
        if abs(vertex[0]) < tolerance:
            centerline_points.append((i, vertex))
    
    return centerline_points

def analyze_face_regions(vertices):
    """分析面部不同区域的点分布"""
    print("=== 面部区域分析 ===")
    
    # 按Y坐标分区域
    y_coords = vertices[:, 1]
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # 定义区域
    regions = {
        '额头区域': (y_max * 0.6, y_max),
        '眼部区域': (y_max * 0.0, y_max * 0.6), 
        '鼻子区域': (y_max * -0.3, y_max * 0.0),
        '嘴部区域': (y_max * -0.6, y_max * -0.3),
        '下巴区域': (y_min, y_max * -0.6)
    }
    
    for region_name, (y_low, y_high) in regions.items():
        mask = (vertices[:, 1] >= y_low) & (vertices[:, 1] <= y_high)
        region_vertices = vertices[mask]
        
        if len(region_vertices) > 0:
            x_range = (region_vertices[:, 0].min(), region_vertices[:, 0].max())
            z_range = (region_vertices[:, 2].min(), region_vertices[:, 2].max())
            centerline_count = np.sum(np.abs(region_vertices[:, 0]) < 1e-5)
            
            print(f"{region_name}: {len(region_vertices)}个点, X范围{x_range}, Z范围{z_range}, 中线点{centerline_count}个")

def create_alignment_template(vertices):
    """基于canonical模型创建对齐模板"""
    print("\n=== 创建对齐模板 ===")
    
    # 找出关键的landmark点索引（基于coordinates和位置推测）
    key_landmarks = {}
    
    # 鼻尖（通常是Z坐标最大的中线点）
    centerline_points = find_centerline_points(vertices)
    if centerline_points:
        nose_tip_candidates = [(i, v) for i, v in centerline_points if v[2] > 5.5]
        if nose_tip_candidates:
            nose_tip_idx = max(nose_tip_candidates, key=lambda x: x[1][2])[0]
            key_landmarks['nose_tip'] = nose_tip_idx
            print(f"鼻尖点 (索引{nose_tip_idx}): {vertices[nose_tip_idx]}")
    
    # 左右眼角（X坐标绝对值较大，Z坐标较大的点）
    eye_candidates = []
    for i, v in enumerate(vertices):
        if abs(v[0]) > 6.0 and v[2] > 0 and v[1] > 2:  # 眼部区域
            eye_candidates.append((i, v))
    
    if len(eye_candidates) >= 2:
        # 按X坐标排序，取最左和最右
        eye_candidates.sort(key=lambda x: x[1][0])
        left_eye_idx = eye_candidates[0][0]
        right_eye_idx = eye_candidates[-1][0]
        key_landmarks['left_eye'] = left_eye_idx
        key_landmarks['right_eye'] = right_eye_idx
        print(f"左眼角 (索引{left_eye_idx}): {vertices[left_eye_idx]}")
        print(f"右眼角 (索引{right_eye_idx}): {vertices[right_eye_idx]}")
    
    # 左右嘴角（X坐标有一定值，Y坐标为负的点）
    mouth_candidates = []
    for i, v in enumerate(vertices):
        if abs(v[0]) > 1.0 and v[1] < 0 and v[2] > 2:  # 嘴部区域
            mouth_candidates.append((i, v))
    
    if len(mouth_candidates) >= 2:
        mouth_candidates.sort(key=lambda x: x[1][0])
        left_mouth_idx = mouth_candidates[0][0]
        right_mouth_idx = mouth_candidates[-1][0]
        key_landmarks['left_mouth'] = left_mouth_idx
        key_landmarks['right_mouth'] = right_mouth_idx
        print(f"左嘴角 (索引{left_mouth_idx}): {vertices[left_mouth_idx]}")
        print(f"右嘴角 (索引{right_mouth_idx}): {vertices[right_mouth_idx]}")
    
    return key_landmarks

def generate_alignment_strategy(vertices):
    """生成基于canonical模型的对齐策略"""
    print("\n=== 对齐策略生成 ===")
    
    # 分析对称性
    symmetric_pairs = find_exact_symmetric_pairs(vertices)
    centerline_points = find_centerline_points(vertices)
    
    print(f"找到 {len(symmetric_pairs)} 对精确对称点")
    print(f"找到 {len(centerline_points)} 个中线点")
    
    # 生成中线点索引列表
    centerline_indices = [i for i, _ in centerline_points]
    print(f"中线点索引: {centerline_indices[:20]}...")  # 显示前20个
    
    # 生成对称点对索引列表
    symmetric_indices = [(i, j) for i, j, _, _ in symmetric_pairs[:10]]  # 显示前10对
    print(f"对称点对索引（前10对）: {symmetric_indices}")
    
    # 分析几何中心
    center = np.mean(vertices, axis=0)
    print(f"几何中心: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
    
    # 分析主要轴向
    # X轴对称性
    x_coords = vertices[:, 0]
    x_symmetric_deviation = []
    for i, j, v1, v2 in symmetric_pairs:
        x_symmetric_deviation.append(abs(v1[0] + v2[0]))
    
    if x_symmetric_deviation:
        avg_x_deviation = np.mean(x_symmetric_deviation)
        print(f"对称点X坐标平均偏差: {avg_x_deviation:.6f}")
    
    # 输出对齐建议
    print("\n=== 对齐建议 ===")
    print("1. 确保以下点的X坐标为0（中线点）：")
    print(f"   索引: {centerline_indices}")
    
    print("2. 确保以下点对保持对称（X坐标相反，Y、Z坐标相同）：")
    for i, (idx1, idx2, v1, v2) in enumerate(symmetric_pairs[:5]):
        print(f"   点对 {i+1}: 索引({idx1}, {idx2})")
    
    print("3. 模型几何中心应接近原点，但可以有适当的Z轴偏移")
    
    return {
        'centerline_indices': centerline_indices,
        'symmetric_pairs': [(i, j) for i, j, _, _ in symmetric_pairs],
        'target_center': center,
        'key_landmarks': create_alignment_template(vertices)
    }

def save_alignment_config(config, filename="canonical_alignment_config.txt"):
    """保存对齐配置到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Canonical Face Model 对齐配置\n")
        f.write("# 基于canonical_face_model.obj的分析结果\n\n")
        
        f.write("[中线点索引]\n")
        f.write(f"centerline_indices = {config['centerline_indices']}\n\n")
        
        f.write("[对称点对索引]\n")
        f.write("# 格式: (左点索引, 右点索引)\n")
        for i, (left, right) in enumerate(config['symmetric_pairs'][:20]):  # 保存前20对
            f.write(f"pair_{i+1} = ({left}, {right})\n")
        f.write("\n")
        
        f.write("[目标几何中心]\n")
        center = config['target_center']
        f.write(f"target_center = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})\n\n")
        
        f.write("[关键Landmark点]\n")
        for name, idx in config['key_landmarks'].items():
            f.write(f"{name} = {idx}\n")
    
    print(f"对齐配置已保存到: {filename}")

def visualize_symmetric_structure(vertices, symmetric_pairs, centerline_points):
    """可视化对称结构"""
    fig = plt.figure(figsize=(15, 5))
    
    # 三个角度的2D投影
    angles = [
        ('X-Y 平面 (前视图)', 0, 1),
        ('X-Z 平面 (俯视图)', 0, 2), 
        ('Y-Z 平面 (侧视图)', 1, 2)
    ]
    
    for i, (title, axis1, axis2) in enumerate(angles):
        ax = fig.add_subplot(1, 3, i+1)
        
        # 绘制所有点
        ax.scatter(vertices[:, axis1], vertices[:, axis2], 
                  alpha=0.3, s=10, c='gray', label='所有点')
        
        # 绘制中线点
        if centerline_points:
            centerline_coords = np.array([v for _, v in centerline_points])
            ax.scatter(centerline_coords[:, axis1], centerline_coords[:, axis2], 
                      c='red', s=30, label='中线点', marker='o')
        
        # 绘制对称点对
        for j, (idx1, idx2, v1, v2) in enumerate(symmetric_pairs[:10]):  # 只显示前10对
            ax.plot([v1[axis1], v2[axis1]], [v1[axis2], v2[axis2]], 
                   'b-', alpha=0.6, linewidth=1)
            if j == 0:  # 只添加一次标签
                ax.plot([v1[axis1], v2[axis1]], [v1[axis2], v2[axis2]], 
                       'b-', alpha=0.6, linewidth=1, label='对称点对连线')
        
        ax.set_xlabel(f'坐标轴 {"XYZ"[axis1]}')
        ax.set_ylabel(f'坐标轴 {"XYZ"[axis2]}')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('canonical_face_structure.png', dpi=150, bbox_inches='tight')
    print("对称结构图已保存为: canonical_face_structure.png")
    plt.show()

def main():
    """主函数"""
    print("=== Canonical Face Model 结构分析工具 ===")
    
    # 加载canonical模型
    canonical_path = "obj/canonical_face_model.obj"
    if not os.path.exists(canonical_path):
        print(f"错误：找不到文件 {canonical_path}")
        return
    
    print(f"正在加载: {canonical_path}")
    vertices = load_vertices_from_obj(canonical_path)
    print(f"加载完成: {len(vertices)} 个顶点")
    
    # 基本统计
    print(f"\n=== 基本统计 ===")
    print(f"X坐标范围: [{vertices[:, 0].min():.6f}, {vertices[:, 0].max():.6f}]")
    print(f"Y坐标范围: [{vertices[:, 1].min():.6f}, {vertices[:, 1].max():.6f}]")
    print(f"Z坐标范围: [{vertices[:, 2].min():.6f}, {vertices[:, 2].max():.6f}]")
    
    # 分析对称结构
    print(f"\n=== 对称结构分析 ===")
    symmetric_pairs = find_exact_symmetric_pairs(vertices)
    centerline_points = find_centerline_points(vertices)
    
    print(f"精确对称点对数量: {len(symmetric_pairs)}")
    print(f"中线点数量: {len(centerline_points)}")
    
    # 显示前几对对称点
    print("\n前5对对称点:")
    for i, (idx1, idx2, v1, v2) in enumerate(symmetric_pairs[:5]):
        print(f"点对 {i+1}: 索引({idx1:3d}, {idx2:3d})")
        print(f"  左点: ({v1[0]:8.3f}, {v1[1]:8.3f}, {v1[2]:8.3f})")
        print(f"  右点: ({v2[0]:8.3f}, {v2[1]:8.3f}, {v2[2]:8.3f})")
        print(f"  差异: ({v1[0]+v2[0]:8.3f}, {v1[1]-v2[1]:8.3f}, {v1[2]-v2[2]:8.3f})")
    
    # 显示前几个中线点
    print("\n前10个中线点:")
    for i, (idx, vertex) in enumerate(centerline_points[:10]):
        print(f"点 {i+1}: 索引{idx:3d}, 坐标({vertex[0]:8.6f}, {vertex[1]:8.3f}, {vertex[2]:8.3f})")
    
    # 区域分析
    analyze_face_regions(vertices)
    
    # 生成对齐策略
    alignment_config = generate_alignment_strategy(vertices)
    
    # 保存配置
    save_alignment_config(alignment_config)
    
    # 可视化（可选）
    try:
        visualize_symmetric_structure(vertices, symmetric_pairs, centerline_points)
    except Exception as e:
        print(f"可视化跳过（需要matplotlib）: {e}")
    
    print("\n=== 分析完成 ===")
    print("现在你可以使用生成的配置文件来创建准确的对齐算法了！")

if __name__ == "__main__":
    main() 