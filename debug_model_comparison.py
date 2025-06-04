#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试模型比较工具
检查两个OBJ文件的顶点对应关系和差异
"""

import numpy as np
import os

def load_obj_vertices(obj_path):
    """加载OBJ文件的顶点"""
    vertices = []
    print(f"\n正在加载: {obj_path}")
    
    with open(obj_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
                    if len(vertices) <= 5:  # 显示前5个顶点
                        print(f"  顶点{len(vertices):3d} (行{line_num:2d}): ({x:8.6f}, {y:8.6f}, {z:8.6f})")
    
    print(f"总共加载了 {len(vertices)} 个顶点")
    return np.array(vertices, dtype=np.float32)

def analyze_difference(vertices1, vertices2, name1, name2):
    """分析两个模型的差异"""
    print(f"\n=== 差异分析: {name1} vs {name2} ===")
    
    # 确保顶点数量匹配
    min_count = min(len(vertices1), len(vertices2))
    if len(vertices1) != len(vertices2):
        print(f"⚠️ 顶点数量不匹配: {len(vertices1)} vs {len(vertices2)}")
        print(f"使用前 {min_count} 个顶点进行比较")
        vertices1 = vertices1[:min_count]
        vertices2 = vertices2[:min_count]
    
    # 计算差异向量
    diff = vertices2 - vertices1  # 目标模型 - 对齐模型
    
    print(f"前5个点的坐标对比:")
    for i in range(min(5, len(vertices1))):
        print(f"  点{i+1:3d}: 对齐模型({vertices1[i,0]:8.6f}, {vertices1[i,1]:8.6f}, {vertices1[i,2]:8.6f})")
        print(f"        目标模型({vertices2[i,0]:8.6f}, {vertices2[i,1]:8.6f}, {vertices2[i,2]:8.6f})")
        print(f"        差异向量({diff[i,0]:8.6f}, {diff[i,1]:8.6f}, {diff[i,2]:8.6f})")
        print()
    
    # 统计差异
    distances = np.linalg.norm(diff, axis=1)
    print(f"差异统计:")
    print(f"  平均距离: {np.mean(distances):.6f}")
    print(f"  最大距离: {np.max(distances):.6f}")
    print(f"  最小距离: {np.min(distances):.6f}")
    print(f"  标准差:   {np.std(distances):.6f}")
    
    print(f"  X方向差异: [{diff[:,0].min():.6f}, {diff[:,0].max():.6f}], 平均: {diff[:,0].mean():.6f}")
    print(f"  Y方向差异: [{diff[:,1].min():.6f}, {diff[:,1].max():.6f}], 平均: {diff[:,1].mean():.6f}")
    print(f"  Z方向差异: [{diff[:,2].min():.6f}, {diff[:,2].max():.6f}], 平均: {diff[:,2].mean():.6f}")
    
    # 检查是否存在明显的偏移或缩放
    center1 = np.mean(vertices1, axis=0)
    center2 = np.mean(vertices2, axis=0)
    center_diff = center2 - center1
    
    print(f"\n几何中心比较:")
    print(f"  对齐模型中心: ({center1[0]:8.6f}, {center1[1]:8.6f}, {center1[2]:8.6f})")
    print(f"  目标模型中心: ({center2[0]:8.6f}, {center2[1]:8.6f}, {center2[2]:8.6f})")
    print(f"  中心差异:     ({center_diff[0]:8.6f}, {center_diff[1]:8.6f}, {center_diff[2]:8.6f})")
    
    # 检查缩放比例
    scale1 = np.std(vertices1, axis=0)
    scale2 = np.std(vertices2, axis=0)
    scale_ratio = scale2 / scale1
    
    print(f"\n缩放比例分析:")
    print(f"  对齐模型标准差: ({scale1[0]:8.6f}, {scale1[1]:8.6f}, {scale1[2]:8.6f})")
    print(f"  目标模型标准差: ({scale2[0]:8.6f}, {scale2[1]:8.6f}, {scale2[2]:8.6f})")
    print(f"  缩放比例:       ({scale_ratio[0]:8.6f}, {scale_ratio[1]:8.6f}, {scale_ratio[2]:8.6f})")
    
    return diff

def main():
    """主函数"""
    print("=== 模型差异调试工具 ===")
    
    # 文件路径
    aligned_file = "result_file/averaged_landmarks_1749047085_face_model_precise_aligned.obj"
    target_file = "obj/Andy_Wah_facemesh.obj"
    
    # 检查文件是否存在
    if not os.path.exists(aligned_file):
        print(f"❌ 文件不存在: {aligned_file}")
        return
    
    if not os.path.exists(target_file):
        print(f"❌ 文件不存在: {target_file}")
        return
    
    # 加载顶点
    aligned_vertices = load_obj_vertices(aligned_file)
    target_vertices = load_obj_vertices(target_file)
    
    # 分析差异
    diff = analyze_difference(aligned_vertices, target_vertices, 
                            "对齐后活人脸", "目标模型(Andy_Wah)")
    
    # 保存差异向量
    output_file = "npy/debug_shape_difference.npy"
    os.makedirs("npy", exist_ok=True)
    np.save(output_file, diff)
    print(f"\n差异向量已保存到: {output_file}")
    
    # 给出建议
    print(f"\n=== 诊断建议 ===")
    avg_distance = np.mean(np.linalg.norm(diff, axis=1))
    center_distance = np.linalg.norm(np.mean(target_vertices, axis=0) - np.mean(aligned_vertices, axis=0))
    
    if avg_distance > 1.0:
        print("⚠️ 平均差异距离较大，可能存在以下问题:")
        print("   1. 两个模型的比例尺不同")
        print("   2. 对齐过程中出现错误")
        print("   3. 模型本身差异较大")
    
    if center_distance > 0.5:
        print("⚠️ 几何中心差异较大，模型可能未正确对齐")
    
    if avg_distance < 0.5 and center_distance < 0.2:
        print("✅ 模型差异在合理范围内，可以用于变形")
    
    print(f"\n建议在实时变形时使用以下参数:")
    print(f"  差异向量文件: {output_file}")
    print(f"  预期变形强度: {'温和' if avg_distance < 1.0 else '强烈'}")

if __name__ == "__main__":
    main() 