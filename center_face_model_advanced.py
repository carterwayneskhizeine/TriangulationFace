#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级面部模型居中对齐工具
采用保守的对齐策略，主要基于优化中线点位置和整体几何中心
"""

import os
import sys
import numpy as np
from scipy.optimize import minimize

class AdvancedFaceModelCenteringTool:
    def __init__(self):
        self.vertices = []
        self.texture_coords = []
        self.faces = []
        self.comments = []
        
        # 定义一些已知的对称点对（左右对称）
        # 基于MediaPipe面部landmark的对称性
        self.symmetric_pairs = [
            # 脸颊轮廓对称点
            (132, 361),  # 左右眼角外侧
            (149, 378),  # 左右眼角
            (172, 397),  # 左右脸颊
            (136, 365),  # 左右下颌
            (150, 379),  # 左右下巴侧面
            (58, 288),   # 左右嘴唇
            (61, 291),   # 左右嘴角
            (84, 314),   # 左右上唇
            (46, 276),   # 左右眼睛
            (33, 263),   # 左右鼻翼
        ]
        
        # 应该在中轴线上的点（X坐标应该接近0）
        self.center_line_points = [
            0,    # 鼻尖
            1, 2, # 鼻梁
            4, 5, 6, # 鼻子和额头中线
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, # 脸部中线轮廓
            19, 20, # 鼻子下方
            94,   # 下唇中央
            151, 152, # 脸部下方中线
            164,  # 下巴
            168,  # 鼻子下方
            175,  # 下巴下方
            195,  # 上唇中央
            199, 200  # 下巴最下方
        ]
        
        # 目标第一个点的坐标（鼻尖）
        self.target_first_point = np.array([0.000000, -3.406404, 5.979507])
        
    def load_obj_file(self, obj_path):
        """加载OBJ文件"""
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ文件不存在: {obj_path}")
            
        print(f"正在加载OBJ文件: {obj_path}")
        
        with open(obj_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('v '):
                    # 顶点坐标
                    parts = line.split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    self.vertices.append([x, y, z])
                elif line.startswith('vt '):
                    # 纹理坐标
                    self.texture_coords.append(line)
                elif line.startswith('f '):
                    # 面
                    self.faces.append(line)
                elif line.startswith('#'):
                    # 注释
                    self.comments.append(line)
                    
        print(f"加载完成: {len(self.vertices)} 个顶点, {len(self.faces)} 个面")
        self.vertices = np.array(self.vertices)
        
    def compute_conservative_alignment(self):
        """计算保守的对齐变换"""
        print("计算保守对齐变换...")
        
        # 步骤1：平移使鼻尖到达目标位置
        current_nose_tip = self.vertices[0]
        translation = self.target_first_point - current_nose_tip
        
        print(f"平移向量: {translation}")
        
        # 步骤2：寻找最佳的轻微旋转来优化对称性
        # 使用优化算法找到最小化以下目标的旋转角度：
        # 1. 中线点的X坐标偏差
        # 2. 对称点对的不对称性
        
        def rotation_matrix_y(angle):
            """绕Y轴旋转的旋转矩阵"""
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            return np.array([
                [cos_a,  0, sin_a],
                [0,      1, 0    ],
                [-sin_a, 0, cos_a]
            ])
            
        def objective_function(angle):
            """优化目标函数"""
            # 应用平移和旋转
            R = rotation_matrix_y(angle[0])
            transformed_vertices = (R @ (self.vertices + translation).T).T
            
            # 目标1：最小化中线点的X坐标偏差
            center_line_cost = 0
            for idx in self.center_line_points:
                if idx < len(transformed_vertices):
                    center_line_cost += abs(transformed_vertices[idx, 0]) ** 2
            
            # 目标2：最小化对称点对的不对称性
            symmetry_cost = 0
            for left_idx, right_idx in self.symmetric_pairs:
                if left_idx < len(transformed_vertices) and right_idx < len(transformed_vertices):
                    left_point = transformed_vertices[left_idx]
                    right_point = transformed_vertices[right_idx]
                    
                    # 对称点应该有相反的X坐标，相同的Y和Z坐标
                    x_symmetry_error = (left_point[0] + right_point[0]) ** 2
                    y_symmetry_error = (left_point[1] - right_point[1]) ** 2
                    z_symmetry_error = (left_point[2] - right_point[2]) ** 2
                    
                    symmetry_cost += x_symmetry_error + 0.1 * (y_symmetry_error + z_symmetry_error)
            
            # 目标3：保持鼻尖在目标位置（软约束）
            nose_tip_cost = np.linalg.norm(transformed_vertices[0] - self.target_first_point) ** 2
            
            total_cost = center_line_cost + symmetry_cost + 10 * nose_tip_cost
            return total_cost
        
        # 搜索最佳旋转角度（限制在小角度范围内）
        print("优化旋转角度...")
        result = minimize(objective_function, [0.0], bounds=[(-0.2, 0.2)], method='L-BFGS-B')
        
        optimal_angle = result.x[0]
        print(f"最佳旋转角度: {np.degrees(optimal_angle):.2f} 度")
        print(f"优化成本: {result.fun:.6f}")
        
        # 计算最终变换矩阵
        rotation_matrix = rotation_matrix_y(optimal_angle)
        
        return rotation_matrix, translation
        
    def apply_transform(self, rotation_matrix, translation):
        """应用变换到所有顶点"""
        print("应用变换到所有顶点...")
        
        # 先平移再旋转
        translated_vertices = self.vertices + translation
        transformed_vertices = (rotation_matrix @ translated_vertices.T).T
        
        # 轻微调整中线点的X坐标（微调）
        for idx in self.center_line_points:
            if idx < len(transformed_vertices):
                if abs(transformed_vertices[idx, 0]) < 0.1:  # 只调整接近中线的点
                    transformed_vertices[idx, 0] = 0.0
        
        self.vertices = transformed_vertices
        
    def analyze_symmetry_after_transform(self):
        """分析变换后的对称性"""
        print("\n变换后的对称性分析:")
        
        # 检查中线点
        center_x_coords = []
        for idx in self.center_line_points:
            if idx < len(self.vertices):
                x_coord = self.vertices[idx, 0]
                center_x_coords.append(abs(x_coord))
                if abs(x_coord) > 0.01:
                    print(f"中线点 {idx+1}: X = {x_coord:.6f}")
                    
        avg_center_deviation = np.mean(center_x_coords) if center_x_coords else 0
        print(f"中线点平均X偏差: {avg_center_deviation:.6f}")
        
        # 检查对称点对
        symmetry_errors = []
        good_symmetric_pairs = 0
        for left_idx, right_idx in self.symmetric_pairs:
            if left_idx < len(self.vertices) and right_idx < len(self.vertices):
                left_point = self.vertices[left_idx]
                right_point = self.vertices[right_idx]
                
                # 对称点应该有相反的X坐标，相同的Y和Z坐标
                x_symmetry_error = abs(left_point[0] + right_point[0])
                y_symmetry_error = abs(left_point[1] - right_point[1])
                z_symmetry_error = abs(left_point[2] - right_point[2])
                
                total_error = x_symmetry_error + y_symmetry_error + z_symmetry_error
                symmetry_errors.append(total_error)
                
                if total_error < 0.1:  # 良好的对称性
                    good_symmetric_pairs += 1
                
        if symmetry_errors:
            avg_symmetry_error = np.mean(symmetry_errors)
            print(f"对称点对平均误差: {avg_symmetry_error:.6f}")
            print(f"良好对称的点对数量: {good_symmetric_pairs}/{len(self.symmetric_pairs)}")
        
        # 验证第一个点是否在目标位置
        first_point = self.vertices[0]
        distance_to_target = np.linalg.norm(first_point - self.target_first_point)
        print(f"鼻尖点与目标位置距离: {distance_to_target:.6f}")
        
        # 检查模型的整体几何中心
        center = np.mean(self.vertices, axis=0)
        print(f"模型几何中心: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
        
    def save_obj_file(self, output_path):
        """保存变换后的OBJ文件"""
        print(f"正在保存到: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as file:
            # 写入注释
            file.write("# 保守对齐后的面部模型\n")
            file.write("# 基于优化的平移和小角度旋转变换\n")
            
            # 写入顶点
            for vertex in self.vertices:
                file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
            # 写入纹理坐标
            for vt in self.texture_coords:
                file.write(f"{vt}\n")
                
            # 写入面
            for face in self.faces:
                file.write(f"{face}\n")
                
        print(f"模型保存完成: {output_path}")
        
    def process_model(self, input_path, output_path=None):
        """处理模型的主函数"""
        print("=== 保守面部模型对齐工具 ===\n")
        
        # 加载模型
        self.load_obj_file(input_path)
        
        # 计算对齐变换
        rotation_matrix, translation = self.compute_conservative_alignment()
        
        # 应用变换
        self.apply_transform(rotation_matrix, translation)
        
        # 分析结果
        self.analyze_symmetry_after_transform()
        
        # 保存结果
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_conservative_aligned.obj"
            
        self.save_obj_file(output_path)
        
        print(f"\n✅ 保守对齐处理完成!")
        print(f"原始文件: {input_path}")
        print(f"对齐文件: {output_path}")
        print("\n现在模型应该具有更好的对称性和对齐方式。")
        
        return output_path

def main():
    """主函数"""
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # 查找最新的模型文件
        result_dir = "result_file"
        if not os.path.exists(result_dir):
            print("错误：未找到 result_file 目录")
            return
            
        model_files = [f for f in os.listdir(result_dir) if f.endswith('_face_model.obj') and 'aligned' not in f]
        if not model_files:
            print("错误：未找到面部模型文件")
            return
            
        # 按修改时间排序，选择最新的
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(result_dir, x)), reverse=True)
        input_file = os.path.join(result_dir, model_files[0])
        
    if not os.path.exists(input_file):
        print(f"错误：文件不存在 {input_file}")
        return
        
    # 创建工具实例并处理
    tool = AdvancedFaceModelCenteringTool()
    output_file = tool.process_model(input_file)
    
    print(f"\n处理完成！输出文件: {output_file}")

if __name__ == "__main__":
    main() 