#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe 人脸标志检测器 - 视频文件处理
处理 .mp4 视频文件进行人脸标志检测和显示 python = 3.9
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
import os
import csv
import json
from PIL import Image, ImageDraw, ImageFont
import platform
from face_warper import FaceWarper
import argparse

# 按照官方文档推荐的导入方式
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class FaceLandmarkerVideo:
    def __init__(self, video_path=None):
        """初始化人脸标志检测器"""
        self.result = None
        self.output_image = None
        self.timestamp = 0
        self.video_path = video_path
        
        # 创建输出文件夹
        self.output_dir = "output_path"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出文件夹: {self.output_dir}")
        
        # 创建JSON文件夹
        self.json_dir = "jsonfile"
        if not os.path.exists(self.json_dir):
            os.makedirs(self.json_dir)
            print(f"创建JSON文件夹: {self.json_dir}")
        
        # JSON输出控制
        self.enable_json_output = False  # 控制是否启用JSON输出
        self.clear_json_on_start = True   # 启动时是否清理之前的JSON文件
        
        # 下载模型文件
        self.model_path = self.download_model()
        
        # 创建FaceLandmarker
        self.landmarker = self.create_landmarker()
        
        # 预处理：加载模型差异和标准模型顶点，初始化实时变形参数
        try:
            # 直接加载自定义模型顶点
            self.custom_vertices = self.load_obj_vertices(os.path.join('obj', 'Andy_Wah_facemesh.obj'))
            print("已加载自定义模型顶点")
        except Exception as e:
            print(f"自定义模型加载失败: {e}")
            self.custom_vertices = None
        self.landmark_buffer = []
        self.frame_count = 0
        self.N = 60
        self.warp_ready = False
        self.diff_transformed = None
        self.transform_file = "face_transform.npy"  # 保存变换参数的文件
        
        # 加载中文字体用于GUI显示
        self.chinese_font = self.load_chinese_font()
        
        # 尝试加载已保存的变换参数
        self.load_saved_transform()
        
        # 宽高比处理参数（将在run方法中根据实际视频分辨率更新）
        self.video_width = 1280  # 默认值，会被实际视频分辨率覆盖
        self.video_height = 720  # 默认值，会被实际视频分辨率覆盖
        self.aspect_ratio = self.video_width / self.video_height  # 默认16:9比例
        self.x_scale_factor = self.aspect_ratio / 1  # 用于修正x坐标的拉伸，会根据实际视频更新
        
        # 变形显示控制
        self.show_warped = True  # True=显示变形后，False=显示原始
        
        # 初始化人脸变形器
        self.face_warper = FaceWarper()
        
        # 像素变形控制
        self.enable_pixel_warp = False  # 控制是否启用像素级变形
        self.previous_landmarks = None  # 用于平滑处理
        # 透视投影控制
        self.enable_perspective_projection = True  # 是否使用透视投影
        # 新的透视投影参数
        self.perspective_strength = 0.2  # 透视强度 (0.0-1.0)
        self.depth_enhancement = 3.0    # 深度增强系数 (1.0-20.0)
        
        # 显示控制
        self.show_landmarks = True  # 控制是否显示landmarks点和连线
        
        # landmarks大小调整
        self.landmarks_scale = 1.0  # landmarks缩放系数
        self.scale_step = 0.05      # 每次调整的步长
        
        # landmarks宽度比例调整
        self.width_scale = 1.0      # landmarks宽度缩放系数（高度保持不变）
        self.width_scale_step = 0.05 # 宽度调整步长
        
        # 边缘模糊控制
        self.enable_edge_blur = False  # 是否启用边缘滤波
        
        # 视频保存控制
        self.save_output_video = False
        self.output_video_writer = None
        
        # 循环播放控制
        self.loop_playback = True  # 默认启用循环播放
        self.loop_count = 999  # 循环次数计数
        
        print("人脸标志检测器初始化完成")
        print("按 'Q' 键退出程序")
        print("按 'S' 键保存当前帧")
        print("按 'L' 键保存当前landmarks到CSV文件")
        print("按 'M' 键重新检测前60帧并计算变换")
        print("按 'X' 键切换原始/变形landmarks显示")
        print("按 'P' 键切换像素级人脸变形显示")
        print("按 'H' 键隐藏/显示landmarks线框")
        print("按 '[/]' - 调整landmarks整体缩放")
        print("按 '3/4' - 调整landmarks宽度比例")
        print("按 'T' - 切换透视投影/弱透视投影")
        print("按 '5/6' - 调整透视强度")
        print("按 '7/8' - 调整深度增强")
        print("按 'R' 键开始/停止录制输出视频")
        print("按 'SPACE' 键暂停/继续播放")
        print("按 'C' 键切换循环播放模式")
        print("按 'ESC' 键或 'Q' 键退出程序")
        print("按 'J/j' - 开启/关闭JSON输出")
        print("按 'K/k' - 清理所有JSON文件")

    def download_model(self):
        """下载人脸标志检测模型"""
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        model_path = "face_landmarker.task"
        
        if not os.path.exists(model_path):
            print("正在下载人脸标志检测模型...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"模型下载完成: {model_path}")
            except Exception as e:
                print(f"模型下载失败: {e}")
                print("请手动下载模型文件到项目目录")
                return None
        else:
            print(f"模型文件已存在: {model_path}")
        
        return model_path

    def result_callback(self, result, output_image, timestamp_ms):
        """处理检测结果的回调函数"""
        try:
            self.result = result
            self.output_image = output_image
            self.timestamp = timestamp_ms
            
            # 采集前 N 帧 landmarks 并估计变换
            if result.face_landmarks and not self.warp_ready:
                # 仅使用第一个检测到的人脸
                landmarks = result.face_landmarks[0]
                # 只使用前468个landmarks，避免478 vs 468的维度不匹配问题
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks[:468]], dtype=np.float32)
                self.landmark_buffer.append(coords)
                self.frame_count += 1
                print(f"收集第 {self.frame_count}/{self.N} 帧 landmarks (使用前468个点)")
                
                if self.frame_count >= self.N:
                    print("开始计算相似变换...")
                    avg_landmarks = np.mean(self.landmark_buffer, axis=0)
                    self.estimate_transform(avg_landmarks)
                    self.landmark_buffer = []
        except Exception as e:
            print(f"回调函数错误: {e}")
            import traceback
            traceback.print_exc()

    def create_landmarker(self):
        """创建人脸标志检测器 - 按照官方文档推荐方式"""
        if not self.model_path:
            return None
            
        # 配置选项 - 完全按照官方文档格式
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,  # 视频模式
            num_faces=2,  # 最多检测2张人脸
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,  # 输出面部表情系数
            output_facial_transformation_matrixes=True,  # 输出面部转换矩阵
        )
        
        # 按照官方文档推荐，直接返回创建的landmarker对象
        return FaceLandmarker.create_from_options(options)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """在图像上绘制人脸标志点并应用像素级变形"""
        try:
            # 如果启用像素变形且变换已就绪，进行像素级变形
            if (self.enable_pixel_warp and self.warp_ready and 
                self.diff_transformed is not None and self.show_warped and 
                detection_result.face_landmarks):
                
                # 进行像素级人脸变形
                warped_image = self.apply_pixel_warp(rgb_image, detection_result)
                annotated_image = np.copy(warped_image)
            else:
                annotated_image = np.copy(rgb_image)
            
            # 绘制人脸标志点
            if detection_result.face_landmarks:
                # 获取MediaPipe面部网格连接信息
                mp_face_mesh = mp.solutions.face_mesh
                
                for face_landmarks in detection_result.face_landmarks:
                    height, width, _ = annotated_image.shape
                    # 原始坐标数组
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks[:468]], dtype=np.float32)
                    
                    # 应用形状偏移（用于显示landmarks位置）
                    if self.warp_ready and self.diff_transformed is not None and self.show_warped:
                        # 先修正当前landmarks的x坐标
                        corrected_coords = coords.copy()
                        corrected_coords[:, 0] *= self.x_scale_factor
                        
                        # 直接加上形状差异，将活人脸变形为自定义模型的形状
                        warped_coords = corrected_coords + self.diff_transformed
                        
                        # 将变形后的x坐标还原到16:9坐标系
                        warped_coords[:, 0] /= self.x_scale_factor
                        
                        # 应用landmarks缩放调整
                        if self.landmarks_scale != 1.0:
                            # 计算landmarks中心点
                            center = np.mean(warped_coords, axis=0)
                            # 以中心点为基准进行缩放
                            warped_coords = center + (warped_coords - center) * self.landmarks_scale
                        
                        # 应用宽度比例调整（只调整X坐标，保持Y坐标不变）
                        if self.width_scale != 1.0:
                            # 计算人脸中心的X坐标（使用所有landmarks的X坐标平均值）
                            face_center_x = np.mean(warped_coords[:, 0])
                            # 以人脸中心X坐标为基准，只调整X坐标
                            warped_coords[:, 0] = face_center_x + (warped_coords[:, 0] - face_center_x) * self.width_scale
                        
                        coords = warped_coords
                    else:
                        # 原始landmarks模式，也应用宽度比例调整
                        if self.width_scale != 1.0:
                            # 计算人脸中心的X坐标（使用所有landmarks的X坐标平均值）
                            face_center_x = np.mean(coords[:, 0])
                            # 以人脸中心X坐标为基准，只调整X坐标
                            coords[:, 0] = face_center_x + (coords[:, 0] - face_center_x) * self.width_scale
                    
                    # 根据显示控制决定是否绘制landmarks
                    if self.show_landmarks:
                        # 绘制landmarks点（红色）
                        for x_norm, y_norm, _ in coords:
                            x = int(x_norm * width)
                            y = int(y_norm * height)
                            # 确保坐标在有效范围内
                            if 0 <= x < width and 0 <= y < height:
                                cv2.circle(annotated_image, (x, y), 1, (0, 0, 255), -1)
                        
                        # 绘制面部网格连线（绿色）- 使用FACEMESH_TESSELATION连接
                        if hasattr(mp_face_mesh, 'FACEMESH_TESSELATION'):
                            connections = mp_face_mesh.FACEMESH_TESSELATION
                            for (start_idx, end_idx) in connections:
                                if start_idx < len(coords) and end_idx < len(coords):
                                    sx = int(coords[start_idx, 0] * width)
                                    sy = int(coords[start_idx, 1] * height)
                                    ex = int(coords[end_idx, 0] * width)
                                    ey = int(coords[end_idx, 1] * height)
                                    # 确保坐标在有效范围内
                                    if (0 <= sx < width and 0 <= sy < height and 
                                        0 <= ex < width and 0 <= ey < height):
                                        cv2.line(annotated_image, (sx, sy), (ex, ey), (0, 255, 0), 1)
            
            return annotated_image
        except Exception as e:
            print(f"绘制landmarks错误: {e}")
            import traceback
            traceback.print_exc()
            return rgb_image  # 返回原始图像
    
    def convert_to_perspective_landmarks(self, landmarks_normalized):
        """
        将MediaPipe的弱透视投影landmarks转换为透视投影landmarks
        
        Args:
            landmarks_normalized: 归一化的landmarks坐标 (N, 3)
        
        Returns:
            perspective_landmarks: 透视投影的landmarks坐标 (N, 3)
        """
        try:
            perspective_landmarks = landmarks_normalized.copy()
            height, width = self.video_height, self.video_width
            
            for i, (x_norm, y_norm, z_weak) in enumerate(landmarks_normalized):
                # 转换为像素坐标
                x_pixel = x_norm * width
                y_pixel = y_norm * height
                
                # 计算透视深度
                # z_weak是弱透视的Z值，我们需要将其转换为真实深度
                z_perspective = self.perspective_strength * z_weak + (1 - self.perspective_strength) * self.depth_enhancement
                
                # 透视投影逆变换：从2D像素坐标和深度得到3D坐标
                manual_fx = self.video_width
                manual_fy = self.video_width  
                manual_cx = self.video_width / 2
                manual_cy = self.video_height / 2
                
                x_3d = (x_pixel - manual_cx) * z_perspective / manual_fx
                y_3d = (y_pixel - manual_cy) * z_perspective / manual_fy
                
                # 重新投影到归一化坐标，但保持透视效果
                # 这里我们保持X,Y的相对关系，但调整Z以反映真实深度
                perspective_landmarks[i] = [x_norm, y_norm, z_perspective / self.depth_enhancement]
            
            return perspective_landmarks
            
        except Exception as e:
            print(f"透视投影转换失败: {e}")
            return landmarks_normalized
    
    def apply_perspective_warp_to_landmarks(self, src_landmarks, dst_landmarks):
        """
        对landmarks应用透视投影变形
        
        Args:
            src_landmarks: 源landmarks (N, 3)
            dst_landmarks: 目标landmarks (N, 3)
        
        Returns:
            perspective_dst_landmarks: 透视投影变形后的landmarks (N, 3)
        """
        try:
            if not self.enable_perspective_projection:
                return dst_landmarks
            
            perspective_dst = dst_landmarks.copy()
            
            # 【重新设计】使用真正的透视投影公式
            # 设置基础参数
            focal_length = min(self.video_width, self.video_height) * 0.8  # 焦距
            base_depth = 50.0  # 基础深度（厘米）
            depth_variation = 5.0  # 深度变化范围（厘米）
             
            # 对每个landmark应用透视变换
            for i in range(len(dst_landmarks)):
                x_norm, y_norm, z_norm = dst_landmarks[i]
                
                # 转换为像素坐标
                x_pixel = x_norm * self.video_width
                y_pixel = y_norm * self.video_height
                
                # 【新算法】计算3D坐标
                # Z值：负值表示离相机近，正值表示离相机远
                z_3d = base_depth + (z_norm * depth_variation)
                
                # 将像素坐标转换为相机坐标系
                x_3d = (x_pixel - self.video_width/2) * z_3d / focal_length
                y_3d = (y_pixel - self.video_height/2) * z_3d / focal_length
                
                # 【关键】应用透视投影：P' = f * P / Z
                # 这里我们通过调整Z来产生透视效果
                perspective_z = z_3d * (1.0 + z_norm * self.perspective_strength * self.depth_enhancement)  # 根据原始Z值调整透视深度
                
                # 重新投影到2D
                new_x_pixel = (x_3d * focal_length / perspective_z) + self.video_width/2
                new_y_pixel = (y_3d * focal_length / perspective_z) + self.video_height/2
                
                # 转换回归一化坐标
                perspective_dst[i] = [
                    new_x_pixel / self.video_width,
                    new_y_pixel / self.video_height,
                    z_norm
                ]
            
            return perspective_dst
            
        except Exception as e:
            print(f"透视变形失败: {e}")
            return dst_landmarks
    
    def apply_pixel_warp(self, rgb_image, detection_result):
        """应用像素级人脸变形"""
        try:
            if not detection_result.face_landmarks:
                return rgb_image
            
            # 获取第一个检测到的人脸landmarks
            face_landmarks = detection_result.face_landmarks[0]
            
            # 转换为归一化坐标数组
            original_coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks[:468]], dtype=np.float32)
            
            # 平滑处理，减少抖动
            if self.previous_landmarks is not None:
                original_coords = self.face_warper.smooth_landmarks(
                    original_coords, self.previous_landmarks, smoothing_factor=0.7
                )
            self.previous_landmarks = original_coords.copy()
            
            # 计算变形后的坐标
            warped_coords = original_coords.copy()
            
            # 应用形状差异变换
            corrected_coords = original_coords.copy()
            corrected_coords[:, 0] *= self.x_scale_factor
            
            # 加上形状差异
            warped_coords = corrected_coords + self.diff_transformed
            
            # 将x坐标还原到16:9坐标系
            warped_coords[:, 0] /= self.x_scale_factor
            
            # 应用landmarks缩放调整
            if self.landmarks_scale != 1.0:
                # 计算landmarks中心点
                center = np.mean(warped_coords, axis=0)
                # 以中心点为基准进行缩放
                warped_coords = center + (warped_coords - center) * self.landmarks_scale
            
            # 应用宽度比例调整（只调整X坐标，保持Y坐标不变）
            if self.width_scale != 1.0:
                # 计算人脸中心的X坐标（使用所有landmarks的X坐标平均值）
                face_center_x = np.mean(warped_coords[:, 0])
                # 以人脸中心X坐标为基准，只调整X坐标
                warped_coords[:, 0] = face_center_x + (warped_coords[:, 0] - face_center_x) * self.width_scale
            
            # 【关键】如果启用透视投影，对变形后的landmarks应用透视变换
            if self.enable_perspective_projection:
                warped_coords = self.apply_perspective_warp_to_landmarks(original_coords, warped_coords)
                print(f"透视投影已应用，强度: {self.perspective_strength:.2f}, 增强: {self.depth_enhancement:.1f}")
            
            # 应用人脸变形
            warped_image = self.face_warper.apply_face_warp(
                rgb_image,
                original_coords,
                warped_coords,
                blend_ratio=1.0,  # 完全替换
                enable_blur_edge=self.enable_edge_blur
            )
            
            return warped_image
            
        except Exception as e:
            print(f"像素变形失败: {e}")
            import traceback
            traceback.print_exc()
            return rgb_image
    

    def draw_face_contours(self, image, landmarks, width, height):
        """绘制面部轮廓 - 已废弃，现在使用FACEMESH_TESSELATION绘制完整面部网格"""
        # 此方法已被新的draw_landmarks_on_image方法中的FACEMESH_TESSELATION绘制替代
        pass
    
    def save_landmarks_to_csv(self, landmarks_list, filename=None):
        """保存landmarks到CSV文件"""
        if not landmarks_list:
            print("没有landmarks数据可保存")
            return False
        
        if filename is None:
            filename = f'face_landmarks_{int(time.time())}.csv'
        
        # 确保文件保存在输出文件夹中
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['point_id', 'x', 'y', 'z']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 写入表头
                writer.writeheader()
                
                # 遍历所有人脸的landmarks（通常只有一个人脸）
                for face_idx, face_landmarks in enumerate(landmarks_list):
                    for point_id, landmark in enumerate(face_landmarks):
                        writer.writerow({
                            'point_id': point_id,
                            'x': landmark.x,
                            'y': landmark.y, 
                            'z': landmark.z
                        })
                    break  # 只保存第一个检测到的人脸
                
            print(f"Landmarks已保存到: {filepath}")
            print(f"总共保存了 {len(face_landmarks)} 个landmarks点")
            return True
            
        except Exception as e:
            print(f"保存landmarks失败: {e}")
            return False

    def save_landmarks_to_json(self, detection_result, frame_number, timestamp_ms):
        """保存当前帧的landmarks到JSON文件"""
        if not self.enable_json_output:
            return False
            
        if not detection_result.face_landmarks:
            return False
        
        try:
            # 只保存第一个检测到的人脸的前468个点
            face_landmarks = detection_result.face_landmarks[0]
            landmarks_data = []
            
            for i, landmark in enumerate(face_landmarks[:468]):
                landmarks_data.append({
                    "id": i,
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z)
                })
            
            # 构建JSON数据结构
            json_data = {
                "frame": frame_number,
                "timestamp_ms": timestamp_ms,
                "total_landmarks": len(landmarks_data),
                "landmarks": landmarks_data
            }
            
            # 保存到JSON文件
            json_filename = f"{frame_number}.json"
            json_filepath = os.path.join(self.json_dir, json_filename)
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"保存JSON失败 (帧{frame_number}): {e}")
            return False
    
    def clear_json_files(self):
        """清理jsonfile文件夹中的所有JSON文件"""
        try:
            if os.path.exists(self.json_dir):
                for filename in os.listdir(self.json_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(self.json_dir, filename)
                        os.remove(file_path)
                print(f"已清理JSON文件夹: {self.json_dir}")
                return True
        except Exception as e:
            print(f"清理JSON文件失败: {e}")
            return False

    def draw_info_on_image(self, image, detection_result, current_frame=0, total_frames=0, paused=False):
        """在图像上绘制检测信息"""
        height, width, _ = image.shape
        
        # 绘制检测到的人脸数量
        if detection_result.face_landmarks:
            face_count = len(detection_result.face_landmarks)
            image = self.put_chinese_text(image, f'检测到人脸: {face_count}', 
                                        (10, 30), font_size=24, color=(0, 255, 0))
            
            # 如果有面部表情数据，显示一些信息
            if detection_result.face_blendshapes:
                y_offset = 70
                image = self.put_chinese_text(image, '面部表情检测已启用', 
                                            (10, y_offset), font_size=18, color=(255, 255, 0))
        else:
            image = self.put_chinese_text(image, '未检测到人脸', 
                                        (10, 30), font_size=24, color=(0, 0, 255))
        
        # 显示视频进度信息
        if total_frames > 0:
            progress = (current_frame / total_frames) * 100
            progress_text = f'进度: {current_frame}/{total_frames} ({progress:.1f}%)'
            image = self.put_chinese_text(image, progress_text, 
                                        (width - 300, 30), font_size=20, color=(255, 255, 255))
        
        # 显示循环播放状态和循环次数
        loop_status = "循环播放: 开启" if self.loop_playback else "循环播放: 关闭"
        if self.loop_count > 0:
            loop_text = f'{loop_status} (已循环 {self.loop_count} 次)'
        else:
            loop_text = loop_status
        image = self.put_chinese_text(image, loop_text, 
                                    (width - 300, 70), font_size=18, color=(0, 255, 255))
        
        # 显示暂停状态
        if paused:
            image = self.put_chinese_text(image, '已暂停 (按空格继续)', 
                                        (width - 200, 110), font_size=18, color=(255, 255, 0))
        
        # 显示录制状态
        if self.save_output_video:
            image = self.put_chinese_text(image, '正在录制输出视频', 
                                        (10, height - 160), font_size=18, color=(255, 0, 0))
        
        # 显示JSON输出状态
        if self.enable_json_output:
            image = self.put_chinese_text(image, 'JSON输出已启用 (按J关闭)', 
                                        (10, height - 220), font_size=18, color=(0, 255, 255))
        else:
            image = self.put_chinese_text(image, 'JSON输出已关闭 (按J启用)', 
                                        (10, height - 220), font_size=18, color=(128, 128, 128))
        
        # 显示landmarks状态
        if self.show_landmarks:
            image = self.put_chinese_text(image, 'landmarks显示已开启 (按H隐藏)', 
                                        (10, height - 140), font_size=18, color=(0, 255, 0))
        else:
            image = self.put_chinese_text(image, 'landmarks显示已隐藏 (按H显示)', 
                                        (10, height - 140), font_size=18, color=(128, 128, 128))
        
        # 显示边缘模糊状态
        if self.enable_edge_blur:
            image = self.put_chinese_text(image, '边缘滤波已启用 (按B关闭)', 
                                        (10, height - 20), font_size=18, color=(255, 192, 0))
        else:
            image = self.put_chinese_text(image, '边缘滤波已关闭 (按B启用)', 
                                        (10, height - 20), font_size=18, color=(128, 128, 128))
        
        # 显示透视投影状态
        if self.enable_perspective_projection:
            perspective_text = f'透视投影已启用 (强度:{self.perspective_strength:.2f}, 增强:{self.depth_enhancement:.1f})'
            image = self.put_chinese_text(image, perspective_text, 
                                        (10, height - 40), font_size=18, color=(255, 128, 255))
        else:
            image = self.put_chinese_text(image, '透视投影已关闭 (弱透视模式)', 
                                        (10, height - 40), font_size=18, color=(128, 128, 128))
        
        # 显示landmarks缩放状态
        if self.warp_ready and self.show_warped:
            scale_text = f'landmarks缩放: {self.landmarks_scale:.2f}x (按[/]调整)'
            if self.enable_pixel_warp:
                y_pos = height - 180
            else:
                y_pos = height - 140
            image = self.put_chinese_text(image, scale_text, 
                                        (10, y_pos), font_size=18, color=(255, 255, 0))
        
        # 显示宽度比例状态（所有模式下都显示）
        width_text = f'宽度比例: {self.width_scale:.2f}x (按3/4调整)'
        if self.warp_ready and self.show_warped:
            if self.enable_pixel_warp:
                y_pos = height - 200
            else:
                y_pos = height - 160
        else:
            y_pos = height - 80
        image = self.put_chinese_text(image, width_text, 
                                    (10, y_pos), font_size=18, color=(255, 192, 255))
        
        # 显示变形状态
        if self.warp_ready:
            if self.show_warped:
                if self.enable_pixel_warp:
                    image = self.put_chinese_text(image, '像素级人脸变形已启用 (按P关闭)', 
                                                (10, height - 100), font_size=18, color=(255, 0, 255))
                    image = self.put_chinese_text(image, '脸型变形已启用 (按X切换)', 
                                                (10, height - 60), font_size=18, color=(0, 255, 255))
                else:
                    image = self.put_chinese_text(image, '脸型变形已启用 (按X切换, P启用像素变形)', 
                                                (10, height - 60), font_size=18, color=(0, 255, 255))
            else:
                image = self.put_chinese_text(image, '显示原始landmarks (按X切换)', 
                                            (10, height - 60), font_size=18, color=(255, 128, 0))
        elif self.frame_count > 0 and self.frame_count < self.N:
            image = self.put_chinese_text(image, f'正在收集landmarks: {self.frame_count}/{self.N}', 
                                        (10, height - 60), font_size=18, color=(255, 255, 0))
        else:
            image = self.put_chinese_text(image, '按M键开始变形检测', 
                                        (10, height - 60), font_size=18, color=(128, 128, 128))
        
        return image

    def init_video_writer(self, width, height, fps):
        """初始化视频输出写入器"""
        try:
            # 生成输出视频文件名
            timestamp = int(time.time())
            output_video_name = f'processed_video_{timestamp}.mp4'
            output_video_path = os.path.join(self.output_dir, output_video_name)
            
            # 设置编码器和输出参数
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_video_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, (width, height)
            )
            
            if self.output_video_writer.isOpened():
                print(f"开始录制输出视频: {output_video_path}")
                return True
            else:
                print("无法初始化视频写入器")
                return False
        except Exception as e:
            print(f"初始化视频写入器失败: {e}")
            return False

    def cleanup_video_writer(self):
        """清理视频写入器"""
        if self.output_video_writer:
            self.output_video_writer.release()
            self.output_video_writer = None
            print("视频录制已停止")

    def update_video_aspect_ratio(self, video_width, video_height):
        """根据实际视频分辨率更新宽高比参数"""
        self.video_width = video_width
        self.video_height = video_height
        self.aspect_ratio = video_width / video_height
        
        # 计算x坐标修正系数
        # MediaPipe landmarks是基于正方形坐标系(1:1)归一化的
        # 需要根据实际视频的宽高比进行修正
        self.x_scale_factor = self.aspect_ratio / 1.0
        
        print(f"宽高比参数已更新:")
        print(f"  实际分辨率: {video_width}x{video_height}")
        print(f"  宽高比: {self.aspect_ratio:.3f}")
        print(f"  X坐标修正系数: {self.x_scale_factor:.3f}")

    def run(self, video_path=None):
        """运行视频文件人脸检测"""
        if not self.landmarker:
            print("人脸标志检测器初始化失败，请检查模型文件")
            print("请确保模型文件 'face_landmarker.task' 存在且完整")
            return
        
        # 确定视频文件路径
        if video_path:
            self.video_path = video_path
        
        if not self.video_path:
            print("错误：未指定视频文件路径")
            print("使用方法: python face_landmarker_video.py <video_file.mp4>")
            return
        
        if not os.path.exists(self.video_path):
            print(f"错误：视频文件不存在: {self.video_path}")
            return
        
        if not self.video_path.lower().endswith('.mp4'):
            print("错误：只支持 .mp4 格式的视频文件")
            return
        
        # 打开视频文件
        print(f"正在打开视频文件: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"错误：无法打开视频文件: {self.video_path}")
            print("请检查：")
            print("1. 视频文件路径是否正确")
            print("2. 视频文件是否损坏")
            print("3. 视频编码格式是否支持")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 重要：根据实际视频分辨率更新宽高比参数
        self.update_video_aspect_ratio(video_width, video_height)
        
        print(f"视频信息:")
        print(f"  分辨率: {video_width}x{video_height}")
        print(f"  帧率: {fps:.2f} FPS")
        print(f"  总帧数: {total_frames}")
        print(f"  视频时长: {total_frames/fps:.2f} 秒")
        print(f"  循环播放: {'启用' if self.loop_playback else '禁用'}")
        
        frame_count = 0
        paused = False
        
        # 如果启用了JSON输出且设置了启动时清理，则清理之前的JSON文件
        if self.enable_json_output and self.clear_json_on_start:
            self.clear_json_files()
        
        print("开始处理视频，按键控制:")
        print("  'Q' 或 ESC - 退出")
        print("  'SPACE' - 暂停/继续")
        print("  'S' - 保存当前帧")
        print("  'R' - 开始/停止录制输出视频")
        print("  'C' - 切换循环播放模式")
        print("  'M' - 重新检测并计算变换")
        print("  'X' - 切换原始/变形显示")
        print("  'P' - 切换像素级变形")
        print("  'H' - 隐藏/显示landmarks线框")
        print("  '[/]' - 调整landmarks整体缩放")
        print("  '3/4' - 调整landmarks宽度比例")
        print("  'T' - 切换透视投影/弱透视投影")
        print("  '5/6' - 调整透视强度")
        print("  '7/8' - 调整深度增强")
        print("  'J/j' - 开启/关闭JSON输出")
        print("  'K/k' - 清理所有JSON文件")
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        if self.loop_playback:
                            # 视频结束，重新开始播放
                            print(f"第 {self.loop_count + 1} 次播放完成，重新开始循环播放...")
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到视频开始位置
                            frame_count = 0  # 重置帧计数
                            self.loop_count += 1  # 增加循环计数
                            
                            # 重新读取第一帧
                            ret, frame = cap.read()
                            if not ret:
                                print("错误：无法重新读取视频第一帧")
                                break
                            frame_count = 1
                        else:
                            print("视频播放完成，循环播放已禁用")
                            break
                    else:
                        frame_count += 1
                
                # 转换为RGB格式
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 创建MediaPipe图像对象
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # 计算时间戳（基于帧数和帧率）
                frame_timestamp_ms = int((frame_count / fps) * 1000)
                
                # 进行检测
                detection_result = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                # 保存当前帧的landmarks到JSON文件
                if self.enable_json_output:
                    self.save_landmarks_to_json(detection_result, frame_count, frame_timestamp_ms)
                
                # 处理检测结果（用于变形功能）
                if detection_result.face_landmarks and not self.warp_ready:
                    # 仅使用第一个检测到的人脸
                    landmarks = detection_result.face_landmarks[0]
                    # 只使用前468个landmarks，避免478 vs 468的维度不匹配问题
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks[:468]], dtype=np.float32)
                    self.landmark_buffer.append(coords)
                    self.frame_count += 1
                    
                    if self.frame_count >= self.N:
                        print("开始计算相似变换...")
                        avg_landmarks = np.mean(self.landmark_buffer, axis=0)
                        self.estimate_transform(avg_landmarks)
                        self.landmark_buffer = []
                
                # 绘制标志点
                annotated_image = self.draw_landmarks_on_image(rgb_frame, detection_result)
                
                # 转换回BGR格式用于显示
                annotated_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                
                # 绘制信息
                display_frame = self.draw_info_on_image(
                    annotated_frame, detection_result, frame_count, total_frames, paused
                )
                
                # 保存到输出视频（如果启用）
                if self.save_output_video and self.output_video_writer:
                    self.output_video_writer.write(display_frame)
                
                # 显示帧
                cv2.imshow('MediaPipe 人脸标志检测 - 视频处理', display_frame)
                
                # 处理按键
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                if key == ord('q') or key == 27:  # 'q' 或 ESC 键退出
                    break
                elif key == ord(' '):  # 空格键暂停/继续
                    paused = not paused
                    if paused:
                        print("视频已暂停，按空格键继续")
                    else:
                        print("视频继续播放")
                elif key == ord('s'):  # 's' 键保存图像
                    save_name = f'video_frame_{frame_count}_{int(time.time())}.jpg'
                    save_path = os.path.join(self.output_dir, save_name)
                    cv2.imwrite(save_path, display_frame)
                    print(f"图像已保存: {save_path}")
                elif key == ord('l'):  # 'l' 键保存landmarks
                    if detection_result.face_landmarks:
                        csv_filename = f'video_landmarks_{frame_count}_{int(time.time())}.csv'
                        if self.save_landmarks_to_csv(detection_result.face_landmarks, csv_filename):
                            print(f"Landmarks已保存到: {csv_filename}")
                        else:
                            print("保存landmarks失败")
                    else:
                        print("没有可用的landmarks数据")
                elif key == ord('M') or key == ord('m'):  # 'M' 或 'm' 键重新检测前60帧并计算变换
                    self.reset_transform()
                    print("开始重新检测前60帧...")
                elif key == ord('X') or key == ord('x'):  # 'X' 或 'x' 键切换显示模式
                    if self.warp_ready:
                        self.show_warped = not self.show_warped
                        if self.show_warped:
                            print("切换到变形landmarks显示")
                        else:
                            print("切换到原始landmarks显示")
                    else:
                        print("变形功能未启用，请先按M键进行检测")
                elif key == ord('P') or key == ord('p'):  # 'P' 或 'p' 键切换像素变形
                    if self.warp_ready and self.show_warped:
                        self.enable_pixel_warp = not self.enable_pixel_warp
                        if self.enable_pixel_warp:
                            print("像素级人脸变形已启用")
                            # 重置平滑处理的缓存
                            self.previous_landmarks = None
                        else:
                            print("像素级人脸变形已关闭")
                    else:
                        print("请先启用变形功能（按M键检测，按X键切换到变形显示）")
                elif key == ord('H') or key == ord('h'):  # 'H' 或 'h' 键切换landmarks显示
                    self.show_landmarks = not self.show_landmarks
                    if self.show_landmarks:
                        print("landmarks线框显示已开启")
                    else:
                        print("landmarks线框显示已隐藏")
                elif key == ord('['):  # '[' 键缩小landmarks
                    if self.warp_ready and self.show_warped:
                        self.landmarks_scale = max(0.1, self.landmarks_scale - self.scale_step)
                        print(f"landmarks缩放调整为: {self.landmarks_scale:.2f}x")
                    else:
                        print("请先启用变形功能")
                elif key == ord(']'):  # ']' 键放大landmarks
                    if self.warp_ready and self.show_warped:
                        self.landmarks_scale = min(3.0, self.landmarks_scale + self.scale_step)
                        print(f"landmarks缩放调整为: {self.landmarks_scale:.2f}x")
                    else:
                        print("请先启用变形功能")
                elif key == ord('B') or key == ord('b'):  # 'B' 或 'b' 键切换边缘模糊
                    if self.warp_ready and self.show_warped and self.enable_pixel_warp:
                        self.enable_edge_blur = not self.enable_edge_blur
                        if self.enable_edge_blur:
                            print("边缘滤波已启用")
                        else:
                            print("边缘滤波已关闭")
                    else:
                        print("请先启用像素级人脸变形功能")
                elif key == ord('R') or key == ord('r'):  # 'R' 或 'r' 键切换录制
                    if not self.save_output_video:
                        # 开始录制
                        if self.init_video_writer(video_width, video_height, fps):
                            self.save_output_video = True
                        else:
                            print("无法开始录制")
                    else:
                        # 停止录制
                        self.save_output_video = False
                        self.cleanup_video_writer()
                elif key == ord('C'):  # 'C' 键切换循环播放模式
                    self.loop_playback = not self.loop_playback
                    if self.loop_playback:
                        print("循环播放已启用")
                    else:
                        print("循环播放已停止")
                elif key == ord('3'):  # '3' 键减小宽度比例（让landmarks更窄）
                    self.width_scale = max(0.1, self.width_scale - self.width_scale_step)
                    print(f"宽度比例调整为: {self.width_scale:.2f}x")
                elif key == ord('4'):  # '4' 键增大宽度比例（让landmarks更宽）
                    self.width_scale = min(3.0, self.width_scale + self.width_scale_step)
                    print(f"宽度比例调整为: {self.width_scale:.2f}x")
                elif key == ord('T'):  # 'T' 键切换透视投影/弱透视投影
                    self.enable_perspective_projection = not self.enable_perspective_projection
                    if self.enable_perspective_projection:
                        print("透视投影已启用")
                    else:
                        print("透视投影已关闭")
                elif key == ord('5'):  # '5' 键减小透视强度
                    self.perspective_strength = max(0.0, self.perspective_strength - 0.1)
                    print(f"透视强度调整为: {self.perspective_strength:.2f}")
                elif key == ord('6'):  # '6' 键增大透视强度
                    self.perspective_strength = min(1.0, self.perspective_strength + 0.1)
                    print(f"透视强度调整为: {self.perspective_strength:.2f}")
                elif key == ord('7'):  # '7' 键减小深度增强
                    self.depth_enhancement = max(1.0, self.depth_enhancement - 1.0)
                    print(f"深度增强调整为: {self.depth_enhancement:.2f}")
                elif key == ord('8'):  # '8' 键增大深度增强
                    self.depth_enhancement = min(20.0, self.depth_enhancement + 1.0)
                    print(f"深度增强调整为: {self.depth_enhancement:.2f}")
                elif key == ord('J') or key == ord('j'):  # 'J' 或 'j' 键切换JSON输出
                    self.enable_json_output = not self.enable_json_output
                    if self.enable_json_output:
                        print("JSON输出已启用，开始保存每帧landmarks到JSON文件")
                        # 如果是首次启用且设置了清理，则清理旧文件
                        if self.clear_json_on_start:
                            self.clear_json_files()
                    else:
                        print("JSON输出已关闭")
                elif key == ord('K') or key == ord('k'):  # 'K' 或 'k' 键清理JSON文件
                    if self.clear_json_files():
                        print("所有JSON文件已清理")
                    else:
                        print("清理JSON文件失败")
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            if self.save_output_video:
                self.cleanup_video_writer()
            print("视频处理完成")

    def load_obj_vertices(self, path):
        """
        从 OBJ 文件加载顶点 (v x y z)，返回 (N,3) numpy 数组
        """
        verts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    verts.append([x, y, z])
        return np.array(verts, dtype=np.float32)

    def compute_similarity_transform(self, src, dst):
        """
        计算 src->dst 的相似变换 (R, s, t)，src,dst: (N,3)
        """
        src_centroid = np.mean(src, axis=0)
        dst_centroid = np.mean(dst, axis=0)
        src_c = src - src_centroid
        dst_c = dst - dst_centroid
        cov = src_c.T @ dst_c
        U, S, Vt = np.linalg.svd(cov)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        s = np.sum(S) / np.sum(src_c**2)
        t = dst_centroid - s * R @ src_centroid
        return R, s, t

    def estimate_transform(self, avg_landmarks):
        """
        使用非刚性拉伸方案：将自定义模型对齐到活人脸，计算形状差异
        """
        if self.custom_vertices is None:
            print("自定义模型未加载，无法计算变换")
            return
        
        print("开始计算自定义模型到活人脸的变换...")
        print(f"视频分辨率: {self.video_width}x{self.video_height}")
        print(f"视频宽高比: {self.aspect_ratio:.3f}")
        print(f"X坐标修正系数: {self.x_scale_factor:.3f}")
        
        # 修正平均landmarks的x坐标以适应宽高比
        corrected_landmarks = avg_landmarks.copy()
        corrected_landmarks[:, 0] *= self.x_scale_factor  # 只修正x坐标
        print(f"X坐标修正前范围: [{avg_landmarks[:, 0].min():.4f}, {avg_landmarks[:, 0].max():.4f}]")
        print(f"X坐标修正后范围: [{corrected_landmarks[:, 0].min():.4f}, {corrected_landmarks[:, 0].max():.4f}]")
        
        # 计算从自定义模型到活人脸的相似变换
        R, s, t = self.compute_similarity_transform(self.custom_vertices, corrected_landmarks)
        
        # 将自定义模型变换到活人脸坐标系
        custom_in_live = (s * (R @ self.custom_vertices.T).T) + t
        
        # 计算形状差异向量：变换后的自定义模型 - 修正后的活人脸平均位置
        self.diff_transformed = custom_in_live - corrected_landmarks
        
        # 将差异的x坐标还原到原始坐标系
        self.diff_transformed[:, 0] /= self.x_scale_factor
        self.warp_ready = True
        
        # 保存变换参数到文件
        try:
            np.save(self.transform_file, self.diff_transformed)
            print(f"变换参数已保存到: {self.transform_file}")
        except Exception as e:
            print(f"保存变换参数失败: {e}")
        
        print("已计算自定义模型形状差异，实时变形已启用")
        print(f"形状差异范围: X=[{self.diff_transformed[:, 0].min():.4f}, {self.diff_transformed[:, 0].max():.4f}]")
        print(f"              Y=[{self.diff_transformed[:, 1].min():.4f}, {self.diff_transformed[:, 1].max():.4f}]")

    def load_chinese_font(self):
        """加载中文字体用于GUI显示"""
        try:
            # Windows系统的中文字体路径
            if platform.system() == "Windows":
                font_paths = [
                    "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
                    "C:/Windows/Fonts/simsun.ttc",    # 宋体
                    "C:/Windows/Fonts/simhei.ttf",    # 黑体
                ]
            else:
                # Linux/Mac 系统可以添加相应的字体路径
                font_paths = [
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "/System/Library/Fonts/PingFang.ttc",  # Mac
                ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, 24)
            
            # 如果找不到字体，使用默认字体
            return ImageFont.load_default()
        except Exception as e:
            print(f"加载中文字体失败: {e}")
            return ImageFont.load_default()
    
    def put_chinese_text(self, img, text, position, font_size=24, color=(0, 255, 0)):
        """在图像上绘制中文文字"""
        try:
            # 将OpenCV图像转换为PIL图像
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 如果需要特定大小的字体
            if hasattr(self, 'chinese_font'):
                try:
                    font = ImageFont.truetype(self.chinese_font.path, font_size)
                except:
                    font = self.chinese_font
            else:
                font = ImageFont.load_default()
            
            # 绘制文字
            draw.text(position, text, font=font, fill=color)
            
            # 转换回OpenCV格式
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img_cv
        except Exception as e:
            print(f"绘制中文文字失败: {e}")
            # 如果失败，使用英文替代
            cv2.putText(img, text.encode('ascii', 'ignore').decode('ascii'), 
                       position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return img

    def load_saved_transform(self):
        """加载已保存的变换参数"""
        try:
            if os.path.exists(self.transform_file):
                self.diff_transformed = np.load(self.transform_file)
                if self.custom_vertices is not None:
                    self.warp_ready = True
                    print(f"已加载保存的变换参数: {self.transform_file}")
                    print("脸型变形功能已启用，按M键重新检测")
                else:
                    print("自定义模型未加载，无法启用变换")
            else:
                print("未找到保存的变换参数，请按M键开始检测")
        except Exception as e:
            print(f"加载变换参数失败: {e}")

    def reset_transform(self):
        """重置变换参数"""
        self.landmark_buffer = []
        self.frame_count = 0
        self.warp_ready = False
        self.diff_transformed = None
        self.enable_pixel_warp = False  # 重置像素变形状态
        self.previous_landmarks = None  # 重置平滑处理缓存
        print("已重置变换参数")


def main():
    """主函数"""
    print("MediaPipe 人脸标志检测器 - 视频处理版")
    print("=" * 50)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MediaPipe 人脸标志检测器 - 视频处理版')
    parser.add_argument('video', nargs='?', help='输入视频文件路径 (.mp4)')
    args = parser.parse_args()
    
    # 检查是否提供了视频文件路径
    if not args.video:
        print("使用方法: python face_landmarker_video.py <video_file.mp4>")
        print("示例: python face_landmarker_video.py video/sample.mp4")
        return
    
    # 创建并运行检测器
    detector = FaceLandmarkerVideo()
    detector.run(args.video)


if __name__ == "__main__":
    main() 