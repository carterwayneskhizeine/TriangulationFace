#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe 人脸标志检测器 - 离线视频处理版
从视频文件逐帧离线处理并输出到视频文件 python = 3.9
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
import os
import csv
from PIL import Image, ImageDraw, ImageFont
import platform
from face_warper import FaceWarper
import glob

# 按照官方文档推荐的导入方式
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class FaceLandmarkerVideoOffline:
    def __init__(self):
        """初始化人脸标志检测器 - 离线视频处理版"""
        
        # ==================== 配置参数区域 ====================
        # 输入输出文件路径配置
        self.input_video_path = "video/test.mp4"  # 输入视频文件路径
        self.output_frames_dir = "video_output_file"  # 输出帧文件夹
        self.output_video_name = "processed_output02.mp4"  # 最终输出视频文件名
        
        # 处理配置参数
        self.enable_alignment_detection = False  # 不启用前60帧对齐检测
        self.alignment_frames = 60  # 用于对齐检测的帧数
        self.auto_enable_pixel_warp = True  # 对齐后是否自动启用像素级变形
        self.auto_enable_face_only_mode = True  # 对齐后是否自动启用面部专用模式
        
        # 默认处理参数 - 直接启用所需功能
        self.enable_pixel_warp = True  # 直接启用像素级变形
        self.enable_face_only_mode = True  # 直接启用面部专用模式（黑色背景）
        self.enable_lambert_material = False  # 不使用Lambert材质渲染
        self.enable_wireframe = False  # 不显示黑色线框
        self.wireframe_thickness = 0.2  # 线框粗细
        self.enable_edge_blur = False  # 边缘滤波
        self.show_landmarks = False  # 不显示landmarks点和连线（节省处理时间）
        self.enable_perspective_projection = True  # 透视投影
        self.perspective_strength = 0.2  # 透视强度
        self.depth_enhancement = 3.0  # 深度增强系数
        self.landmarks_scale = 1.0  # landmarks缩放系数
        self.width_scale = 1.0  # 宽度缩放系数
        
        # 视频处理参数
        self.output_fps = 30.0  # 输出视频帧率
        self.output_quality = 95  # 输出质量(0-100)
        self.frame_skip = 1  # 帧跳过数量(1=不跳过,2=每隔一帧处理)
        
        # 进度显示配置
        self.show_progress = True  # 是否显示处理进度
        self.progress_interval = 30  # 每多少帧显示一次进度
        
        # ==================== 配置参数区域结束 ====================
        
        self.result = None
        self.output_image = None
        self.timestamp = 0
        
        # 创建输出文件夹
        if not os.path.exists(self.output_frames_dir):
            os.makedirs(self.output_frames_dir)
            print(f"创建输出文件夹: {self.output_frames_dir}")
        
        # 清理旧的输出帧
        self.cleanup_old_frames()
        
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
        self.warp_ready = False
        self.diff_transformed = None
        self.transform_file = "face_transform.npy"  # 保存变换参数的文件
        
        # 加载中文字体用于GUI显示
        self.chinese_font = self.load_chinese_font()
        
        # 尝试加载已保存的变换参数
        if self.enable_alignment_detection:
            self.load_saved_transform()
        else:
            # 不启用对齐检测时，直接尝试加载已保存的变换参数
            print("未启用对齐检测，尝试加载已保存的变换参数...")
            self.load_saved_transform()
            
            # 如果没有加载到变换参数但仍需要进行像素级变形，创建默认变换
            if not self.warp_ready and self.enable_pixel_warp:
                print("未找到保存的变换参数，将使用原始landmarks进行像素级变形")
                print("注意：建议先运行一次带对齐检测的版本来生成变换参数")
                # 设置为就绪状态，使用原始landmarks
                self.warp_ready = True
                self.diff_transformed = None  # 不使用形状差异，保持原始脸型
            
            # 直接启用所需功能
            if self.auto_enable_face_only_mode:
                self.enable_face_only_mode = True
                print("直接启用面部专用模式（黑色背景）")
            
            if self.auto_enable_pixel_warp:
                self.enable_pixel_warp = True
                print("直接启用像素级人脸变形")
        
        # 视频分辨率参数（将在process_video方法中根据实际视频分辨率更新）
        self.video_width = 640   # 默认视频分辨率
        self.video_height = 480  # 默认视频分辨率
        self.aspect_ratio = self.video_width / self.video_height
        self.x_scale_factor = self.aspect_ratio / 1.0  # 用于修正x坐标的拉伸，会根据实际视频更新
        
        # 变形显示控制
        self.show_warped = True  # True=显示变形后，False=显示原始
        
        # 初始化人脸变形器
        self.face_warper = FaceWarper()
        
        # 像素变形控制
        self.previous_landmarks = None  # 用于平滑处理
        
        # FPS统计
        self.processed_frames = 0
        self.start_time = time.time()
        
        # Face Geometry 模块
        self.enable_face_geometry = True  # 是否启用 Face Geometry 模块
        self.geometry_matrices = None
        
        # 【关键修改】加载真实相机校准参数
        self.use_real_calibration = True  # 是否使用真实校准参数
        self.calibration_intrinsic_path = "Camera-Calibration/output/intrinsic.txt"  # 内参文件路径
        self.calibration_extrinsic_path = "Camera-Calibration/output/extrinsic.txt"  # 外参文件路径
        
        # 相机参数（将根据真实校准或手动设置）
        self.camera_fx = None
        self.camera_fy = None
        self.camera_cx = None
        self.camera_cy = None
        self.camera_skew = 0.0  # 倾斜参数
        
        # 加载相机校准参数
        self.load_camera_calibration()
        
        print("人脸标志检测器初始化完成 - 离线视频处理模式")
        print(f"输入视频: {self.input_video_path}")
        print(f"输出文件夹: {self.output_frames_dir}")
        print(f"输出视频: {self.output_video_name}")
        print(f"对齐检测: {'启用' if self.enable_alignment_detection else '关闭'}")
        print(f"像素级变形: {'启用' if self.enable_pixel_warp else '关闭'}")
        print(f"面部专用模式(黑色背景): {'启用' if self.enable_face_only_mode else '关闭'}")
        print(f"Lambert材质: {'启用' if self.enable_lambert_material else '关闭'}")
        print(f"透视投影: {'启用' if self.enable_perspective_projection else '关闭'}")
        print(f"线框显示: {'启用' if self.enable_wireframe else '关闭'}")
        print(f"相机校准: {'使用真实校准' if self.use_real_calibration else '使用手动估计'}")
        print("=" * 60)

    def cleanup_old_frames(self):
        """清理旧的输出帧文件"""
        try:
            old_frames = glob.glob(os.path.join(self.output_frames_dir, "frame_*.jpg"))
            if old_frames:
                for frame_file in old_frames:
                    os.remove(frame_file)
                print(f"清理了 {len(old_frames)} 个旧的帧文件")
        except Exception as e:
            print(f"清理旧帧文件失败: {e}")

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

    def process_frame_landmarks(self, detection_result, frame_number):
        """处理单帧的landmarks数据"""
        try:
            # 采集前 N 帧 landmarks 并估计变换
            if (detection_result.face_landmarks and 
                self.enable_alignment_detection and 
                not self.warp_ready and 
                frame_number <= self.alignment_frames):
                
                # 仅使用第一个检测到的人脸
                landmarks = detection_result.face_landmarks[0]
                # 只使用前468个landmarks，避免478 vs 468的维度不匹配问题
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks[:468]], dtype=np.float32)
                self.landmark_buffer.append(coords)
                self.frame_count += 1
                
                if self.show_progress and self.frame_count % 10 == 0:
                    print(f"收集第 {self.frame_count}/{self.alignment_frames} 帧 landmarks")
                
                if self.frame_count >= self.alignment_frames:
                    print("开始计算相似变换...")
                    avg_landmarks = np.mean(self.landmark_buffer, axis=0)
                    self.estimate_transform(avg_landmarks)
                    self.landmark_buffer = []
                    
                    # 自动启用像素级变形和面部专用模式
                    if self.auto_enable_pixel_warp:
                        self.enable_pixel_warp = True
                        print("自动启用像素级人脸变形")
                    
                    if self.auto_enable_face_only_mode:
                        self.enable_face_only_mode = True
                        print("自动启用面部专用模式")
            
            # 如果启用 Face Geometry 模块，获取面部转换矩阵（4x4）
            if self.enable_face_geometry and hasattr(detection_result, 'facial_transformation_matrixes'):
                try:
                    self.geometry_matrices = detection_result.facial_transformation_matrixes
                except Exception as e:
                    if self.show_progress and frame_number % 100 == 0:
                        print(f"获取面部转换矩阵失败: {e}")
                        
        except Exception as e:
            print(f"处理landmarks错误 (帧 {frame_number}): {e}")

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """在图像上绘制人脸标志点并应用像素级变形"""
        try:
            # 如果启用像素变形且变换已就绪，进行像素级变形
            if (self.enable_pixel_warp and self.warp_ready and 
                self.show_warped and detection_result.face_landmarks):
                
                # 进行像素级人脸变形（即使没有形状差异参数也可以进行）
                warped_image = self.apply_pixel_warp(rgb_image, detection_result)
                annotated_image = np.copy(warped_image)
            else:
                annotated_image = np.copy(rgb_image)
            
            # 绘制人脸标志点（在离线模式下通常不需要，节省处理时间）
            if self.show_landmarks and detection_result.face_landmarks:
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
            return rgb_image  # 返回原始图像

    def apply_perspective_warp_to_landmarks(self, src_landmarks, dst_landmarks):
        """对landmarks应用透视投影变形"""
        try:
            if not self.enable_perspective_projection:
                return dst_landmarks
            
            perspective_dst = dst_landmarks.copy()
            
            # 【重新设计】使用真正的透视投影公式，基于真实相机参数
            # 使用真实的相机参数或回退到估计参数
            if self.camera_fx is not None and self.camera_fy is not None:
                # 使用真实校准的相机参数
                focal_length_x = self.camera_fx
                focal_length_y = self.camera_fy
                principal_point_x = self.camera_cx
                principal_point_y = self.camera_cy
                print_debug = False  # 避免过多输出，只在第一帧显示
            else:
                # 回退到手动估计
                focal_length_x = min(self.video_width, self.video_height) * 0.8
                focal_length_y = focal_length_x
                principal_point_x = self.video_width / 2.0
                principal_point_y = self.video_height / 2.0
                print_debug = False
            
            # 3D投影参数
            base_depth = 50.0  # 基础深度（厘米）
            depth_variation = 5.0  # 深度变化范围（厘米）
            
            if print_debug:
                print(f"透视投影参数:")
                print(f"  fx: {focal_length_x:.2f}, fy: {focal_length_y:.2f}")
                print(f"  cx: {principal_point_x:.2f}, cy: {principal_point_y:.2f}")
             
            # 对每个landmark应用透视变换
            for i in range(len(dst_landmarks)):
                x_norm, y_norm, z_norm = dst_landmarks[i]
                
                # 转换为像素坐标
                x_pixel = x_norm * self.video_width
                y_pixel = y_norm * self.video_height
                
                # 【新算法】计算3D坐标
                # Z值：负值表示离相机近，正值表示离相机远
                z_3d = base_depth + (z_norm * depth_variation)
                
                # 将像素坐标转换为相机坐标系（使用真实的主点位置）
                x_3d = (x_pixel - principal_point_x) * z_3d / focal_length_x
                y_3d = (y_pixel - principal_point_y) * z_3d / focal_length_y
                
                # 【关键】应用透视投影：P' = f * P / Z
                # 这里我们通过调整Z来产生透视效果
                perspective_z = z_3d * (1.0 + z_norm * self.perspective_strength * self.depth_enhancement)  # 根据原始Z值调整透视深度
                
                # 重新投影到2D（使用对应的焦距）
                new_x_pixel = (x_3d * focal_length_x / perspective_z) + principal_point_x
                new_y_pixel = (y_3d * focal_length_y / perspective_z) + principal_point_y
                
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
            
            # 如果有形状差异参数，应用形状差异变换
            if self.diff_transformed is not None:
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
            
            # 应用人脸变形
            warped_image = self.face_warper.apply_face_warp(
                rgb_image,
                original_coords,
                warped_coords,
                blend_ratio=1.0,  # 完全替换
                enable_blur_edge=self.enable_edge_blur,
                enable_wireframe=self.enable_wireframe,
                use_lambert_material=self.enable_lambert_material,
                face_only_mode=self.enable_face_only_mode,
                wireframe_thickness=self.wireframe_thickness
            )
            
            return warped_image
            
        except Exception as e:
            print(f"像素变形失败: {e}")
            return rgb_image

    def update_video_aspect_ratio(self, video_width, video_height):
        """根据实际视频分辨率更新宽高比参数"""
        self.video_width = video_width
        self.video_height = video_height
        self.aspect_ratio = video_width / video_height
        
        # 计算x坐标修正系数
        # MediaPipe landmarks是基于正方形坐标系(1:1)归一化的
        # 需要根据实际视频的宽高比进行修正
        self.x_scale_factor = self.aspect_ratio / 1.0  # 对于16:9，约为1.777
        
        # 【修改】只在未使用真实校准参数时才更新相机参数
        if not self.use_real_calibration or self.camera_fx is None:
            # 正确设置相机参数（遵循标准透视投影矩阵）
            # 焦距应该基于较小的尺寸来保持比例正确
            base_focal_length = min(video_width, video_height)
            self.camera_fx = base_focal_length  # 使用较小尺寸作为基准焦距
            self.camera_fy = base_focal_length  # Y方向使用相同焦距
            self.camera_cx = video_width / 2.0   # 主点X坐标（图像中心）
            self.camera_cy = video_height / 2.0  # 主点Y坐标（图像中心）
            
            print(f"视频宽高比参数已更新 (手动估计模式):")
        else:
            print(f"视频宽高比参数已更新 (保持真实校准参数):")
        
        print(f"  实际分辨率: {video_width}x{video_height}")
        print(f"  宽高比: {self.aspect_ratio:.3f}")
        print(f"  X坐标修正系数: {self.x_scale_factor:.3f}")
        
        # 显示当前使用的相机参数
        if self.camera_fx is not None:
            print(f"  当前相机参数:")
            print(f"    fx: {self.camera_fx:.2f}")
            print(f"    fy: {self.camera_fy:.2f}")
            print(f"    cx: {self.camera_cx:.2f}")
            print(f"    cy: {self.camera_cy:.2f}")
            if self.camera_skew != 0.0:
                print(f"    skew: {self.camera_skew:.4f}")
        else:
            print("  警告：相机参数未设置")

    def load_obj_vertices(self, path):
        """从 OBJ 文件加载顶点 (v x y z)，返回 (N,3) numpy 数组"""
        verts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    verts.append([x, y, z])
        return np.array(verts, dtype=np.float32)

    def compute_similarity_transform(self, src, dst):
        """计算 src->dst 的相似变换 (R, s, t)，src,dst: (N,3)"""
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
        """使用非刚性拉伸方案：将自定义模型对齐到活人脸，计算形状差异"""
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

    def load_saved_transform(self):
        """加载已保存的变换参数"""
        try:
            if os.path.exists(self.transform_file):
                self.diff_transformed = np.load(self.transform_file)
                if self.custom_vertices is not None:
                    self.warp_ready = True
                    print(f"已加载保存的变换参数: {self.transform_file}")
                    print("脸型变形功能已启用")
                    
                    # 如果加载了变换参数，跳过对齐检测
                    self.enable_alignment_detection = False
                    
                    # 自动启用像素级变形和面部专用模式
                    if self.auto_enable_pixel_warp:
                        self.enable_pixel_warp = True
                        print("自动启用像素级人脸变形")
                    
                    if self.auto_enable_face_only_mode:
                        self.enable_face_only_mode = True
                        print("自动启用面部专用模式")
                else:
                    print("自定义模型未加载，无法启用变换")
            else:
                print("未找到保存的变换参数，将进行对齐检测")
        except Exception as e:
            print(f"加载变换参数失败: {e}")

    def save_frame(self, image, frame_number):
        """保存单帧图像"""
        try:
            # 生成帧文件名（确保排序正确）
            frame_filename = f"frame_{frame_number:06d}.jpg"
            frame_path = os.path.join(self.output_frames_dir, frame_filename)
            
            # 转换为BGR格式并保存
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, bgr_image, [cv2.IMWRITE_JPEG_QUALITY, self.output_quality])
            
            return True
        except Exception as e:
            print(f"保存帧 {frame_number} 失败: {e}")
            return False

    def create_output_video(self):
        """将处理后的帧合成为最终视频"""
        try:
            print("开始合成最终视频...")
            
            # 获取所有帧文件
            frame_files = sorted(glob.glob(os.path.join(self.output_frames_dir, "frame_*.jpg")))
            
            if not frame_files:
                print("没有找到处理后的帧文件")
                return False
            
            # 读取第一帧获取尺寸
            first_frame = cv2.imread(frame_files[0])
            height, width, _ = first_frame.shape
            
            # 创建视频写入器
            output_video_path = os.path.join(self.output_frames_dir, self.output_video_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, self.output_fps, (width, height))
            
            if not video_writer.isOpened():
                print("无法创建输出视频文件")
                return False
            
            # 逐帧写入视频
            for i, frame_file in enumerate(frame_files):
                frame = cv2.imread(frame_file)
                if frame is not None:
                    video_writer.write(frame)
                    
                    if self.show_progress and (i + 1) % 100 == 0:
                        print(f"合成进度: {i + 1}/{len(frame_files)} 帧")
                else:
                    print(f"警告: 无法读取帧文件 {frame_file}")
            
            # 释放资源
            video_writer.release()
            print(f"视频合成完成: {output_video_path}")
            print(f"总共合成 {len(frame_files)} 帧")
            
            # 合成成功后，删除所有中间jpg帧文件
            print("正在清理中间帧文件...")
            deleted_count = 0
            failed_count = 0
            
            for frame_file in frame_files:
                try:
                    os.remove(frame_file)
                    deleted_count += 1
                except Exception as e:
                    print(f"删除文件失败 {frame_file}: {e}")
                    failed_count += 1
            
            print(f"清理完成: 成功删除 {deleted_count} 个帧文件")
            if failed_count > 0:
                print(f"警告: {failed_count} 个文件删除失败")
            else:
                print("所有中间帧文件已清理完毕，仅保留最终视频文件")
            
            return True
            
        except Exception as e:
            print(f"合成视频失败: {e}")
            return False

    def process_video(self):
        """处理视频文件的主函数"""
        if not self.landmarker:
            print("人脸标志检测器初始化失败，请检查模型文件")
            return
        
        # 检查输入视频文件
        if not os.path.exists(self.input_video_path):
            print(f"错误：输入视频文件不存在: {self.input_video_path}")
            return
        
        # 打开视频文件
        print(f"正在打开视频文件: {self.input_video_path}")
        cap = cv2.VideoCapture(self.input_video_path)
        
        if not cap.isOpened():
            print(f"错误：无法打开视频文件: {self.input_video_path}")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 更新视频宽高比参数
        self.update_video_aspect_ratio(video_width, video_height)
        
        print(f"视频信息:")
        print(f"  分辨率: {video_width}x{video_height}")
        print(f"  帧率: {video_fps:.2f} FPS")
        print(f"  总帧数: {total_frames}")
        print(f"  预计处理时间: {total_frames / video_fps:.1f} 秒")
        
        # 开始处理
        print("开始离线处理视频...")
        self.start_time = time.time()
        
        try:
            frame_number = 0
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # 跳帧处理（如果设置了frame_skip）
                if frame_number % self.frame_skip != 0:
                    continue
                
                # 转换为RGB格式
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 创建MediaPipe图像对象
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # 计算时间戳（基于帧数和帧率）
                frame_timestamp_ms = int((frame_number / video_fps) * 1000)
                
                # 进行检测
                detection_result = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                # 处理landmarks数据（用于对齐检测）
                self.process_frame_landmarks(detection_result, frame_number)
                
                # 绘制标志点和应用变形
                processed_image = self.draw_landmarks_on_image(rgb_frame, detection_result)
                
                # 保存处理后的帧
                if self.save_frame(processed_image, processed_count):
                    processed_count += 1
                
                # 显示进度
                if self.show_progress and frame_number % self.progress_interval == 0:
                    elapsed_time = time.time() - self.start_time
                    progress = frame_number / total_frames * 100
                    estimated_total_time = elapsed_time / (frame_number / total_frames) if frame_number > 0 else 0
                    remaining_time = estimated_total_time - elapsed_time
                    
                    print(f"处理进度: {frame_number}/{total_frames} ({progress:.1f}%) "
                          f"- 已用时: {elapsed_time:.1f}s, 预计剩余: {remaining_time:.1f}s")
                    
                    if self.warp_ready:
                        status = []
                        if self.enable_pixel_warp:
                            status.append("像素变形")
                        if self.enable_face_only_mode:
                            status.append("面部专用模式")
                        if status:
                            print(f"  当前状态: {', '.join(status)}")
            
            # 处理完成
            elapsed_time = time.time() - self.start_time
            print(f"\n视频处理完成!")
            print(f"总处理时间: {elapsed_time:.1f} 秒")
            print(f"处理帧数: {processed_count}")
            print(f"平均处理速度: {processed_count / elapsed_time:.2f} FPS")
            
            # 合成最终视频
            if processed_count > 0:
                self.create_output_video()
            else:
                print("没有处理任何帧，无法生成输出视频")
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理资源
            cap.release()
            print("视频处理结束")

    def load_camera_calibration(self):
        """加载真实的相机校准参数"""
        if not self.use_real_calibration:
            print("未启用真实相机校准，将在运行时使用手动估计参数")
            return
        
        try:
            # 加载内参矩阵
            if os.path.exists(self.calibration_intrinsic_path):
                print(f"正在加载相机内参: {self.calibration_intrinsic_path}")
                
                # 读取内参文件
                with open(self.calibration_intrinsic_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析内参矩阵 - 支持多种格式
                lines = content.strip().split('\n')
                matrix_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('A=') and '[' in line and ']' in line:
                        # 清理方括号并提取数字
                        line = line.replace('[', '').replace(']', '')
                        matrix_lines.append(line)
                
                if len(matrix_lines) >= 3:
                    # 解析3x3内参矩阵
                    intrinsic_matrix = []
                    for line in matrix_lines[:3]:
                        # 分割数字（处理可能的科学计数法）
                        values = []
                        parts = line.split()
                        for part in parts:
                            try:
                                values.append(float(part))
                            except ValueError:
                                continue
                        if len(values) >= 3:
                            intrinsic_matrix.append(values[:3])
                    
                    if len(intrinsic_matrix) == 3:
                        # 提取相机参数
                        A = np.array(intrinsic_matrix)
                        self.camera_fx = A[0, 0]  # fx
                        self.camera_fy = A[1, 1]  # fy
                        self.camera_cx = A[0, 2]  # cx (主点x坐标)
                        self.camera_cy = A[1, 2]  # cy (主点y坐标)
                        self.camera_skew = A[0, 1]  # skew (倾斜参数)
                        
                        print("成功加载相机内参:")
                        print(f"  fx (x方向焦距): {self.camera_fx:.2f}")
                        print(f"  fy (y方向焦距): {self.camera_fy:.2f}")
                        print(f"  cx (主点x坐标): {self.camera_cx:.2f}")
                        print(f"  cy (主点y坐标): {self.camera_cy:.2f}")
                        print(f"  skew (倾斜参数): {self.camera_skew:.4f}")
                    else:
                        raise ValueError("无法解析内参矩阵格式")
                else:
                    raise ValueError("内参文件格式不正确")
                    
            else:
                print(f"内参文件不存在: {self.calibration_intrinsic_path}")
                self.use_real_calibration = False
                return
            
            # 加载外参矩阵（可选，用于更复杂的3D投影）
            if os.path.exists(self.calibration_extrinsic_path):
                print(f"检测到外参文件: {self.calibration_extrinsic_path}")
                # 注意：当前代码主要使用内参进行透视投影，外参暂不使用
                # 如果需要更精确的3D几何计算，可以在这里加载外参矩阵
            
            print("相机校准参数加载完成")
            
        except Exception as e:
            print(f"加载相机校准参数失败: {e}")
            print("将回退到手动估计相机参数")
            self.use_real_calibration = False
            # 重置相机参数
            self.camera_fx = None
            self.camera_fy = None
            self.camera_cx = None
            self.camera_cy = None
            self.camera_skew = 0.0


def main():
    """主函数"""
    print("MediaPipe 人脸标志检测器 - 离线视频处理版")
    print("=" * 60)
    
    # 创建并运行检测器
    processor = FaceLandmarkerVideoOffline()
    processor.process_video()


if __name__ == "__main__":
    main() 