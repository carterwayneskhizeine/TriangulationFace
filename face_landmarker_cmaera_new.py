#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe 人脸标志检测器 - 摄像头实时处理版
使用摄像头进行实时人脸标志检测和显示 python = 3.9
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

# 按照官方文档推荐的导入方式
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class FaceLandmarkerCamera:
    def __init__(self, camera_id=0):
        """初始化人脸标志检测器"""
        self.result = None
        self.output_image = None
        self.timestamp = 0
        self.camera_id = camera_id  # 摄像头ID，默认为0
        
        # 创建输出文件夹
        self.output_dir = "output_path"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出文件夹: {self.output_dir}")
        
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
        self.transform_file = os.path.join("npy", "face_transform.npy")  # 保存变换参数的文件
        
        # 加载中文字体用于GUI显示
        self.chinese_font = self.load_chinese_font()
        
        # 尝试加载已保存的变换参数
        self.load_saved_transform()
        
        # 摄像头分辨率参数（将在run方法中根据实际摄像头分辨率更新）
        self.camera_width = 1280   # 默认摄像头分辨率
        self.camera_height = 720  # 默认摄像头分辨率
        self.aspect_ratio = self.camera_width / self.camera_height
        self.x_scale_factor = self.aspect_ratio / 1.0  # 对于16:9，约为1.777
        
        # 变形显示控制
        self.show_warped = True  # True=显示变形后，False=显示原始
        
        # 初始化人脸变形器
        self.face_warper = FaceWarper()
        
        # 像素变形控制
        self.enable_pixel_warp = False  # 控制是否启用像素级变形
        self.previous_landmarks = None  # 用于平滑处理
        # 透视投影控制
        self.enable_perspective_projection = True  # 是否使用透视投影
        self.enable_landmarks_perspective = True   # 是否在landmarks渲染时应用透视效果
        
        # 【新增】透视效果调节参数 (根据@修改透视，intensity的角色改变，主要依赖base_Z和delta_Z)
        self.perspective_base_depth = 45.0  # 基础深度（厘米）- 假设的标准拍摄距离
        self.perspective_depth_variation = 55.0  # 深度变化范围（厘米）- 控制透视强度，正值=常规透视，负值=反向透视
        self.perspective_intensity = 1.0 # 在新模型中，此参数不再直接控制透视强度，可用于后续可能的畸变调整
        
        # 【新增】上下透视增强参数 (根据@修改透视，设为0.0以实现纯净针孔模型)
        self.vertical_perspective_strength = 0.0  # 上下透视强度（设为0以关闭额外增强）
        self.vertical_perspective_center = 0.5  # 上下透视中心位置
        
        # 【新增】左右透视增强参数 (根据@修改透视，设为0.0以实现纯净针孔模型)
        self.horizontal_perspective_strength = 0.0  # 左右透视强度（设为0以关闭额外增强）
        self.horizontal_perspective_center = 0.5  # 左右透视中心位置
        
        # 【新增】面部整体平移控制
        self.face_offset_x = 0.0  # 面部X方向偏移（归一化坐标，-1.0到1.0）
        self.face_offset_y = 0.0  # 面部Y方向偏移（归一化坐标，-1.0到1.0）
        
        # 【新增】透视中心偏移控制
        self.perspective_center_offset_x = 0.0  # 透视中心X偏移（像素）
        self.perspective_center_offset_y = 0.0  # 透视中心Y偏移（像素）
        
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
        
        # 线框显示控制
        self.enable_wireframe = True   # 是否启用黑色线框显示
        self.wireframe_thickness = 0.2  # 线框粗细（支持亚像素级，如0.2）
        
        # Lambert材质控制
        self.enable_lambert_material = True  # 是否使用Lambert材质渲染
        
        # 面部专用模式控制
        self.enable_face_only_mode = False  # 是否只显示面部，背景纯黑
        
        # 视频保存控制
        self.save_output_video = False
        self.output_video_writer = None
        
        # FPS统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
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
        
        print("人脸标志检测器初始化完成")
        print("按 'Q' 键退出程序")
        print("按 'S' 键保存当前帧")
        print("按 'L' 键保存当前landmarks到CSV文件")
        print("按 'M' 键重新检测前60帧并计算变换")
        print("按 'X' 键切换原始/变形landmarks显示")
        print("按 'P' 键切换像素级人脸变形显示")
        print("按 'H' 键隐藏/显示landmarks线框")
        print("按 '[' 键缩小landmarks，']' 键放大landmarks")
        print("按 '3' 键减小宽度比例，'4' 键增大宽度比例")
        print("按 'B' 键切换边缘滤波效果")
        print("按 'W' 键切换黑色线框显示")
        print("按 '-' 键减细线框，'+' 键增粗线框")
        print("按 'G' 键切换纹理模式 (原始纹理/Lambert材质)")
        print("按 'O' 键切换面部专用模式 (只显示面部，背景纯黑)")
        print("按 'R' 键开始/停止录制输出视频")
        print("按 'T' 键切换透视投影/弱透视投影")
        print("按 '1/2' 键调整透视强度 (模拟不同焦距)")
        print("按 '5/6' 键调整基础深度 (距离远近)")
        print("按 '7/8' 键调整深度变化范围 (立体感强弱)")
        print("按 '9/0' 键调整上下透视强度")
        print("按 '/*' 键调整左右透视强度")
        print("按 'A/D' 键左右移动面部，'Z/C' 键上下移动面部")
        print("按 'J/L' 键调节透视中心X偏移，'I/K' 键调节透视中心Y偏移")
        print("按 'ESC' 键或 'Q' 键退出程序")
        print(f"相机校准: {'使用真实校准' if self.use_real_calibration else '使用手动估计'}")

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
            
            # 如果启用 Face Geometry 模块，获取面部转换矩阵（4x4）
            if self.enable_face_geometry and hasattr(result, 'facial_transformation_matrixes'):
                try:
                    self.geometry_matrices = result.facial_transformation_matrixes
                    print("已获取面部转换矩阵")
                except Exception as e:
                    print(f"获取面部转换矩阵失败: {e}")
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
            num_faces=1,  # 最多检测2张人脸
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
                        
                        # 【重要修改】新的差异应用方式：当前landmarks + 差异 = 目标形状
                        # 由于新的diff_transformed = 目标模型 - 对齐后活人脸
                        # 所以：当前活人脸 + diff_transformed = 目标模型形状
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
                        
                        # 在landmarks渲染阶段应用透视变换
                        if self.enable_perspective_projection and self.enable_landmarks_perspective:
                            # 使用透视变换函数处理landmarks点
                            display_coords = self.apply_perspective_warp_to_landmarks(None, warped_coords.copy())
                        else:
                            display_coords = warped_coords
                        
                        coords = display_coords
                    else:
                        # 原始landmarks模式，也应用宽度比例调整
                        if self.width_scale != 1.0:
                            # 计算人脸中心的X坐标（使用所有landmarks的X坐标平均值）
                            face_center_x = np.mean(coords[:, 0])
                            # 以人脸中心X坐标为基准，只调整X坐标
                            coords[:, 0] = face_center_x + (coords[:, 0] - face_center_x) * self.width_scale
                        
                        # 即使在原始模式下，也可以应用透视效果
                        if self.enable_perspective_projection and self.enable_landmarks_perspective and not self.show_warped:
                            # 注意：这里我们仅对显示坐标应用透视，不影响原始坐标
                            coords = self.apply_perspective_warp_to_landmarks(None, coords.copy())
                    
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
            
            # Face Geometry 示范：在第一个landmark（如鼻尖）位置显示手动透视3D坐标
            if self.enable_face_geometry and self.geometry_matrices and detection_result.face_landmarks:
                h, w, _ = rgb_image.shape
                # 获取第一个检测到的人脸第2个landmark（鼻尖）
                lm_norm = np.array([[lm.x, lm.y, lm.z] for lm in detection_result.face_landmarks[0][:468]], dtype=np.float32)
                x_p = int(lm_norm[1,0] * w)
                y_p = int(lm_norm[1,1] * h)
                z_norm = lm_norm[1,2]
                # 将z_norm映射为深度值（示例映射，可按需调整）
                depth = (0.5 - z_norm) * 2000.0
                X = (x_p - self.camera_cx) * depth / self.camera_fx
                Y = (y_p - self.camera_cy) * depth / self.camera_fy
                Z = depth
                # 在图像上显示3D坐标
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                cv2.putText(annotated_image, f"3D:({X:.2f},{Y:.2f},{Z:.2f})", (x_p, max(20, y_p-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            return annotated_image
        except Exception as e:
            print(f"绘制landmarks错误: {e}")
            import traceback
            traceback.print_exc()
            return rgb_image  # 返回原始图像
    
    def apply_perspective_warp_to_landmarks(self, src_landmarks, dst_landmarks):
        """
        对landmarks应用真正的透视投影变形。
        此函数将MediaPipe的归一化正交视图landmarks，通过以下步骤转换为透视视图landmarks：
        1. 还原：将归一化坐标和相对深度(z)还原为相机坐标系下的3D点云。
        2. 投影：使用针孔相机模型将3D点云重新投影回2D屏幕。
        3. 格式化：将投影后的2D点重新转换为MediaPipe的归一化坐标格式。
        """
        try:
            if not self.enable_perspective_projection:
                return dst_landmarks.copy()

            # 如果深度变化为0，相当于正交投影，直接返回
            if abs(self.perspective_depth_variation) < 1e-6:
                return dst_landmarks.copy()

            # 获取相机内参，如果未校准则使用估计值
            fx = self.camera_fx if self.camera_fx is not None else self.camera_width
            fy = self.camera_fy if self.camera_fy is not None else self.camera_width
            # 获取相机主点，并应用手动偏移作为透视中心
            cx = (self.camera_cx if self.camera_cx is not None else self.camera_width / 2) + self.perspective_center_offset_x
            cy = (self.camera_cy if self.camera_cy is not None else self.camera_height / 2) + self.perspective_center_offset_y

            projected_dst = dst_landmarks.copy()
            z_values = dst_landmarks[:, 2]
            z_mean = np.mean(z_values)

            # --- 步骤1: 还原 - 将正交视图点转换为3D空间点 ---
            # MediaPipe的输出可以看作是"压平"在某个参考平面上的正交投影。
            # 我们首先假设所有点都在一个z=standard_depth的平面上，并计算出它们对应的3D坐标。
            standard_depth = self.perspective_base_depth
            
            # 将归一化坐标转换为像素坐标
            x_pixels = dst_landmarks[:, 0] * self.camera_width
            y_pixels = dst_landmarks[:, 1] * self.camera_height
            
            # 使用针孔相机模型的逆运算，计算出在standard_depth平面上的3D坐标
            # X = (x_pixel - cx) * Z / fx
            cam_X_on_plane = (x_pixels - cx) * standard_depth / fx
            cam_Y_on_plane = (y_pixels - cy) * standard_depth / fy

            # --- 步骤2: 投影 - 应用真实深度并重新投影 ---
            # 现在，我们根据每个点的MediaPipe z值，计算其真实的、变化的深度。
            z_offsets = z_values - z_mean
            depth_changes = z_offsets * self.perspective_depth_variation
            actual_depths = standard_depth + depth_changes
            
            # 确保深度为正值，避免投影到相机后方
            actual_depths[actual_depths < 0.1] = 0.1
            
            # 关键步骤：用新的深度(actual_depths)重新投影这些3D点。
            # 这个过程等效于将点从参考平面移动到它们的真实深度位置，并观察其在屏幕上的新位置。
            # new_x_pixel = fx * X_on_plane / actual_depth + cx
            new_x_pixels = fx * cam_X_on_plane / actual_depths + cx
            new_y_pixels = fy * cam_Y_on_plane / actual_depths + cy
            
            # --- 步骤3: 格式化 - 转换回MediaPipe的归一化格式 ---
            projected_dst[:, 0] = new_x_pixels / self.camera_width
            projected_dst[:, 1] = new_y_pixels / self.camera_height
            # 保持原始z值不变，因为后续管线或调试可能需要
            projected_dst[:, 2] = dst_landmarks[:, 2] 

            return projected_dst
            
        except Exception as e:
            print(f"透视变形失败: {e}")
            import traceback
            traceback.print_exc()
            return dst_landmarks.copy()
    
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
            
            # 应用透视投影变形
            if self.enable_perspective_projection:
                warped_coords = self.apply_perspective_warp_to_landmarks(None, warped_coords)
                print(f"透视投影已应用，基础深度: {self.perspective_base_depth:.0f}cm, 深度变化: {self.perspective_depth_variation:.0f}cm")
            
            # 【新增】应用面部整体平移
            if self.face_offset_x != 0.0 or self.face_offset_y != 0.0:
                warped_coords[:, 0] += self.face_offset_x  # X方向平移
                warped_coords[:, 1] += self.face_offset_y  # Y方向平移
            
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
            import traceback
            traceback.print_exc()
            return rgb_image
    

    def draw_face_contours(self, image, landmarks, width, height):
        """绘制面部轮廓 - 已废弃，现在使用FACEMESH_TESSELATION绘制完整面部网格"""
        # 此方法已被新的draw_landmarks_on_image方法中的FACEMESH_TESSELATION绘制替代
        pass
    
    def save_landmarks_to_csv(self, landmarks_list, filename=None):
        """保存landmarks到CSV文件（用于按L键保存当前帧）"""
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

    def draw_info_on_image(self, image, detection_result):
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
        
        # 显示FPS信息
        fps_text = f'FPS: {self.current_fps:.1f}'
        image = self.put_chinese_text(image, fps_text, 
                                    (width - 150, 30), font_size=20, color=(255, 255, 255))
        
        # 显示摄像头分辨率信息
        resolution_text = f'分辨率: {self.camera_width}x{self.camera_height}'
        image = self.put_chinese_text(image, resolution_text, 
                                    (width - 250, 70), font_size=18, color=(255, 255, 255))
        
        # 显示录制状态
        if self.save_output_video:
            image = self.put_chinese_text(image, '正在录制输出视频', 
                                        (10, height - 160), font_size=18, color=(255, 0, 0))
        
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
        
        # 显示线框状态
        if self.enable_pixel_warp and self.enable_lambert_material:
            if self.enable_wireframe:
                image = self.put_chinese_text(image, f'黑色线框已启用 粗细:{self.wireframe_thickness:.1f} (按W关闭，+/-调节)', 
                                            (10, height - 80), font_size=18, color=(255, 255, 255))
            else:
                image = self.put_chinese_text(image, '黑色线框已关闭 (Lambert材质模式) (按W启用)', 
                                            (10, height - 80), font_size=18, color=(128, 128, 128))
        elif self.enable_pixel_warp and not self.enable_lambert_material:
            image = self.put_chinese_text(image, '线框在原始纹理模式下不显示', 
                                        (10, height - 80), font_size=18, color=(128, 128, 128))
        
        # 显示纹理模式状态
        if self.enable_pixel_warp:
            if self.enable_lambert_material:
                image = self.put_chinese_text(image, '纹理模式: Lambert材质 (按G切换)', 
                                            (10, height - 60), font_size=18, color=(192, 192, 192))
            else:
                image = self.put_chinese_text(image, '纹理模式: 原始纹理复制 (按G切换)', 
                                            (10, height - 60), font_size=18, color=(255, 128, 0))
        
        # 显示面部专用模式状态
        if self.enable_pixel_warp:
            if self.enable_face_only_mode:
                image = self.put_chinese_text(image, '面部专用模式: 只显示面部，背景纯黑 (按O切换)', 
                                            (10, height - 40), font_size=18, color=(255, 255, 0))
            else:
                image = self.put_chinese_text(image, '面部专用模式: 显示完整人脸 (按O切换)', 
                                            (10, height - 40), font_size=18, color=(128, 128, 128))
        
        # 显示透视投影状态
        if self.enable_perspective_projection:
            if abs(self.perspective_depth_variation) < 1e-6:
                effect_type = "正交投影"
            elif self.perspective_depth_variation > 0:
                effect_type = "常规透视"
            else:
                effect_type = "反向透视"
            perspective_text = f'透视投影: 基础深度{self.perspective_base_depth:.0f}cm 深度变化{self.perspective_depth_variation:.0f}cm ({effect_type})'
            image = self.put_chinese_text(image, perspective_text, 
                                        (10, height - 175), font_size=18, color=(255, 128, 255))
            # 显示landmarks渲染透视状态
            landmarks_perspective_text = f'Landmarks渲染透视: {"开启" if self.enable_landmarks_perspective else "关闭"} (按Y切换)'
            image = self.put_chinese_text(image, landmarks_perspective_text, 
                                        (10, height - 235), font_size=18, color=(128, 255, 255))
        else:
            image = self.put_chinese_text(image, '透视投影已关闭 (弱透视模式)', 
                                        (10, height - 175), font_size=18, color=(128, 128, 128))
        
        # 显示面部偏移状态
        if self.face_offset_x != 0.0 or self.face_offset_y != 0.0:
            offset_text = f'面部偏移: X={self.face_offset_x:.2f}, Y={self.face_offset_y:.2f} (A/D/Z/C调节)'
            image = self.put_chinese_text(image, offset_text, 
                                        (10, height - 195), font_size=18, color=(255, 255, 128))
        else:
            image = self.put_chinese_text(image, '面部偏移: 居中 (A/D/Z/C调节)', 
                                        (10, height - 195), font_size=18, color=(128, 128, 128))
        
        # 显示透视中心偏移状态
        if self.perspective_center_offset_x != 0.0 or self.perspective_center_offset_y != 0.0:
            center_text = f'透视中心偏移: X={self.perspective_center_offset_x:.0f}, Y={self.perspective_center_offset_y:.0f}px (J/L/I/K调节)'
            image = self.put_chinese_text(image, center_text, 
                                        (10, height - 215), font_size=18, color=(128, 255, 255))
        else:
            image = self.put_chinese_text(image, '透视中心: 基于人脸中心 (J/L/I/K调节)', 
                                        (10, height - 215), font_size=18, color=(128, 128, 128))
        
        # 显示landmarks缩放状态
        if self.warp_ready and self.show_warped:
            scale_text = f'landmarks缩放: {self.landmarks_scale:.2f}x (按[/]调整)'
            if self.enable_pixel_warp:
                y_pos = height - 240
            else:
                y_pos = height - 200
            image = self.put_chinese_text(image, scale_text, 
                                        (10, y_pos), font_size=18, color=(255, 255, 0))
        
        # 显示宽度比例状态（所有模式下都显示）
        width_text = f'宽度比例: {self.width_scale:.2f}x (按3/4调整)'
        if self.warp_ready and self.show_warped:
            if self.enable_pixel_warp:
                y_pos = height - 260
            else:
                y_pos = height - 220
        else:
            y_pos = height - 140
        image = self.put_chinese_text(image, width_text, 
                                    (10, y_pos), font_size=18, color=(255, 192, 255))
        
        # 显示变形状态
        if self.warp_ready:
            if self.show_warped:
                if self.enable_pixel_warp:
                    texture_mode = "Lambert材质" if self.enable_lambert_material else "原始纹理"
                    image = self.put_chinese_text(image, f'像素级人脸变形已启用 ({texture_mode}) (按P关闭)', 
                                                (10, height - 140), font_size=18, color=(255, 0, 255))
                    image = self.put_chinese_text(image, '脸型变形已启用 (按X切换)', 
                                                (10, height - 100), font_size=18, color=(0, 255, 255))
                else:
                    image = self.put_chinese_text(image, '脸型变形已启用 (按X切换, P启用像素变形)', 
                                                (10, height - 100), font_size=18, color=(0, 255, 255))
            else:
                image = self.put_chinese_text(image, '显示原始landmarks (按X切换)', 
                                            (10, height - 100), font_size=18, color=(255, 128, 0))
        elif self.frame_count > 0 and self.frame_count < self.N:
            image = self.put_chinese_text(image, f'正在收集landmarks: {self.frame_count}/{self.N}', 
                                        (10, height - 100), font_size=18, color=(255, 255, 0))
        else:
            image = self.put_chinese_text(image, '按M键开始变形检测', 
                                        (10, height - 100), font_size=18, color=(128, 128, 128))
        
        # 显示面部转换矩阵第一行（如有）
        if self.geometry_matrices:
            mat = self.geometry_matrices[0]  # 假设仅第一个人脸
            row0 = mat[:4]
            image = self.put_chinese_text(image, f"Mat0: {row0[0]:.2f},{row0[1]:.2f},{row0[2]:.2f},{row0[3]:.2f}",
                                         (10, height-300), font_size=18, color=(255,192,0))
        
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

    def update_camera_aspect_ratio(self, camera_width, camera_height):
        """根据实际摄像头分辨率更新宽高比参数"""
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.aspect_ratio = camera_width / camera_height
        
        # 计算x坐标修正系数
        # MediaPipe landmarks是基于正方形坐标系(1:1)归一化的
        # 需要根据实际摄像头的宽高比进行修正
        self.x_scale_factor = self.aspect_ratio / 1.0  # 对于16:9，约为1.777
        
        # 根据@修改透视建议：
        # 仅在 未启用真实校准 或 真实校准参数加载失败(self.camera_fx is None) 时，
        # 才进行手动相机参数估计。
        if not self.use_real_calibration or self.camera_fx is None:
            # 如果 self.camera_fx 等是 None (例如真实校准文件不存在或解析失败)，则在这里进行默认估计
            if self.camera_fx is None:
                self.camera_fx = min(camera_width, camera_height) * 0.8 # 默认估计焦距
                print(f"提示: 未加载真实相机内参 fx, 使用估计值: {self.camera_fx:.2f}")
            if self.camera_fy is None:
                self.camera_fy = self.camera_fx # 假设 fy = fx
                print(f"提示: 未加载真实相机内参 fy, 使用估计值: {self.camera_fy:.2f}")
            if self.camera_cx is None:
                self.camera_cx = camera_width / 2.0
                print(f"提示: 未加载真实相机内参 cx, 使用估计值: {self.camera_cx:.2f}")
            if self.camera_cy is None:
                self.camera_cy = camera_height / 2.0
                print(f"提示: 未加载真实相机内参 cy, 使用估计值: {self.camera_cy:.2f}")
            
            print(f"摄像头参数更新：使用手动估计或部分估计的相机参数")
        else:
            # 如果 self.use_real_calibration 为 True 且 self.camera_fx 不是 None，
            # 说明真实校准参数已成功加载，直接使用这些参数。
            print(f"摄像头参数更新：使用已成功加载的真实校准参数")
        
        print(f"  实际分辨率: {camera_width}x{camera_height}")
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

    def update_fps(self):
        """更新FPS计算"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # 每30帧更新一次FPS
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.current_fps = 30.0 / elapsed
            self.fps_start_time = current_time

    def run(self):
        """运行摄像头实时人脸检测"""
        if not self.landmarker:
            print("人脸标志检测器初始化失败，请检查模型文件")
            print("请确保模型文件 'face_landmarker.task' 存在且完整")
            return
        
        # 打开摄像头
        print(f"正在打开摄像头 (ID: {self.camera_id})...")
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 (ID: {self.camera_id})")
            print("请检查：")
            print("1. 摄像头是否正确连接")
            print("2. 摄像头是否被其他程序占用")
            print("3. 摄像头驱动是否正常")
            return
        
        # 设置摄像头分辨率（尝试设置为更高分辨率）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 获取实际摄像头信息
        camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 重要：根据实际摄像头分辨率更新宽高比参数
        self.update_camera_aspect_ratio(camera_width, camera_height)
        
        print(f"摄像头信息:")
        print(f"  分辨率: {camera_width}x{camera_height}")
        print(f"  帧率: {fps:.2f} FPS")
        
        frame_count = 0
        
        print("按键控制:")
        print("  'Q' 或 ESC - 退出")
        print("  'S' - 保存当前帧")
        print("  'R' - 开始/停止录制输出视频")
        print("  'M' - 重新检测并计算变换")
        print("  'X' - 切换原始/变形显示")
        print("  'P' - 切换像素级变形")
        print("  'E' - 导出变形后的人脸模型为OBJ文件")
        print("  'H' - 隐藏/显示landmarks线框")
        print("  '[' 键缩小landmarks，']' 键放大landmarks")
        print("  '3' 键减小宽度比例，'4' 键增大宽度比例")
        print("  'B' 键切换边缘滤波效果")
        print("  'W' 键切换黑色线框显示")
        print("  '-' 键减细线框，'+' 键增粗线框")
        print("  'G' 键切换纹理模式 (原始纹理/Lambert材质)")
        print("  'O' 键切换面部专用模式 (只显示面部，背景纯黑)")
        print("  'R' - 开始/停止录制输出视频")
        print("  'T' 键切换透视投影/弱透视投影")
        print("  'Y' 键切换landmarks渲染的透视效果")
        print("  '1/2' 键调整透视强度 (模拟不同焦距)")
        print("  '5/6' 键调整基础深度 (距离远近)")
        print("  '7/8' 键调整深度变化范围 (立体感强弱)")
        print("  '9/0' 键调整上下透视强度")
        print("  '/*' 键调整左右透视强度")
        print("  'A/D' 键左右移动面部，'Z/C' 键上下移动面部")
        print("  'J/L' 键调节透视中心X偏移，'I/K' 键调节透视中心Y偏移")
        print("  'ESC' 键或 'Q' 键退出程序")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("错误：无法从摄像头读取图像")
                    break
                frame = cv2.flip(frame, 1)  # 左右翻转（镜像）
                
                frame_count += 1
                
                # 转换为RGB格式
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 创建MediaPipe图像对象
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # 计算时间戳（基于帧数和帧率）
                frame_timestamp_ms = int((frame_count / 30.0) * 1000)  # 假设30FPS
                
                # 进行检测
                detection_result = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
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
                
                # 更新FPS
                self.update_fps()
                
                # 绘制信息
                display_frame = self.draw_info_on_image(annotated_frame, detection_result)
                
                # 保存到输出视频（如果启用）
                if self.save_output_video and self.output_video_writer:
                    self.output_video_writer.write(display_frame)
                
                # 显示帧
                cv2.imshow('MediaPipe 人脸标志检测 - 摄像头实时', display_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' 或 ESC 键退出
                    break
                elif key == ord('s'):  # 's' 键保存图像
                    save_name = f'camera_frame_{frame_count}_{int(time.time())}.jpg'
                    save_path = os.path.join(self.output_dir, save_name)
                    cv2.imwrite(save_path, display_frame)
                    print(f"图像已保存: {save_path}")
                elif key == ord('l'):  # 'l' 键保存landmarks
                    if detection_result.face_landmarks:
                        csv_filename = f'camera_landmarks_{frame_count}_{int(time.time())}.csv'
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
                elif key == ord('E') or key == ord('e'):  # 'E' 或 'e' 键导出变形后的landmarks为OBJ文件
                    if self.warp_ready and self.diff_transformed is not None and detection_result and detection_result.face_landmarks:
                        # 获取当前帧的landmarks
                        current_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in detection_result.face_landmarks[0][:468]], dtype=np.float32)
                        
                        # 修正x坐标（与实时变形保持一致）
                        corrected_coords = current_landmarks.copy()
                        corrected_coords[:, 0] *= self.x_scale_factor
                        
                        # 应用形状差异变换
                        warped_coords = corrected_coords + self.diff_transformed
                        
                        # 还原x坐标到16:9坐标系
                        warped_coords[:, 0] /= self.x_scale_factor
                        
                        # 导出为OBJ文件
                        exported_file = self.export_warped_landmarks_to_obj(warped_coords)
                        if exported_file:
                            print(f"✅ 变形后的人脸模型已导出: {exported_file}")
                        else:
                            print("❌ 导出变形后的人脸模型失败")
                    else:
                        if not self.warp_ready:
                            print("⚠️ 还未进行M键对齐，无法导出变形模型")
                        elif detection_result is None or not detection_result.face_landmarks:
                            print("⚠️ 当前帧未检测到人脸，无法导出")
                        else:
                            print("⚠️ 变形功能未就绪，无法导出")
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
                elif key == ord('W') or key == ord('w'):  # 'W' 或 'w' 键切换黑色线框显示
                    if self.enable_pixel_warp and self.enable_lambert_material:
                        self.enable_wireframe = not self.enable_wireframe
                        if self.enable_wireframe:
                            print("黑色线框显示已开启 (Lambert材质模式)")
                        else:
                            print("黑色线框显示已关闭 (Lambert材质模式)")
                    elif self.enable_pixel_warp and not self.enable_lambert_material:
                        print("线框功能仅在Lambert材质模式下可用")
                    else:
                        print("请先启用像素级人脸变形功能")
                elif key == ord('G') or key == ord('g'):  # 'G' 或 'g' 键切换纹理模式
                    self.enable_lambert_material = not self.enable_lambert_material
                    print(f"纹理模式已切换为: {'Lambert材质' if self.enable_lambert_material else '原始纹理'}")
                elif key == ord('O') or key == ord('o'):  # 'O' 或 'o' 键切换面部专用模式
                    self.enable_face_only_mode = not self.enable_face_only_mode
                    print(f"面部专用模式已切换为: {'只显示面部，背景纯黑' if self.enable_face_only_mode else '显示完整人脸'}")
                elif key == ord('R') or key == ord('r'):  # 'R' 或 'r' 键切换录制
                    if not self.save_output_video:
                        # 开始录制
                        if self.init_video_writer(camera_width, camera_height, 30.0):  # 假设30FPS
                            self.save_output_video = True
                        else:
                            print("无法开始录制")
                    else:
                        # 停止录制
                        self.save_output_video = False
                        self.cleanup_video_writer()
                elif key == ord('T') or key == ord('t'):  # 'T' 或 't' 键切换透视投影/弱透视投影
                    if self.enable_perspective_projection:
                        self.enable_perspective_projection = False
                        print("切换到弱透视投影")
                    else:
                        self.enable_perspective_projection = True
                        print("切换到透视投影")
                elif key == ord('Y') or key == ord('y'):  # 'Y' 或 'y' 键切换landmarks渲染的透视效果
                    self.enable_landmarks_perspective = not self.enable_landmarks_perspective
                    print(f"Landmarks渲染透视效果: {'开启' if self.enable_landmarks_perspective else '关闭'}")
                elif key == ord('1'):  # '1' 键减小透视强度（模拟长焦镜头）
                    self.perspective_intensity = max(0.1, self.perspective_intensity - 0.1)
                    focal_length_equiv = int(35 / self.perspective_intensity) if self.perspective_intensity > 0 else 350
                    print(f"透视强度: {self.perspective_intensity:.1f} (约等效{focal_length_equiv}mm焦距)")
                elif key == ord('2'):  # '2' 键增大透视强度（模拟广角镜头）
                    self.perspective_intensity = min(2.0, self.perspective_intensity + 0.1)
                    focal_length_equiv = int(35 / self.perspective_intensity)
                    print(f"透视强度: {self.perspective_intensity:.1f} (约等效{focal_length_equiv}mm焦距)")
                elif key == ord('5'):  # '5' 键减小基础深度（更近距离拍摄效果）
                    self.perspective_base_depth = max(10.0, self.perspective_base_depth - 5.0)
                    print(f"基础深度: {self.perspective_base_depth:.1f}cm (距离越近透视越强)")
                elif key == ord('6'):  # '6' 键增大基础深度（更远距离拍摄效果）
                    self.perspective_base_depth = min(100.0, self.perspective_base_depth + 5.0)
                    print(f"基础深度: {self.perspective_base_depth:.1f}cm (距离越远透视越弱)")
                elif key == ord('7'):  # '7' 键减小深度变化范围
                    self.perspective_depth_variation = max(-100.0, self.perspective_depth_variation - 5.0)
                    effect_type = "反向透视" if self.perspective_depth_variation < 0 else "常规透视" if self.perspective_depth_variation > 0 else "正交投影"
                    print(f"深度变化: {self.perspective_depth_variation:.1f}cm ({effect_type})")
                elif key == ord('8'):  # '8' 键增大深度变化范围
                    self.perspective_depth_variation = min(100.0, self.perspective_depth_variation + 5.0)
                    effect_type = "反向透视" if self.perspective_depth_variation < 0 else "常规透视" if self.perspective_depth_variation > 0 else "正交投影"
                    print(f"深度变化: {self.perspective_depth_variation:.1f}cm ({effect_type})")
                elif key == ord('9'):  # '9' 键减小上下透视强度
                    self.vertical_perspective_strength = max(0.0, self.vertical_perspective_strength - 0.1)
                    print(f"上下透视强度: {self.vertical_perspective_strength:.1f}")
                elif key == ord('0'):  # '0' 键增大上下透视强度
                    self.vertical_perspective_strength = min(2.0, self.vertical_perspective_strength + 0.1)
                    print(f"上下透视强度: {self.vertical_perspective_strength:.1f}")
                elif key == ord('/'):  # '/' 键减小左右透视强度
                    self.horizontal_perspective_strength = max(0.0, self.horizontal_perspective_strength - 0.1)
                    print(f"左右透视强度: {self.horizontal_perspective_strength:.1f}")
                elif key == ord('*'):  # '*' 键增大左右透视强度
                    self.horizontal_perspective_strength = min(2.0, self.horizontal_perspective_strength + 0.1)
                    print(f"左右透视强度: {self.horizontal_perspective_strength:.1f}")
                elif key == ord('A') or key == ord('a'):  # 'A' 或 'a' 键左右移动面部
                    self.face_offset_x = max(-1.0, self.face_offset_x - 0.05)
                    print(f"面部X方向偏移调整为: {self.face_offset_x:.2f}")
                elif key == ord('D') or key == ord('d'):  # 'D' 或 'd' 键左右移动面部
                    self.face_offset_x = min(1.0, self.face_offset_x + 0.05)
                    print(f"面部X方向偏移调整为: {self.face_offset_x:.2f}")
                elif key == ord('Z') or key == ord('z'):  # 'Z' 或 'z' 键上下移动面部
                    self.face_offset_y = max(-1.0, self.face_offset_y - 0.05)
                    print(f"面部Y方向偏移调整为: {self.face_offset_y:.2f}")
                elif key == ord('C') or key == ord('c'):  # 'C' 或 'c' 键上下移动面部
                    self.face_offset_y = min(1.0, self.face_offset_y + 0.05)
                    print(f"面部Y方向偏移调整为: {self.face_offset_y:.2f}")
                elif key == ord('J') or key == ord('j'):  # 'J' 或 'j' 键调节透视中心X偏移
                    self.perspective_center_offset_x = max(-100, self.perspective_center_offset_x - 5)
                    print(f"透视中心X偏移调整为: {self.perspective_center_offset_x}")
                elif key == ord('L') or key == ord('l'):  # 'L' 或 'l' 键调节透视中心X偏移
                    self.perspective_center_offset_x = min(100, self.perspective_center_offset_x + 5)
                    print(f"透视中心X偏移调整为: {self.perspective_center_offset_x}")
                elif key == ord('I') or key == ord('i'):  # 'I' 或 'i' 键调节透视中心Y偏移
                    self.perspective_center_offset_y = max(-100, self.perspective_center_offset_y - 5)
                    print(f"透视中心Y偏移调整为: {self.perspective_center_offset_y}")
                elif key == ord('K') or key == ord('k'):  # 'K' 或 'k' 键调节透视中心Y偏移
                    self.perspective_center_offset_y = min(100, self.perspective_center_offset_y + 5)
                    print(f"透视中心Y偏移调整为: {self.perspective_center_offset_y}")
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            if self.save_output_video:
                self.cleanup_video_writer()
            print("摄像头检测完成")

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
        使用自动化精确对齐流程：
        1. 保存平均landmarks到CSV
        2. 自动转换为OBJ文件  
        3. 自动精确对齐到canonical标准
        4. 计算与Andy_Wah_facemesh.obj的差异
        """
        print("开始自动化精确对齐流程...")
        print(f"视频分辨率: {self.camera_width}x{self.camera_height}")
        print(f"视频宽高比: {self.aspect_ratio:.3f}")
        print(f"X坐标修正系数: {self.x_scale_factor:.3f}")
        
        # 步骤1：保存平均landmarks到CSV文件
        print("\n=== 步骤1：保存平均landmarks ===")
        csv_timestamp = int(time.time())
        csv_filename = f'averaged_landmarks_{csv_timestamp}.csv'
        csv_filepath = self.save_averaged_landmarks_to_csv(avg_landmarks, csv_filename)
        if not csv_filepath:
            print("❌ 保存CSV文件失败")
            return
        print(f"✅ CSV文件已保存: {csv_filepath}")
        
        # 步骤2：自动转换CSV为OBJ文件
        print("\n=== 步骤2：转换CSV为OBJ ===")
        obj_filepath = self.auto_convert_csv_to_obj(csv_filepath)
        if not obj_filepath:
            print("❌ CSV转OBJ失败")
            return
        print(f"✅ OBJ文件已生成: {obj_filepath}")
        
        # 步骤3：自动精确对齐OBJ文件
        print("\n=== 步骤3：精确对齐到canonical标准 ===")
        aligned_obj_filepath = self.auto_precise_alignment(obj_filepath)
        if not aligned_obj_filepath:
            print("❌ 精确对齐失败")
            return
        print(f"✅ 对齐后OBJ文件: {aligned_obj_filepath}")
        
        # 步骤4：计算与目标模型的差异
        print("\n=== 步骤4：计算形状差异 ===")
        if self.custom_vertices is None:
            print("❌ 目标模型(Andy_Wah_facemesh.obj)未加载")
            return
            
        success = self.compute_shape_difference(aligned_obj_filepath)
        if not success:
            print("❌ 计算形状差异失败")
            return
            
        print("✅ 自动化精确对齐流程完成！")
        print("实时变形功能已启用，可以按X键切换显示效果")
    
    def save_averaged_landmarks_to_csv(self, avg_landmarks, filename=None):
        """保存平均landmarks到CSV文件，返回文件路径"""
        # 确保csv_files文件夹存在
        csv_dir = "csv_files"
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
            print(f"创建CSV文件夹: {csv_dir}")
        
        # 生成文件路径
        if filename is None:
            timestamp = int(time.time())
            filename = f'averaged_landmarks_{timestamp}.csv'
        filepath = os.path.join(csv_dir, filename)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['point_id', 'x', 'y', 'z']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 写入表头
                writer.writeheader()
                
                # 写入平均landmarks数据
                for point_id, landmark in enumerate(avg_landmarks):
                    writer.writerow({
                        'point_id': point_id,
                        'x': landmark[0],
                        'y': landmark[1], 
                        'z': landmark[2]
                    })
                
            print(f"平均Landmarks已保存: {len(avg_landmarks)} 个点")
            return filepath
            
        except Exception as e:
            print(f"保存平均landmarks失败: {e}")
            return None
    
    def auto_convert_csv_to_obj(self, csv_filepath):
        """自动将CSV文件转换为OBJ文件"""
        try:
            # 导入转换器模块
            import importlib.util
            
            # 动态加载csv_to_obj_converter模块
            spec = importlib.util.spec_from_file_location("csv_to_obj_converter", "csv_to_obj_converter.py")
            converter_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(converter_module)
            
            # 创建转换器实例
            converter = converter_module.LandmarksToObjConverter()
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(csv_filepath))[0]
            output_obj_name = f"{base_name}_face_model.obj"
            
            # 执行转换
            success = converter.convert(csv_filepath, output_obj_name)
            
            if success:
                # 转换器会自动把文件放在result_file文件夹中
                result_filepath = os.path.join("result_file", output_obj_name)
                if os.path.exists(result_filepath):
                    return result_filepath
                else:
                    print(f"转换后的文件未找到: {result_filepath}")
                    return None
            else:
                return None
                
        except Exception as e:
            print(f"自动CSV转OBJ失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def auto_precise_alignment(self, obj_filepath):
        """自动对OBJ文件进行精确对齐"""
        try:
            # 动态加载precise_alignment_tool模块
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("precise_alignment_tool", "precise_alignment_tool.py")
            alignment_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(alignment_module)
            
            # 创建对齐工具实例
            alignment_tool = alignment_module.PreciseFaceAlignmentTool()
            
            # 执行对齐
            aligned_filepath = alignment_tool.process_model(obj_filepath)
            
            if aligned_filepath and os.path.exists(aligned_filepath):
                return aligned_filepath
            else:
                print(f"对齐后的文件未找到: {aligned_filepath}")
                return None
                
        except Exception as e:
            print(f"自动精确对齐失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_shape_difference(self, aligned_obj_filepath):
        """计算对齐后的活人脸模型与目标模型的形状差异"""
        try:
            # 加载对齐后的活人脸模型
            aligned_vertices = self.load_obj_vertices(aligned_obj_filepath)
            print(f"对齐后活人脸模型: {len(aligned_vertices)} 个顶点")
            
            # 确保顶点数量匹配
            if len(aligned_vertices) != len(self.custom_vertices):
                print(f"⚠️ 顶点数量不匹配: 活人脸{len(aligned_vertices)} vs 目标模型{len(self.custom_vertices)}")
                # 取较小的数量
                min_count = min(len(aligned_vertices), len(self.custom_vertices))
                aligned_vertices = aligned_vertices[:min_count]
                custom_vertices_trimmed = self.custom_vertices[:min_count]
            else:
                custom_vertices_trimmed = self.custom_vertices
            
            print("=== 坐标系转换 ===")
            print("OBJ文件使用3D世界坐标系，需要转换为MediaPipe归一化坐标系")
            
            # 【关键修复】将3D OBJ坐标转换为MediaPipe归一化坐标
            # 参考csv_to_obj_converter.py的逆向转换
            
            # 转换参数（与csv_to_obj_converter.py中的参数对应）
            aspect_ratio_correction = 16.0 / 9.0  # ≈ 1.777
            overall_scale = 55.0
            
            def obj_to_normalized(obj_coords):
                """将OBJ 3D坐标转换为MediaPipe归一化坐标"""
                normalized = np.zeros_like(obj_coords)
                
                # 逆向转换（参考csv_to_obj_converter.py）
                # OBJ: x_3d = (x - 0.5) * aspect_ratio_correction * overall_scale
                # 逆向: x = x_3d / (aspect_ratio_correction * overall_scale) + 0.5
                normalized[:, 0] = obj_coords[:, 0] / (aspect_ratio_correction * overall_scale) + 0.5
                
                # OBJ: y_3d = -(y - 0.5) * overall_scale  
                # 逆向: y = 0.5 - y_3d / overall_scale
                normalized[:, 1] = 0.5 - obj_coords[:, 1] / overall_scale
                
                # OBJ: z_3d = -z * aspect_ratio_correction * overall_scale
                # 逆向: z = -z_3d / (aspect_ratio_correction * overall_scale)
                normalized[:, 2] = -obj_coords[:, 2] / (aspect_ratio_correction * overall_scale)
                
                return normalized
            
            # 转换两个模型到归一化坐标系
            aligned_normalized = obj_to_normalized(aligned_vertices)
            target_normalized = obj_to_normalized(custom_vertices_trimmed)
            
            print(f"转换后坐标范围检查:")
            print(f"  对齐模型 X: [{aligned_normalized[:, 0].min():.6f}, {aligned_normalized[:, 0].max():.6f}]")
            print(f"  对齐模型 Y: [{aligned_normalized[:, 1].min():.6f}, {aligned_normalized[:, 1].max():.6f}]")
            print(f"  对齐模型 Z: [{aligned_normalized[:, 2].min():.6f}, {aligned_normalized[:, 2].max():.6f}]")
            print(f"  目标模型 X: [{target_normalized[:, 0].min():.6f}, {target_normalized[:, 0].max():.6f}]")
            print(f"  目标模型 Y: [{target_normalized[:, 1].min():.6f}, {target_normalized[:, 1].max():.6f}]")
            print(f"  目标模型 Z: [{target_normalized[:, 2].min():.6f}, {target_normalized[:, 2].max():.6f}]")
            
            # 计算形状差异向量：目标模型 - 对齐后的活人脸（都在归一化坐标系中）
            self.diff_transformed = target_normalized - aligned_normalized
            
            self.warp_ready = True
            
            # 保存变换参数到文件
            try:
                np.save(self.transform_file, self.diff_transformed)
                print(f"变换参数已保存到: {self.transform_file}")
            except Exception as e:
                print(f"保存变换参数失败: {e}")
            
            # 分析差异统计
            diff_stats = {
                'x_range': (self.diff_transformed[:, 0].min(), self.diff_transformed[:, 0].max()),
                'y_range': (self.diff_transformed[:, 1].min(), self.diff_transformed[:, 1].max()),
                'z_range': (self.diff_transformed[:, 2].min(), self.diff_transformed[:, 2].max()),
                'mean_distance': np.mean(np.linalg.norm(self.diff_transformed, axis=1))
            }
            
            print("归一化坐标系中的形状差异统计:")
            print(f"  X方向: [{diff_stats['x_range'][0]:.6f}, {diff_stats['x_range'][1]:.6f}]")
            print(f"  Y方向: [{diff_stats['y_range'][0]:.6f}, {diff_stats['y_range'][1]:.6f}]") 
            print(f"  Z方向: [{diff_stats['z_range'][0]:.6f}, {diff_stats['z_range'][1]:.6f}]")
            print(f"  平均差异距离: {diff_stats['mean_distance']:.6f}")
            
            if diff_stats['mean_distance'] < 0.1:
                print("✅ 差异在合理范围内，适合实时变形")
            elif diff_stats['mean_distance'] < 0.2:
                print("⚠️ 差异较大，变形效果可能明显")
            else:
                print("❌ 差异过大，可能需要调整对齐参数")
            
            return True
            
        except Exception as e:
            print(f"计算形状差异失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def export_warped_landmarks_to_obj(self, warped_landmarks, filename=None):
        """将变形后的landmarks导出为OBJ文件"""
        try:
            if filename is None:
                timestamp = int(time.time())
                filename = f'warped_face_model_{timestamp}.obj'
            
            # 确保保存在result_file文件夹中（而不是output_dir）
            result_dir = "result_file"
            os.makedirs(result_dir, exist_ok=True)
            filepath = os.path.join(result_dir, filename)
            
            # 坐标系转换：MediaPipe归一化坐标 -> OBJ 3D坐标
            # 使用与csv_to_obj_converter.py相同的转换参数
            aspect_ratio_correction = 16.0 / 9.0  # ≈ 1.777
            overall_scale = 55.0
            
            print(f"\n=== 导出变形后的人脸模型 ===")
            print(f"将变形后的归一化坐标转换为OBJ 3D坐标")
            print(f"输出文件: {filepath}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # 写入OBJ文件头部
                f.write("# 变形后的面部模型\n")
                f.write(f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 基于MediaPipe face landmarks，应用了目标模型形状差异\n")
                f.write(f"# 总顶点数: {len(warped_landmarks)}\n")
                f.write("\n")
                
                # 转换坐标并写入顶点
                for i, (x_norm, y_norm, z_norm) in enumerate(warped_landmarks):
                    # 坐标转换（与csv_to_obj_converter.py保持一致）
                    x_3d = (x_norm - 0.5) * aspect_ratio_correction * overall_scale
                    y_3d = -(y_norm - 0.5) * overall_scale  # Y轴翻转
                    z_3d = -z_norm * aspect_ratio_correction * overall_scale  # Z轴翻转并缩放
                    
                    f.write(f"v {x_3d:.6f} {y_3d:.6f} {z_3d:.6f}\n")
                
                # 添加面部网格连接信息（可选）
                f.write("\n# 面部网格连接\n")
                f.write("# 使用MediaPipe FACEMESH_TESSELATION连接\n")
                
                # 如果需要添加面的定义，可以基于MediaPipe的连接信息
                # 这里暂时只保存顶点，因为主要用于形状分析
            
            print(f"✅ 变形后的人脸模型已导出: {filepath}")
            
            # 统计变形后的坐标范围
            coords_3d = []
            for x_norm, y_norm, z_norm in warped_landmarks:
                x_3d = (x_norm - 0.5) * aspect_ratio_correction * overall_scale
                y_3d = -(y_norm - 0.5) * overall_scale
                z_3d = -z_norm * aspect_ratio_correction * overall_scale
                coords_3d.append([x_3d, y_3d, z_3d])
            
            coords_3d = np.array(coords_3d)
            print(f"变形后模型3D坐标范围:")
            print(f"  X: [{coords_3d[:, 0].min():.3f}, {coords_3d[:, 0].max():.3f}]")
            print(f"  Y: [{coords_3d[:, 1].min():.3f}, {coords_3d[:, 1].max():.3f}]")
            print(f"  Z: [{coords_3d[:, 2].min():.3f}, {coords_3d[:, 2].max():.3f}]")
            
            return filepath
            
        except Exception as e:
            print(f"导出变形后模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None

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
    print("MediaPipe 人脸标志检测器 - 摄像头实时版")
    print("=" * 50)
    
    # 创建并运行检测器
    detector = FaceLandmarkerCamera()
    detector.run()


if __name__ == "__main__":
    main() 