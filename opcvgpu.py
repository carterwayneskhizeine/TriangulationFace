# 在Python中运行这个检查
import cv2
print("OpenCV版本:", cv2.__version__)
print("CUDA设备数量:", cv2.cuda.getCudaEnabledDeviceCount())
print("GPU模块可用:", cv2.cuda.getCudaEnabledDeviceCount() > 0)