
# bản quyền thuộc về FWD
import sys
import os

# Import torch and YOLO first to avoid DLL search path conflicts with PyQt5 on Windows
try:
    import torch
    from ultralytics import YOLO
except Exception as e:
    print(f"Pre-importing torch/YOLO failed: {e}")

import time
import threading
import subprocess
import importlib
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from fairino import Robot
import target_point
from typing import NamedTuple, Tuple, List
from datetime import datetime
try:
    import open3d as o3d # import open3d xử lý pointcloud
except ImportError:
    o3d = None
    print("Warning: open3d not available, some features will be disabled")

#impor PyQt5 để làm giao diện
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGroupBox, QPushButton, QLabel, 
                             QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
                             QCheckBox, QSlider, QTextEdit, QProgressBar,
                             QFrame, QFileDialog, QMessageBox, QSpacerItem,
                             QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QMutex, QRect
from PyQt5.QtGui import QPixmap, QImage, QFont, QMouseEvent, QPainter, QPen

# Import các class và function từ pca_robot.py
class PoseEstimation:
    """Class để lưu trữ kết quả ước lượng tư thế"""
    def __init__(self, center: np.ndarray, axes: np.ndarray, rotation_matrix: np.ndarray):
        self.center = center
        self.axes = axes
        self.rotation_matrix = rotation_matrix

class CameraInfo(NamedTuple):
    """Camera intrinsics"""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    scale: float

#region Các chức năng về View
# Vẽ mask overlay lên ảnh với màu và độ trong suốt cho trước
# Input: image_bgr - ảnh BGR gốc, mask_bool - mask boolean, color - màu overlay, alpha - độ trong suốt
# Output: ảnh đã được vẽ mask overlay
def draw_mask_overlay(image_bgr: np.ndarray, mask_bool: np.ndarray, color=(0, 255, 0), alpha: float = 0.35):
    """Vẽ mask overlay lên ảnh."""
    overlay = image_bgr.copy()
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay


#endregion


#region Các hàm về PCA
# Với eigenvectors là trục chính
# với eigenvalues là mức độ phân tán theo trục đó
# Thực hiện phân tích thành phần chính (PCA) sử dụng numpy
# Input: points - mảng điểm 3D
# Output: eigenvalues (giá trị riêng) và eigenvectors (vector riêng) đã sắp xếp theo độ lớn
def numpy_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Thực hiện PCA sử dụng numpy"""
    centered_points = points - np.mean(points, axis=0)
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

# Ước lượng tư thế của object từ point cloud sử dụng PCA
# Input: point_cloud - đám mây điểm 3D của object
# Output: PoseEstimation object chứa center, axes và rotation_matrix
def estimate_pose_with_pca(point_cloud) -> PoseEstimation:
    """Ước lượng tư thế object bằng PCA (3D đầy đủ)"""
    if len(point_cloud.points) < 3:
        return None
    
    points = np.asarray(point_cloud.points)
    center = np.mean(points, axis=0)
    centered_points = points - center
    eigenvalues, eigenvectors = numpy_pca(centered_points)

    x_axis = eigenvectors[:, 0]
    y_axis = eigenvectors[:, 1]
    z_axis = eigenvectors[:, 2]

    # Chuẩn hóa
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

    # Bảo đảm tay phải
    if np.dot(np.cross(x_axis, y_axis), z_axis) < 0:
        z_axis = -z_axis

    axes = np.stack([x_axis, y_axis, z_axis], axis=1)
    rotation_matrix = axes
    
    return PoseEstimation(center, axes, rotation_matrix)



#endregion


# Tạo point cloud từ depth image sử dụng camera intrinsics
# Input: depth - ảnh depth, camera - thông tin camera intrinsics, organized - có tổ chức hay không
# Output: mảng điểm 3D (organized hoặc flattened)
def create_point_cloud_from_depth_image(depth: np.ndarray, camera: CameraInfo, organized: bool = True, roi_offset=(0, 0)):
    """Tạo point cloud từ depth image"""
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    x = (u + roi_offset[0] - camera.cx) * depth / camera.fx
    y = (v + roi_offset[1] - camera.cy) * depth / camera.fy
    z = depth
    
    y_open3d = -y  # Flip Y axis
    
    if organized:
        points = np.stack([x, y_open3d, z], axis=-1)
        return points
    else:
        points = np.stack([x.flatten(), y_open3d.flatten(), z.flatten()], axis=1)
        return points

# Tạo PointCloud từ detection mask với các tham số lọc khoảng cách
# Input: color, depth, mask - dữ liệu ảnh và mask, camera - thông tin camera, min/max_distance - giới hạn khoảng cách
# Output: PointCloud object hoặc None nếu không có điểm hợp lệ
def build_point_cloud_from_detection(
    color: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    camera: CameraInfo,
    min_distance_m: float = None,
    max_distance_m: float = None,
    roi_offset=(0, 0)
):
    """Tạo PointCloud từ detection mask"""
    if o3d is None:
        return None
        
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True, roi_offset=roi_offset)
    valid = (mask > 0) & (depth > 0)

    if min_distance_m is not None or max_distance_m is not None:
        z_in_meters = cloud[:, :, 2]
        if min_distance_m is not None:
            valid &= (z_in_meters >= float(min_distance_m))
        if max_distance_m is not None:
            valid &= (z_in_meters <= float(max_distance_m))

    cloud_masked = cloud[valid]
    color_masked = color[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    if len(pcd.points) > 0:
        pcd.estimate_normals()
        pcd.normalize_normals()
    return pcd

# Kiểm tra tính hợp lệ của ma trận RT (rotation + translation)
# Input: RT - ma trận 4x4 chứa rotation và translation
# Output: True nếu ma trận hợp lệ (determinant = 1, orthogonal), False nếu không
def validate_rt_matrix(RT: np.ndarray) -> bool:
    """Kiểm tra tính hợp lệ của ma trận RT"""
    if RT.shape != (4, 4):
        return False
    
    R = RT[:3, :3]
    T = RT[:3, 3]
    
    # Kiểm tra rotation matrix
    det_R = np.linalg.det(R)
    if abs(det_R - 1.0) > 1e-6:
        return False
    
    # Kiểm tra orthogonality
    should_be_identity = R @ R.T
    identity = np.eye(3)
    if not np.allclose(should_be_identity, identity, atol=1e-6):
        return False
    
    return True

# Tính toán offset để gắp object dựa trên loại gắp và tư thế object
# Input: pose - tư thế object, grasp_type - kiểu gắp (center, top, side, pen_grasp)
# Output: vector offset 3D để điều chỉnh vị trí gắp
def calculate_grasp_offset(pose: PoseEstimation, grasp_type: str = "center") -> np.ndarray:
    """Tính toán offset để gắp object dựa trên loại object và tư thế"""
    if grasp_type == "center":
        return np.array([0.0, 0.0, 0.0])
    elif grasp_type == "top":
        return np.array([0.0, 0.0, 0.05])
    elif grasp_type == "side":
        return np.array([0.05, 0.0, 0.0])
    elif grasp_type == "pen_grasp":
        return np.array([0.05, 0.0, 0.0])
    else:
        return np.array([0.0, 0.0, 0.0])

# Giới hạn các góc quay để đảm bảo an toàn cho robot
# Input: R_euler - góc quay Euler, max_rx/ry/rz - giới hạn góc tối đa
# Output: góc quay đã được giới hạn và danh sách cảnh báo
def clamp_rotation_angles(R_euler: np.ndarray, max_rx: float = 15.0, max_ry: float = 180.0, max_rz: float = 180.0) -> Tuple[np.ndarray, List[str]]:
    """Giới hạn các góc quay để đảm bảo an toàn cho robot"""
    warnings = []
    clamped = R_euler.copy()
    
    # Giới hạn Rx
    if abs(R_euler[0]) > max_rx:
        clamped[0] = np.clip(R_euler[0], -max_rx, max_rx)
        warnings.append(f"Rx {R_euler[0]:.1f}° clamped to {clamped[0]:.1f}°")
    
    # Giới hạn Ry
    if abs(R_euler[1]) > max_ry:
        clamped[1] = np.clip(R_euler[1], -max_ry, max_ry)
        warnings.append(f"Ry {R_euler[1]:.1f}° clamped to {clamped[1]:.1f}°")
    
    # Giới hạn Rz
    if abs(R_euler[2]) > max_rz:
        clamped[2] = np.clip(R_euler[2], -max_rz, max_rz)
        warnings.append(f"Rz {R_euler[2]:.1f}° clamped to {clamped[2]:.1f}°")
    
    return clamped, warnings

# Giới hạn góc quay trong desc_pos cho lệnh MoveL của robot
# Input: desc_pos - vị trí robot [X,Y,Z,RX,RY,RZ], max_rx/ry/rz - giới hạn góc tối đa
# Output: desc_pos đã được giới hạn và danh sách cảnh báo
def clamp_desc_pos_angles(desc_pos: List[float], max_rx: float = 15.0, max_ry: float = 180.0, max_rz: float = 180.0) -> Tuple[List[float], List[str]]:
    """Giới hạn góc quay trong desc_pos cho lệnh MoveL"""
    warnings = []
    clamped = desc_pos.copy()
    
    # Giới hạn Rx (index 3)
    if abs(desc_pos[3]) > max_rx:
        clamped[3] = np.clip(desc_pos[3], -max_rx, max_rx)
        warnings.append(f"Rx {desc_pos[3]:.1f}° clamped to {clamped[3]:.1f}°")
    
    # Giới hạn Ry (index 4)
    if abs(desc_pos[4]) > max_ry:
        clamped[4] = np.clip(desc_pos[4], -max_ry, max_ry)
        warnings.append(f"Ry {desc_pos[4]:.1f}° clamped to {clamped[4]:.1f}°")
    
    # Giới hạn Rz (index 5)
    if abs(desc_pos[5]) > max_rz:
        clamped[5] = np.clip(desc_pos[5], -max_rz, max_rz)
        warnings.append(f"Rz {desc_pos[5]:.1f}° clamped to {clamped[5]:.1f}°")
    
    return clamped, warnings

# Chuyển đổi pose từ camera coordinates sang robot coordinates
# Input: pose - tư thế trong camera, RT - ma trận chuyển đổi, grasp_offset - offset gắp
# Output: góc quay Euler và vị trí trong hệ tọa độ robot
def pose_to_robot_coordinates(pose: PoseEstimation, RT: np.ndarray, grasp_offset: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Chuyển đổi pose từ camera coordinates sang robot coordinates"""
    S = np.diag([1.0, -1.0, 1.0]).astype(np.float64)
    R_o3d = pose.rotation_matrix.astype(np.float64)
    R_cam_to_obj = S @ R_o3d @ S
    T_cam_to_obj = (S @ pose.center.reshape(3, 1).astype(np.float64))
    
    if grasp_offset is not None:
        grasp_offset_cam = R_cam_to_obj @ grasp_offset.reshape(3, 1)
        T_cam_to_grasp = T_cam_to_obj + grasp_offset_cam
    else:
        T_cam_to_grasp = T_cam_to_obj
    
    RT_cam_to_grasp = np.column_stack((R_cam_to_obj, T_cam_to_grasp))
    RT_cam_to_grasp = np.vstack((RT_cam_to_grasp, np.array([0, 0, 0, 1])))
    
    RT_grasp_to_base = RT @ RT_cam_to_grasp
    
    R_grasp_to_base = RT_grasp_to_base[:3, :3]
    T_grasp_to_base = RT_grasp_to_base[:3, 3]
    
    R_euler = target_point.rotationMatrixToEulerAngles(R_grasp_to_base) * 180 / np.pi
    
    # Giới hạn các góc quay để đảm bảo an toàn
    R_euler_clamped, warnings = clamp_rotation_angles(R_euler, max_rx=15.0, max_ry=180.0, max_rz=180.0)
    
    # In cảnh báo nếu có góc bị giới hạn (đã loại bỏ print để không spam terminal)
    # for warning in warnings:
    #     print(f"WARNING: {warning}")
    
    return R_euler_clamped, T_grasp_to_base

class YOLODetectionThread(QThread):
    """Thread để xử lý YOLO detection và pose estimation"""
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, list, list)  # color_frame, depth_frame, poses, targets
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.mutex = QMutex()
        self.model = None
        self.pipeline = None
        self.camera_info = None
        self.RT = None
        self.robot = None
        
        # Parameters
        self.weights_path = "best.pt"
        self.confidence = 0.25
        self.iou = 0.45
        self.robot_ip = "192.168.58.2"
        self.grasp_type = "center"
        self.tool = 2
        self.user = 0
        self.auto_move = False
        self.move_vel = 70
        self.move_acc = 100
        self.offset_down = [0, 0, 27, 0, 0, 0]
        self.offset_up = [0, 0, 100, 0, 0, 0]
        self.offset_up1 = [0, 0, 200, 0, 0, 0]
        self.moved_recently = False
        self.move_cooldown_s = 1.0  # Tăng thời gian cooldown
        self.last_move_time = 0.0
        # Vị trí chụp (original) mặc định như ảnh: đơn vị mm, deg (X Y Z RX RY RZ)
        self.capture_pose = [-283.066, -16.985, 490.007, -2.729, -0.254, -49.338]
        
        # ROI parameters
        self.roi_enabled = True
        self.roi_rect = None  # (x1, y1, x2, y2)
        self.saved_roi = None  # Lưu ROI trước đó
        self.roi_strict_mode = True  # True: crop trước khi detect, False: detect toàn ảnh rồi lọc
        
        # Trigger tính toán theo yêu cầu (không live)
        self.trigger_once = False
        
        # Parameters for slower scanning and coordinate consensus
        self.scan_delay = 0.5  # sleep delay in seconds after each frame scan
        self.coord_buffer = []  # buffer to store recent coordinates
        self.buffer_size = 10  # number of frames to accumulate
        self.consensus_threshold = 7  # min matching frames out of buffer_size
        self.no_target_count = 0  # counter for frames with no detected target
        
    def update_consensus(self, raw_coord):
        """Adds raw coordinate to buffer and returns consensus coordinate if stable, else None"""
        if raw_coord is None:
            self.no_target_count += 1
            if self.no_target_count >= 3:
                self.coord_buffer.clear()
            return None
            
        self.no_target_count = 0
        self.coord_buffer.append(raw_coord)
        if len(self.coord_buffer) > self.buffer_size:
            self.coord_buffer.pop(0)
            
        if len(self.coord_buffer) < self.buffer_size:
            self.status_update.emit(f"Đang thu thập dữ liệu tọa độ... ({len(self.coord_buffer)}/{self.buffer_size})")
            return None
            
        # Consensus check
        best_cluster = []
        dist_thresh = 10.0  # mm
        angle_thresh = 5.0  # degrees
        
        for i, c1 in enumerate(self.coord_buffer):
            cluster = [c1]
            for j, c2 in enumerate(self.coord_buffer):
                if i == j:
                    continue
                # Calculate translation distance
                dist = np.linalg.norm(np.array(c1[:3]) - np.array(c2[:3]))
                if dist > dist_thresh:
                    continue
                # Calculate rotation differences
                rx_diff = abs(c1[3] - c2[3])
                ry_diff = abs(c1[4] - c2[4])
                rz_diff = abs(c1[5] - c2[5])
                rz_diff = min(rz_diff, 180.0 - rz_diff)  # Rz is modulo 180
                
                if rx_diff < angle_thresh and ry_diff < angle_thresh and rz_diff < angle_thresh:
                    cluster.append(c2)
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
                
        C_size = len(best_cluster)
        if C_size >= self.consensus_threshold:
            # Average coordinates
            xs = [c[0] for c in best_cluster]
            ys = [c[1] for c in best_cluster]
            zs = [c[2] for c in best_cluster]
            rxs = [c[3] for c in best_cluster]
            rys = [c[4] for c in best_cluster]
            
            avg_x = np.mean(xs)
            avg_y = np.mean(ys)
            avg_z = np.mean(zs)
            avg_rx = np.mean(rxs)
            avg_ry = np.mean(rys)
            
            # Circular mean for Rz (modulo 180)
            sin_sum = 0.0
            cos_sum = 0.0
            for c in best_cluster:
                rad = np.radians(c[5] * 2.0)
                sin_sum += np.sin(rad)
                cos_sum += np.cos(rad)
            avg_rz = np.degrees(np.arctan2(sin_sum, cos_sum)) / 2.0
            avg_rz = avg_rz % 180.0
            
            consensus_coord = [avg_x, avg_y, avg_z, avg_rx, avg_ry, avg_rz]
            self.status_update.emit(f"Tọa độ ổn định (Trùng khớp: {C_size}/{self.buffer_size})")
            return consensus_coord
        else:
            self.status_update.emit(f"Tọa độ chưa ổn định, đang quét tiếp... (Trùng khớp nhiều nhất: {C_size}/{self.consensus_threshold})")
            return None

    # Thiết lập model YOLO với đường dẫn weights và các tham số
    # Input: weights_path - đường dẫn file model, confidence - ngưỡng tin cậy, iou - ngưỡng IoU
    # Output: True nếu thành công, False nếu có lỗi
    def setup_model(self, weights_path: str, confidence: float, iou: float):
        """Setup YOLO model"""
        try:
            self.weights_path = weights_path
            self.confidence = confidence
            self.iou = iou
            
            # Setup YOLOv8-seg (Ultralytics)
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.names = self.model.names
            
            # Tìm class ID cho "TOP"
            self.top_class_id = self.find_class_id("TOP")
            if self.top_class_id is not None:
                self.status_update.emit(f"Model loaded successfully - TOP class ID: {self.top_class_id}")
            else:
                self.status_update.emit("Model loaded successfully - TOP class not found, using class 0")
                self.top_class_id = 0
            
            # Hiển thị tất cả classes có trong model
            if hasattr(self, 'names') and self.names:
                if isinstance(self.names, dict):
                    classes_info = ", ".join([f"{class_id}: {name}" for class_id, name in self.names.items()])
                else:
                    classes_info = ", ".join([f"{i}: {n}" for i, n in enumerate(self.names)])
                self.status_update.emit(f"Available classes: {classes_info}")
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Model setup error: {str(e)}")
            return False
    
    # Tìm class ID cho tên class cụ thể trong model
    # Input: class_name - tên class cần tìm
    # Output: ID của class hoặc None nếu không tìm thấy
    def find_class_id(self, class_name: str):
        """Tìm class ID cho tên class cụ thể"""
        if hasattr(self, 'names') and self.names:
            if isinstance(self.names, dict):
                for class_id, name in self.names.items():
                    if str(name).lower() == class_name.lower():
                        return class_id
            else:
                for i, name in enumerate(self.names):
                    if str(name).lower() == class_name.lower():
                        return i
        return None
    
    # Thiết lập camera RealSense với độ phân giải và FPS
    # Input: width, height - độ phân giải, fps - số khung hình/giây
    # Output: True nếu thành công, False nếu có lỗi
    def setup_camera(self, width: int, height: int, fps: int):
        """Setup RealSense camera"""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            
            # Camera intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            self.camera_info = CameraInfo(
                width=width,
                height=height,
                fx=float(intrinsics.fx),
                fy=float(intrinsics.fy),
                cx=float(intrinsics.ppx),
                cy=float(intrinsics.ppy),
                scale=1.0 / depth_scale
            )
            
            self.status_update.emit("Camera initialized successfully")
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Camera setup error: {str(e)}")
            return False
    
    # Thiết lập kết nối robot với IP và các tham số tool/user
    # Input: ip - địa chỉ IP robot, tool - ID tool, user - ID user
    # Output: True nếu thành công, False nếu có lỗi
    def setup_robot(self, ip: str, tool: int, user: int):
        """Setup robot connection"""
        try:
            self.robot_ip = ip
            self.tool = tool
            self.user = user
            self.robot = Robot.RPC(ip)
            self.status_update.emit("Robot connected successfully")
            return True
        except Exception as e:
            self.error_occurred.emit(f"Robot connection error: {str(e)}")
            self.robot = None
            return False
    
    # Tải ma trận calibration từ file hoặc sử dụng mặc định
    # Input: không có
    # Output: True nếu thành công, False nếu có lỗi
    def load_calibration(self):
        """Load calibration matrix"""
        try:
            self.RT = np.load(r'c:\\Downloads\\RT.npy')
            if not validate_rt_matrix(self.RT):
                raise ValueError("Invalid RT matrix")
            self.status_update.emit("Calibration loaded successfully")
            return True
        except Exception as e:
            # Bất kỳ lỗi nào (kể cả không tìm thấy file) đều báo lỗi và dừng
            self.error_occurred.emit(f"Calibration error: {str(e)}")
            return False
    
    
    
    # Thiết lập ROI rectangle cho detection
    # Input: roi_rect - tọa độ ROI (x1, y1, x2, y2)
    # Output: không có
    def set_roi(self, roi_rect):
        """Set ROI rectangle"""
        self.roi_rect = roi_rect
        self.roi_enabled = roi_rect is not None
        if roi_rect is not None:
            self.saved_roi = roi_rect  # Lưu ROI để sử dụng lại
    
    # Tải ROI đã lưu trước đó
    # Input: không có
    # Output: True nếu có ROI đã lưu, False nếu không có
    def load_saved_roi(self):
        """Load saved ROI"""
        if self.saved_roi is not None:
            self.roi_rect = self.saved_roi
            self.roi_enabled = True
            return True
        return False
    
    # Kiểm tra xem detection có nằm trong ROI hay không
    # Input: x1, y1, x2, y2 - tọa độ bounding box của detection
    # Output: True nếu detection nằm trong ROI, False nếu không
    def is_detection_in_roi(self, x1, y1, x2, y2):
        """Check if detection is within ROI"""
        if not self.roi_enabled or self.roi_rect is None:
            return True
        
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_rect
        
        # Kiểm tra xem toàn bộ bounding box có nằm trong ROI hay không
        # Chỉ chấp nhận detection nếu toàn bộ box nằm trong ROI
        is_inside = (x1 >= roi_x1 and y1 >= roi_y1 and x2 <= roi_x2 and y2 <= roi_y2)
        
        # Debug logging để kiểm tra ROI filtering
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
            
        # Log mỗi 30 frames để tránh spam
        if self._debug_count % 30 == 0:
            self.status_update.emit(f"ROI check: detection({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) vs ROI({roi_x1:.0f},{roi_y1:.0f},{roi_x2:.0f},{roi_y2:.0f}) = {is_inside}")
        
        return is_inside
    
    # Sắp xếp detections theo độ ưu tiên: từ trái sang phải, từ trên xuống dưới
    # Input: detections, poses, targets - danh sách detections và thông tin liên quan
    # Output: danh sách đã được sắp xếp theo thứ tự ưu tiên
    def sort_detections_by_priority(self, detections, poses, targets):
        """Sort detections by priority: left to right, top to bottom"""
        if not detections or len(detections) == 0:
            return detections, poses, targets
        
        # Calculate center points for sorting
        centers = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections[i][:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y, i))
        
        # Sort by Y first (top to bottom), then by X (left to right)
        centers.sort(key=lambda x: (x[1], x[0]))
        
        # Reorder arrays based on sorted indices
        sorted_indices = [x[2] for x in centers]
        sorted_detections = [detections[i] for i in sorted_indices]
        sorted_poses = [poses[i] for i in sorted_indices] if poses else []
        sorted_targets = [targets[i] for i in sorted_indices] if targets else []
        
        return sorted_detections, sorted_poses, sorted_targets
    
    # Clone YOLOv5 repository nếu chưa có trong thư mục local
    # Input: local_dir - thư mục local để clone vào
    # Output: đường dẫn tuyệt đối đến thư mục YOLOv5
    def ensure_yolov5_repo(self, local_dir: str = "yolov5") -> str:
        """Clone YOLOv5 repo nếu chưa có"""
        if not os.path.isdir(local_dir) or not os.path.isdir(os.path.join(local_dir, ".git")):
            subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5.git", local_dir])
        return os.path.abspath(local_dir)
    
    # Thêm safe globals cho PyTorch để tránh lỗi serialization
    # Input: yolov5_dir - đường dẫn thư mục YOLOv5
    # Output: không có
    def add_torch_safe_globals(self, yolov5_dir: str):
        """Thêm safe globals cho PyTorch"""
        import torch.serialization as tser
        sys.path.insert(0, yolov5_dir)
        yolo_mod = importlib.import_module("models.yolo")
        try:
            tser.add_safe_globals([yolo_mod.SegmentationModel])
        except Exception:
            pass
    
    # Vòng lặp chính để thực hiện detection và pose estimation
    # Input: không có (sử dụng các thuộc tính đã setup)
    # Output: emit signals với kết quả detection
    def run(self):
        """Main detection loop"""
        if not self.model or not self.pipeline or not self.camera_info or self.RT is None:
            self.error_occurred.emit("System not properly initialized")
            return
        
        self.running = True
        self.status_update.emit("Detection started")
        
        try:
            while self.running:
                frames = self.pipeline.wait_for_frames()
                frames = self.align.process(frames)
                d = frames.get_depth_frame()
                c = frames.get_color_frame()
                
                if not d or not c:
                    continue
                
                depth_mm = np.asanyarray(d.get_data())
                color = np.asanyarray(c.get_data())
                vis_img = color.copy()
                
                # Crop ảnh theo ROI nếu có để tăng hiệu suất và độ chính xác
                if self.roi_enabled and self.roi_rect is not None and self.roi_strict_mode:
                    roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_rect
                    # Đảm bảo ROI nằm trong bounds của ảnh
                    roi_x1 = max(0, int(roi_x1))
                    roi_y1 = max(0, int(roi_y1))
                    roi_x2 = min(color.shape[1], int(roi_x2))
                    roi_y2 = min(color.shape[0], int(roi_y2))
                    
                    # Debug: Log ROI bounds
                    self.status_update.emit(f"ROI bounds: ({roi_x1}, {roi_y1}) to ({roi_x2}, {roi_y2})")
                    
                    # Crop ảnh theo ROI
                    color_cropped = color[roi_y1:roi_y2, roi_x1:roi_x2]
                    depth_cropped = depth_mm[roi_y1:roi_y2, roi_x1:roi_x2]
                    
                    # YOLOv8-seg inference trên ảnh đã crop
                    results = self.model.predict(color_cropped, conf=self.confidence, iou=self.iou, device='cpu', verbose=False)
                    
                    # Điều chỉnh tọa độ detection về tọa độ ảnh gốc
                    if results and len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes
                        if boxes.data is not None and len(boxes.data) > 0:
                            # Điều chỉnh tọa độ về ảnh gốc
                            boxes.xyxy[:, [0, 2]] += roi_x1  # x1, x2
                            boxes.xyxy[:, [1, 3]] += roi_y1  # y1, y2
                            
                            # Debug: Log adjusted coordinates
                            for i in range(len(boxes.data)):
                                x1, y1, x2, y2 = boxes.xyxy[i]
                                self.status_update.emit(f"Adjusted detection {i}: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
                else:
                    # YOLOv8-seg inference trực tiếp trên frame BGR
                    results = self.model.predict(color, conf=self.confidence, iou=self.iou, device='cpu', verbose=False)
                
                # Process detections
                detected_poses = []
                detected_targets_base = []
                valid_detections = []
                
                if results and len(results) > 0:
                    r = results[0]
                    boxes = r.boxes
                    masks_obj = getattr(r, 'masks', None)
                    
                    if boxes is not None and boxes.data is not None and len(boxes.data) > 0:
                        bxyxy = boxes.xyxy.cpu().numpy()
                        bconf = boxes.conf.cpu().numpy()
                        bcls = boxes.cls.cpu().numpy().astype(int)
                        
                        # Filter detections by ROI and TOP class only
                        valid_list = []  # (idx, x1,y1,x2,y2, conf, cls)
                        for i in range(len(bxyxy)):
                            x1, y1, x2, y2 = bxyxy[i]
                            cls = int(bcls[i])
                            conf = float(bconf[i])
                            
                            # Kiểm tra ROI một cách chặt chẽ hơn
                            is_in_roi = self.is_detection_in_roi(x1, y1, x2, y2)
                            
                            # Debug logging cho từng detection
                            self.status_update.emit(f"Detection {i}: class={cls}, conf={conf:.2f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), in_roi={is_in_roi}")
                            
                            if cls == self.top_class_id and is_in_roi:
                                valid_list.append((i, x1, y1, x2, y2, conf, cls))
                                self.status_update.emit(f"✓ Accepted detection {i}")
                            else:
                                if cls != self.top_class_id:
                                    self.status_update.emit(f"✗ Rejected detection {i}: wrong class ({cls} != {self.top_class_id})")
                                else:
                                    self.status_update.emit(f"✗ Rejected detection {i}: outside ROI")
                        
                        # Sort by priority: X first (left to right), then Y (top to bottom)
                        if valid_list:
                            self.status_update.emit(f"Found {len(valid_list)} TOP objects")
                            valid_list.sort(key=lambda x: (((x[1]+x[3])/2.0), ((x[2]+x[4])/2.0)))
                            valid_detections = [v[0] for v in valid_list]
                        else:
                            valid_detections = []
                            self.status_update.emit("No TOP objects detected")
                    
                    # Chuyển đổi depth sang meters
                    if self.roi_enabled and self.roi_rect is not None and self.roi_strict_mode:
                        depth_m = depth_cropped.astype(np.float32) * (1.0 / self.camera_info.scale)
                        color_rgb = color_cropped[:, :, ::-1].astype(np.float32) / 255.0
                    else:
                        depth_m = depth_mm.astype(np.float32) * (1.0 / self.camera_info.scale)
                        color_rgb = color[:, :, ::-1].astype(np.float32) / 255.0
                    
                    # Process filtered detections (chỉ TOP class)
                    has_first_target = False
                    for rank, orig_idx in enumerate(valid_detections):
                        x1, y1, x2, y2 = bxyxy[orig_idx]
                        conf = float(bconf[orig_idx])
                        cls = int(bcls[orig_idx])
                        
                        # Kiểm tra cuối cùng: đảm bảo detection thực sự nằm trong ROI
                        if not self.is_detection_in_roi(x1, y1, x2, y2):
                            self.status_update.emit(f"⚠️ WARNING: Detection {orig_idx} passed initial filter but is outside ROI!")
                            continue
                        
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Log thông tin detection TOP
                        if isinstance(self.names, dict):
                            class_name = self.names.get(int(cls), f"Class_{int(cls)}")
                        else:
                            try:
                                class_name = self.names[int(cls)]
                            except Exception:
                                class_name = f"Class_{int(cls)}"
                        self.status_update.emit(f"Detected TOP: {class_name} (ID: {int(cls)}) - Confidence: {conf:.2f}")
                        
                        # Lấy mask tương ứng
                        if masks_obj is None or masks_obj.data is None or masks_obj.data.shape[0] <= orig_idx:
                            continue
                        mk = masks_obj.data[orig_idx].cpu().numpy() > 0.5
                        # Sử dụng ảnh đúng để resize mask
                        target_img = color_cropped if (self.roi_enabled and self.roi_rect is not None and self.roi_strict_mode) else color
                        if mk.shape != target_img.shape[:2]:
                            mk = cv2.resize(mk.astype(np.uint8), (target_img.shape[1], target_img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                        # Vẽ mask lên vis_img: vật ưu tiên (đầu tiên) màu đỏ, còn lại xanh lá nhạt
                        selected_color = (0, 0, 255)  # BGR đỏ
                        light_green = (144, 238, 144)  # BGR xanh lá nhạt
                        mask_color = selected_color if rank == 0 else light_green
                        # CHỈ vẽ mask nếu detection nằm trong ROI
                        if self.is_detection_in_roi(x1, y1, x2, y2):
                            # Vẽ mask lên ảnh gốc (vis_img) thay vì ảnh đã crop
                            if self.roi_enabled and self.roi_rect is not None and self.roi_strict_mode:
                                # Tạo mask cho ảnh gốc
                                mask_full = np.zeros(color.shape[:2], dtype=bool)
                                roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_rect
                                roi_x1, roi_y1, roi_x2, roi_y2 = int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2)
                                mask_full[roi_y1:roi_y2, roi_x1:roi_x2] = mk
                                vis_img = draw_mask_overlay(vis_img, mask_full, color=mask_color, alpha=0.4)
                            else:
                                vis_img = draw_mask_overlay(vis_img, mk, color=mask_color, alpha=0.4)
                        else:
                            # Debug: Log khi không vẽ mask do nằm ngoài ROI
                            self.status_update.emit(f"⚠️ Skipping mask drawing for detection {rank}: outside ROI")

                        # CHỈ tạo point cloud và ước lượng pose nếu detection nằm trong ROI
                        if self.is_detection_in_roi(x1, y1, x2, y2):
                            # Tạo point cloud và ước lượng pose
                            try:
                                # Sử dụng mask đã được điều chỉnh cho ảnh gốc
                                if self.roi_enabled and self.roi_rect is not None and self.roi_strict_mode:
                                    mask_for_pcl = mask_full.astype(np.uint8)
                                else:
                                    mask_resized = cv2.resize(mk.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]),
                                                              interpolation=cv2.INTER_NEAREST) > 0
                                    mask_for_pcl = mask_resized.astype(np.uint8)

                                if self.roi_enabled and self.roi_rect is not None and self.roi_strict_mode:
                                    roi_offset = (int(self.roi_rect[0]), int(self.roi_rect[1]))
                                else:
                                    roi_offset = (0, 0)

                                detection_pcd = build_point_cloud_from_detection(
                                    color_rgb, depth_m, mask_for_pcl, self.camera_info,
                                    min_distance_m=0.1, max_distance_m=1.0, roi_offset=roi_offset
                                )

                                if detection_pcd is not None and len(detection_pcd.points) > 10:
                                    pose = estimate_pose_with_pca(detection_pcd)
                                    if pose is not None:
                                        # Decouple center estimation using 2D mask centroid and median depth
                                        mask_indices = np.where(mask_for_pcl > 0)
                                        if len(mask_indices[0]) > 0:
                                            vc = np.mean(mask_indices[0])
                                            uc = np.mean(mask_indices[1])
                                            
                                            # Add crop offset if roi_strict_mode is True
                                            if self.roi_enabled and self.roi_rect is not None and self.roi_strict_mode:
                                                roi_x1, roi_y1, _, _ = self.roi_rect
                                                uc_full = uc + roi_x1
                                                vc_full = vc + roi_y1
                                            else:
                                                uc_full = uc
                                                vc_full = vc
                                                
                                            # Get median depth of valid mask pixels
                                            depth_vals = depth_cropped[mask_for_pcl > 0] if (self.roi_enabled and self.roi_rect is not None and self.roi_strict_mode and 'depth_cropped' in locals()) else depth_mm[mask_for_pcl > 0]
                                            valid_depths = depth_vals[depth_vals > 0]
                                            if len(valid_depths) > 0:
                                                z_median_m = np.median(valid_depths) / 1000.0
                                                
                                                # Project to 3D OpenCV coords
                                                cx_cam = (uc_full - self.camera_info.cx) * z_median_m / self.camera_info.fx
                                                cy_cam = (vc_full - self.camera_info.cy) * z_median_m / self.camera_info.fy
                                                
                                                # Override pose center (Open3D style: Y is flipped)
                                                pose.center = np.array([cx_cam, -cy_cam, z_median_m])
                                        detected_poses.append(pose)

                                    # Tính toán vị trí gắp và chuyển đổi sang robot coordinates
                                    grasp_offset = calculate_grasp_offset(pose, self.grasp_type)
                                    R_robot, T_robot = pose_to_robot_coordinates(pose, self.RT, grasp_offset)
                                    
                                    # Chỉ log warning nếu góc quay thực sự có vấn đề
                                    if abs(R_robot[0]) >= 14.5 or abs(R_robot[1]) >= 170.0 or abs(R_robot[2]) >= 170.0:
                                        # Chỉ log một lần mỗi 10 giây để tránh spam
                                        if not hasattr(self, '_last_warning_time') or time.time() - self._last_warning_time > 10:
                                            self.status_update.emit(f"WARNING: Object {rank} rotation angles may be problematic")
                                            self._last_warning_time = time.time()

                                    # Lấy robot pose từ file poserobot.json (bắt buộc có file; không dùng mặc định)
                                    try:
                                        robot_pose_file = r"c:\Downloads\poserobot.json"
                                        if not os.path.exists(robot_pose_file):
                                            raise FileNotFoundError("poserobot.json không tồn tại")
                                        with open(robot_pose_file, 'r', encoding='utf-8') as f:
                                            robot_data = json.load(f)
                                        if "robot_pose" not in robot_data:
                                            raise ValueError("Thiếu khóa 'robot_pose' trong poserobot.json")
                                        ret = robot_data["robot_pose"]
                                        print("robot pose: ", ret)
                                        if isinstance(ret, (list, tuple)) and len(ret) >= 2:
                                            robot_point = np.array(ret[1])
                                            S_conv = np.diag([1.0, -1.0, 1.0]).astype(np.float64)
                                            R_cam_mat = S_conv @ pose.rotation_matrix.astype(np.float64) @ S_conv
                                            tvec = (S_conv @ pose.center.reshape(3, 1).astype(np.float64))
                                            R_base_deg, T_base_mm = target_point.target_point(R_cam_mat, tvec, robot_point, self.RT)
                                            
                                            # --- BẮT ĐẦU: TÍNH RZ DỰA TRÊN ĐỘ NGHIÊNG TRÁI/PHẢI CỦA CAMERA ---
                                            import math
                                            
                                            # Lấy góc 2D của trục X (trục dài nhất của phôi) trên mặt phẳng ảnh
                                            angle_2d_rad = math.atan2(pose.axes[1, 0], pose.axes[0, 0])
                                            angle_2d_deg = math.degrees(angle_2d_rad)
                                            
                                            # Tính góc nghiêng theo phương ngang (X) của camera
                                            n_cam = R_cam_mat[:, 2]
                                            tilt_x_deg = math.degrees(math.atan2(n_cam[0], abs(n_cam[2])))
                                            
                                            if tilt_x_deg > 15.0:
                                                new_rz = 0.0
                                                self.status_update.emit(f"-> Nghiêng phải ({tilt_x_deg:.1f}° > 15°): Ép RZ = 0")
                                            elif tilt_x_deg < -15.0:
                                                new_rz = 180.0
                                                self.status_update.emit(f"-> Nghiêng trái ({tilt_x_deg:.1f}° < -15°): Ép RZ = 180")
                                            else:
                                                new_rz = angle_2d_deg % 180.0
                                                
                                            # Áp dụng RZ mới
                                            R_base_deg[2] = new_rz
                                            # --- KẾT THÚC TÍNH RZ ---
                                            
                                            # --- BẮT ĐẦU: LỌC NHIỄU VÀ ỔN ĐỊNH TỌA ĐỘ BẰNG CONSENSUS FILTER ---
                                            if rank == 0:
                                                has_first_target = True
                                                raw_coord = [
                                                    float(T_base_mm[0]), float(T_base_mm[1]), float(T_base_mm[2]),
                                                    float(R_base_deg[0]), float(R_base_deg[1]), float(R_base_deg[2])
                                                ]
                                                stable_coord = self.update_consensus(raw_coord)
                                                
                                                if stable_coord is not None:
                                                    T_base_mm = np.array(stable_coord[:3])
                                                    R_base_deg = np.array(stable_coord[3:])
                                                    
                                                    # Xuất tọa độ ra JSON và thông báo cho GUI
                                                    import json
                                                    import os
                                                    try:
                                                        os.makedirs("c:\\Downloads", exist_ok=True)
                                                        json_file_path = "c:\\Downloads\\coordinates.json"
                                                        with open(json_file_path, "w") as f:
                                                            json_dict = {
                                                                "coords": [
                                                                    round(float(T_base_mm[0]), 1),
                                                                    round(float(T_base_mm[1]), 1),
                                                                    round(float(T_base_mm[2]), 1),
                                                                    round(float(R_base_deg[0]), 1),
                                                                    round(float(R_base_deg[1]), 1),
                                                                    round(float(R_base_deg[2]), 1)
                                                                ]
                                                            }
                                                            json.dump(json_dict, f, ensure_ascii=False, indent=2)
                                                        self.status_update.emit(f"UPDATE_COORDINATES:[{json_dict['coords'][0]},{json_dict['coords'][1]},{json_dict['coords'][2]},{json_dict['coords'][3]},{json_dict['coords'][4]},{json_dict['coords'][5]}]")
                                                    except Exception as e:
                                                        print(f"Error saving coordinates: {e}")
                                                else:
                                                    # Chưa đạt đồng thuận
                                                    T_base_mm = None
                                                    R_base_deg = None
                                            # --- KẾT THÚC LỌC NHIỄU ---
                                            
                                            # Debug tọa độ base ra màn hình UI
                                            if T_base_mm is not None:
                                                log_msg = f">> [Class: {class_name} | Grasp: {current_grasp_type}] Tọa độ Base -> X: {T_base_mm[0]:.1f}, Y: {T_base_mm[1]:.1f}, Z: {T_base_mm[2]:.1f} | Rx: {R_base_deg[0]:.1f}, Ry: {R_base_deg[1]:.1f}, Rz: {R_base_deg[2]:.1f}"
                                                self.status_update.emit(log_msg)
                                                detected_targets_base.append((R_base_deg, T_base_mm))
                                            else:
                                                detected_targets_base.append((None, None))
                                        else:
                                            detected_targets_base.append((None, None))
                                    except Exception as e:
                                        print(f"Error reading robot pose: {e}")
                                        detected_targets_base.append((None, None))

                                    # Vẽ trục hướng (giữ lại), không vẽ text/circle/đường khác
                                    # CHỈ vẽ trục nếu detection nằm trong ROI
                                    if self.is_detection_in_roi(x1, y1, x2, y2):
                                        cam_center = np.array([pose.center[0], -pose.center[1], pose.center[2]])
                                        # Tính tâm chiếu 2D làm gốc để vẽ trục
                                        if cam_center[2] != 0:
                                            center_2d = (
                                                int(cam_center[0] * self.camera_info.fx / cam_center[2] + self.camera_info.cx),
                                                int(cam_center[1] * self.camera_info.fy / cam_center[2] + self.camera_info.cy)
                                            )
                                        else:
                                            center_2d = None

                                        axis_length = 0.05
                                        axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                                        for j in range(3):
                                            end_point_3d_o3d = pose.center + pose.axes[:, j] * axis_length
                                            end_point_3d = np.array([end_point_3d_o3d[0], -end_point_3d_o3d[1], end_point_3d_o3d[2]])
                                            if end_point_3d[2] != 0 and cam_center[2] != 0:
                                                end_point_2d = (
                                                    int(end_point_3d[0] * self.camera_info.fx / end_point_3d[2] + self.camera_info.cx),
                                                    int(end_point_3d[1] * self.camera_info.fy / end_point_3d[2] + self.camera_info.cy)
                                                )
                                                if center_2d is not None and 0 <= end_point_2d[0] < self.camera_info.width and 0 <= end_point_2d[1] < self.camera_info.height and 0 <= center_2d[0] < self.camera_info.width and 0 <= center_2d[1] < self.camera_info.height:
                                                    cv2.line(vis_img, center_2d, end_point_2d, axis_colors[j], 3)
                                    else:
                                        # Debug: Log khi không vẽ trục do nằm ngoài ROI
                                        self.status_update.emit(f"⚠️ Skipping axis drawing for detection {rank}: outside ROI")
                            except Exception as e:
                                print(f"Error processing detection {rank}: {e}")
                        else:
                            # Debug: Log khi không xử lý point cloud do nằm ngoài ROI
                            self.status_update.emit(f"⚠️ Skipping point cloud processing for detection {rank}: outside ROI")
                    # Nếu không tìm thấy hoặc lỗi khi xử lý phôi đầu tiên, báo cho bộ lọc consensus
                    if not has_first_target:
                        self.update_consensus(None)
                else:
                    # Không có kết quả YOLO nào, báo cho bộ lọc consensus
                    self.update_consensus(None)
                
                # Auto move logic với cải thiện
                if self.auto_move and detected_targets_base and self.robot is not None:
                    try:
                        if not self.moved_recently and detected_targets_base[0][0] is not None:
                            Rb, Tb = detected_targets_base[0]
                            
                            # Kiểm tra xem phôi có bị nghiêng không (ngưỡng 2.0 độ để lọc nhiễu, ưu tiên xoay song song)
                            TILT_THRESHOLD = 2.0
                            rx_val = float(Rb[0]) if abs(float(Rb[0])) > TILT_THRESHOLD else 0.0
                            ry_val = float(Rb[1]) if abs(float(Rb[1])) > TILT_THRESHOLD else 0.0
                            
                            desc_pos = [float(Tb[0]), float(Tb[1]), float(Tb[2]), rx_val, ry_val, float(Rb[2])]
                            
                            # Kiểm tra góc quay trước khi di chuyển
                            desc_pos_clamped, warnings = clamp_desc_pos_angles(desc_pos, max_rx=15.0, max_ry=180.0, max_rz=180.0)
                            
                            # In ra tư thế gốc và tư thế sau khi đã giới hạn
                            self.status_update.emit(f"Tư thế gốc: X={desc_pos[0]:.1f}, Y={desc_pos[1]:.1f}, Z={desc_pos[2]:.1f}, RX={desc_pos[3]:.1f}, RY={desc_pos[4]:.1f}, RZ={desc_pos[5]:.1f}")
                            self.status_update.emit(f"Tư thế MoveL (sau cưỡng bức Rx=15°): X={desc_pos_clamped[0]:.1f}, Y={desc_pos_clamped[1]:.1f}, Z={desc_pos_clamped[2]:.1f}, RX={desc_pos_clamped[3]:.1f}, RY={desc_pos_clamped[4]:.1f}, RZ={desc_pos_clamped[5]:.1f}")
                            
                            # Cập nhật tọa độ hiện tại
                            self.status_update.emit(f"UPDATE_COORDINATES:{desc_pos_clamped}")
                            
                            # Chỉ di chuyển nếu góc quay hợp lý (không quá nhiều warning)
                            if len(warnings) <= 1:  # Cho phép 1 warning tối đa
                                for warning in warnings:
                                    self.status_update.emit(f"MoveL WARNING: {warning}")
                                
                                _ = self.robot.MoveL(desc_pos_clamped, self.tool, self.user, vel=self.move_vel, acc=self.move_acc, offset_flag=2, offset_pos=self.offset_up)
                                _ = self.robot.MoveL(desc_pos_clamped, self.tool, self.user, vel=self.move_vel, acc=self.move_acc, offset_flag=2, offset_pos=self.offset_down)
                                _ = self.robot.MoveL(desc_pos_clamped, self.tool, self.user, vel=self.move_vel, acc=self.move_acc, offset_flag=2, offset_pos=self.offset_up1)
                                # Xoay bút về góc chuẩn nhưng giữ vị trí hiện tại (sau khi lên)
                                current_pose = [float(Tb[0]), float(Tb[1]), float(Tb[2]) + 200, -2.729, -0.253, -49.338]
                                _ = self.robot.MoveL(current_pose, self.tool, self.user, vel=self.move_vel, acc=self.move_acc, offset_flag=0, offset_pos=[0,0,0,0,0,0])
                                
                                # Quay về vị trí chụp để chụp tiếp vật kế tiếp
                                _ = self.robot.MoveL(self.capture_pose, self.tool, self.user, vel=self.move_vel, acc=self.move_acc, offset_flag=0, offset_pos=[0,0,0,0,0,0])
                                self.status_update.emit("Auto Movement executed")
                                # Sau khi di chuyển xong, đánh dấu để tránh lặp
                                self.moved_recently = True
                                self.last_move_time = time.time()
                            else:
                                # Góc quay không hợp lý, bỏ qua
                                self.status_update.emit(f"Skipping movement - too many rotation warnings ({len(warnings)})")
                    except Exception as e:
                        self.status_update.emit(f"Auto move failed: {str(e)}")

                # Xuất tọa độ theo yêu cầu khi bấm Trigger (không live)
                if getattr(self, 'trigger_once', False) and detected_targets_base:
                    try:
                        if detected_targets_base[0][0] is not None:
                            Rb, Tb = detected_targets_base[0]
                            desc_pos = [float(Tb[0]), float(Tb[1]), float(Tb[2]), float(Rb[0]), float(Rb[1]), float(Rb[2])]
                            desc_pos_clamped, _ = clamp_desc_pos_angles(desc_pos, max_rx=15.0, max_ry=180.0, max_rz=180.0)
                            self.status_update.emit(f"UPDATE_COORDINATES:{desc_pos_clamped}")
                            self.status_update.emit("Đã xuất tọa độ theo Trigger")
                        else:
                            self.status_update.emit("Không có mục tiêu hợp lệ để xuất tọa độ")
                    except Exception as e:
                        self.error_occurred.emit(f"Trigger export error: {str(e)}")
                    finally:
                        self.trigger_once = False

                # Cooldown để cho phép lệnh tiếp theo sau khoảng thời gian
                if self.moved_recently and (time.time() - self.last_move_time >= self.move_cooldown_s):
                    self.moved_recently = False

                # Không vẽ ROI label hay bất kỳ text overlay nào lên frame

                # Tạo depth visualization
                depth_vis = self.create_depth_visualization(depth_mm)
                
                # Emit results
                self.frame_ready.emit(vis_img, depth_vis, detected_poses, detected_targets_base)
                
                # Chụp chậm lại (giảm tần suất quét)
                time.sleep(self.scan_delay)
                
        except Exception as e:
            self.error_occurred.emit(f"Detection error: {str(e)}")
        finally:
            self.status_update.emit("Detection stopped")
    
    # Tạo depth visualization từ depth image
    # Input: depth_mm - ảnh depth tính bằng mm
    # Output: ảnh depth đã được colorize và vẽ ROI nếu có
    def create_depth_visualization(self, depth_mm):
        """Tạo depth visualization"""
        # Normalize depth to 0-255 range
        depth_normalized = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # Vẽ ROI nếu có
        if self.roi_enabled and self.roi_rect is not None:
            x1, y1, x2, y2 = self.roi_rect
            cv2.rectangle(depth_colored, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        return depth_colored
    
    # Dừng detection thread và camera pipeline
    # Input: không có
    # Output: không có
    def stop(self):
        """Stop detection thread"""
        self.running = False
        if self.pipeline:
            self.pipeline.stop()

class ROIDrawingLabel(QLabel):
    """QLabel với khả năng vẽ và chỉnh sửa ROI bằng chuột"""
    roi_drawn = pyqtSignal(tuple)  # (x1, y1, x2, y2)
    roi_updated = pyqtSignal(tuple)  # (x1, y1, x2, y2)
    
    def __init__(self):
        super().__init__()
        self.drawing = False
        self.resizing = False
        self.moving = False
        self.start_point = None
        self.end_point = None
        self.roi_rect = None
        self.resize_handle = None
        self.setMouseTracking(True)
        self.setStyleSheet("border: 2px solid #bdc3c7; background-color: #2c3e50;")
        
        # Resize handle size
        self.handle_size = 8
        
    # Xử lý sự kiện nhấn chuột để bắt đầu vẽ ROI hoặc resize/move ROI hiện có
    # Input: event - sự kiện chuột
    # Output: không có
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.roi_rect is not None:
                # Check if clicking on resize handle
                handle = self.get_resize_handle(event.pos())
                if handle:
                    self.resizing = True
                    self.resize_handle = handle
                    return
                
                # Check if clicking inside ROI for moving
                if self.is_point_in_roi(event.pos()):
                    self.moving = True
                    self.start_point = event.pos()
                    return
            
            # Start drawing new ROI
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
    
    # Xử lý sự kiện di chuyển chuột để vẽ ROI, resize hoặc move ROI
    # Input: event - sự kiện chuột
    # Output: không có
    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()
        elif self.resizing and self.resize_handle:
            self.resize_roi(event.pos())
            self.update()
        elif self.moving:
            self.move_roi(event.pos())
            self.update()
        else:
            # Update cursor based on position
            if self.roi_rect is not None:
                handle = self.get_resize_handle(event.pos())
                if handle:
                    self.setCursor(self.get_resize_cursor(handle))
                elif self.is_point_in_roi(event.pos()):
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
    
    # Xử lý sự kiện thả chuột để hoàn thành vẽ ROI hoặc resize/move
    # Input: event - sự kiện chuột
    # Output: không có
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drawing:
                self.drawing = False
                self.end_point = event.pos()
                
                # Calculate ROI rectangle
                x1 = min(self.start_point.x(), self.end_point.x())
                y1 = min(self.start_point.y(), self.end_point.y())
                x2 = max(self.start_point.x(), self.end_point.x())
                y2 = max(self.start_point.y(), self.end_point.y())
                
                # Ensure minimum size
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.roi_rect = (x1, y1, x2, y2)
                    self.roi_drawn.emit(self.roi_rect)
                else:
                    self.roi_rect = None
                    self.roi_drawn.emit(None)
                
                self.update()
            
            elif self.resizing:
                self.resizing = False
                self.resize_handle = None
                if self.roi_rect:
                    self.roi_updated.emit(self.roi_rect)
            
            elif self.moving:
                self.moving = False
                if self.roi_rect:
                    self.roi_updated.emit(self.roi_rect)
    
    # Lấy resize handle tại vị trí chuột
    # Input: pos - vị trí chuột
    # Output: tên handle ('nw', 'ne', 'sw', 'se', 'n', 's', 'w', 'e') hoặc None
    def get_resize_handle(self, pos):
        """Get which resize handle is at the given position"""
        if not self.roi_rect:
            return None
        
        x1, y1, x2, y2 = self.roi_rect
        handles = {
            'nw': QRect(x1 - self.handle_size//2, y1 - self.handle_size//2, self.handle_size, self.handle_size),
            'ne': QRect(x2 - self.handle_size//2, y1 - self.handle_size//2, self.handle_size, self.handle_size),
            'sw': QRect(x1 - self.handle_size//2, y2 - self.handle_size//2, self.handle_size, self.handle_size),
            'se': QRect(x2 - self.handle_size//2, y2 - self.handle_size//2, self.handle_size, self.handle_size),
            'n': QRect((x1 + x2)//2 - self.handle_size//2, y1 - self.handle_size//2, self.handle_size, self.handle_size),
            's': QRect((x1 + x2)//2 - self.handle_size//2, y2 - self.handle_size//2, self.handle_size, self.handle_size),
            'w': QRect(x1 - self.handle_size//2, (y1 + y2)//2 - self.handle_size//2, self.handle_size, self.handle_size),
            'e': QRect(x2 - self.handle_size//2, (y1 + y2)//2 - self.handle_size//2, self.handle_size, self.handle_size)
        }
        
        for handle_name, handle_rect in handles.items():
            if handle_rect.contains(pos):
                return handle_name
        return None
    
    # Lấy cursor phù hợp cho resize handle
    # Input: handle - tên handle
    # Output: Qt cursor type
    def get_resize_cursor(self, handle):
        """Get cursor for resize handle"""
        cursors = {
            'nw': Qt.SizeFDiagCursor,
            'ne': Qt.SizeBDiagCursor,
            'sw': Qt.SizeBDiagCursor,
            'se': Qt.SizeFDiagCursor,
            'n': Qt.SizeVerCursor,
            's': Qt.SizeVerCursor,
            'w': Qt.SizeHorCursor,
            'e': Qt.SizeHorCursor
        }
        return cursors.get(handle, Qt.ArrowCursor)
    
    # Kiểm tra xem điểm có nằm trong ROI hay không
    # Input: pos - vị trí điểm
    # Output: True nếu điểm nằm trong ROI, False nếu không
    def is_point_in_roi(self, pos):
        """Check if point is inside ROI"""
        if not self.roi_rect:
            return False
        x1, y1, x2, y2 = self.roi_rect
        return x1 <= pos.x() <= x2 and y1 <= pos.y() <= y2
    
    # Thay đổi kích thước ROI dựa trên handle được chọn
    # Input: pos - vị trí chuột hiện tại
    # Output: không có
    def resize_roi(self, pos):
        """Resize ROI based on handle"""
        if not self.roi_rect or not self.resize_handle:
            return
        
        x1, y1, x2, y2 = self.roi_rect
        
        if 'n' in self.resize_handle:
            y1 = pos.y()
        if 's' in self.resize_handle:
            y2 = pos.y()
        if 'w' in self.resize_handle:
            x1 = pos.x()
        if 'e' in self.resize_handle:
            x2 = pos.x()
        
        # Ensure minimum size and valid coordinates
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            self.roi_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    
    # Di chuyển ROI theo vị trí chuột
    # Input: pos - vị trí chuột hiện tại
    # Output: không có
    def move_roi(self, pos):
        """Move ROI"""
        if not self.roi_rect or not self.start_point:
            return
        
        dx = pos.x() - self.start_point.x()
        dy = pos.y() - self.start_point.y()
        
        x1, y1, x2, y2 = self.roi_rect
        new_x1 = x1 + dx
        new_y1 = y1 + dy
        new_x2 = x2 + dx
        new_y2 = y2 + dy
        
        # Keep ROI within bounds
        label_size = self.size()
        new_x1 = max(0, min(new_x1, label_size.width() - (x2 - x1)))
        new_y1 = max(0, min(new_y1, label_size.height() - (y2 - y1)))
        new_x2 = new_x1 + (x2 - x1)
        new_y2 = new_y1 + (y2 - y1)
        
        self.roi_rect = (new_x1, new_y1, new_x2, new_y2)
        self.start_point = pos
    
    # Vẽ ROI và các handle lên widget
    # Input: event - sự kiện paint
    # Output: không có
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        
        if self.drawing and self.start_point and self.end_point:
            # Draw temporary rectangle while drawing
            painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
            painter.drawRect(QRect(self.start_point, self.end_point))
        
        elif self.roi_rect:
            # Draw ROI rectangle
            x1, y1, x2, y2 = self.roi_rect
            rect = QRect(x1, y1, x2 - x1, y2 - y1)
            
            # Draw main rectangle
            painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
            painter.drawRect(rect)
            
            # Draw resize handles
            painter.setPen(QPen(Qt.green, 1, Qt.SolidLine))
            painter.setBrush(Qt.green)
            
            handles = [
                QRect(x1 - self.handle_size//2, y1 - self.handle_size//2, self.handle_size, self.handle_size),
                QRect(x2 - self.handle_size//2, y1 - self.handle_size//2, self.handle_size, self.handle_size),
                QRect(x1 - self.handle_size//2, y2 - self.handle_size//2, self.handle_size, self.handle_size),
                QRect(x2 - self.handle_size//2, y2 - self.handle_size//2, self.handle_size, self.handle_size),
                QRect((x1 + x2)//2 - self.handle_size//2, y1 - self.handle_size//2, self.handle_size, self.handle_size),
                QRect((x1 + x2)//2 - self.handle_size//2, y2 - self.handle_size//2, self.handle_size, self.handle_size),
                QRect(x1 - self.handle_size//2, (y1 + y2)//2 - self.handle_size//2, self.handle_size, self.handle_size),
                QRect(x2 - self.handle_size//2, (y1 + y2)//2 - self.handle_size//2, self.handle_size, self.handle_size)
            ]
            
            for handle in handles:
                painter.drawRect(handle)
    
    # Xóa ROI rectangle và reset các trạng thái
    # Input: không có
    # Output: không có
    def clear_roi(self):
        """Clear ROI rectangle"""
        self.roi_rect = None
        self.start_point = None
        self.end_point = None
        self.resizing = False
        self.moving = False
        self.resize_handle = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detection_thread = None
        self.last_detected_targets = []
        self.current_coordinates = None  # Lưu trữ tọa độ hiện tại
        self.roi_config_file = "roi_config.json"
        # Kích thước khung hình hiện tại (color) dùng để quy đổi ROI
        self.current_frame_width = 640
        self.current_frame_height = 480
        # ROI ở hệ toạ độ khung hình (dùng cho detect/vẽ lên ảnh gốc)
        self.frame_roi_rect = None  # (x1, y1, x2, y2) theo pixel của frame
        self.init_ui()
        self.setup_connections()
        self.load_roi_config()
        # Tự động mở camera khi khởi động
        self.start_camera()
        
    # Khởi tạo giao diện người dùng với các panel và controls
    # Input: không có
    # Output: không có
    def init_ui(self):
        """Initialize UI components"""
        self.setWindowTitle("YOLO + PCA Pose Tracking System")
        self.setGeometry(100, 100, 1500, 1000)
        # Cố định kích thước window
        self.setFixedSize(1500, 1000)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel)
        
        # Center panel (camera displays)
        center_panel = self.create_center_panel()
        main_layout.addWidget(center_panel)
        
        # Right panel
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel)
        
        # Apply clean white theme with colorful accents
        self.setStyleSheet("""
            QMainWindow {
                background: #f8f9fa;
            }
            QFrame {
                background: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e9ecef;
                border-radius: 10px;
                margin-top: 1ex;
                padding-top: 15px;
                background: #ffffff;
                color: #495057;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #007bff;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton {
                background: #007bff;
                border: none;
                color: white;
                padding: 10px 18px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: #0056b3;
            }
            QPushButton:pressed {
                background: #004085;
            }
            QPushButton:disabled {
                background: #6c757d;
                color: #ffffff;
            }
            QPushButton#startButton {
                background: #28a745;
            }
            QPushButton#startButton:hover {
                background: #1e7e34;
            }
            QPushButton#stopButton {
                background: #dc3545;
            }
            QPushButton#stopButton:hover {
                background: #c82333;
            }
            QPushButton#moveRobotButton {
                background: #fd7e14;
            }
            QPushButton#moveRobotButton:hover {
                background: #e55a00;
            }
            QPushButton#drawROIButton {
                background: #6f42c1;
            }
            QPushButton#drawROIButton:hover {
                background: #5a32a3;
            }
            QPushButton#clearROIButton {
                background: #6c757d;
            }
            QPushButton#clearROIButton:hover {
                background: #545b62;
            }
            QPushButton#loadROIButton {
                background: #20c997;
            }
            QPushButton#loadROIButton:hover {
                background: #1aa179;
            }
            QPushButton#saveROIButton {
                background: #e83e8c;
            }
            QPushButton#saveROIButton:hover {
                background: #d91a72;
            }
            QLabel {
                color: #495057;
                font-weight: 500;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                border: 2px solid #ced4da;
                border-radius: 6px;
                padding: 8px 12px;
                background: #ffffff;
                color: #495057;
                font-size: 11px;
                min-height: 20px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #007bff;
                background: #f8f9fa;
            }
            QCheckBox {
                color: #495057;
                font-weight: 500;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #ced4da;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #007bff;
                background: #007bff;
            }
            QSlider::groove:horizontal {
                border: 1px solid #ced4da;
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #007bff;
                border: 1px solid #0056b3;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QTextEdit {
                border: 2px solid #ced4da;
                border-radius: 6px;
                background: #f8f9fa;
                color: #495057;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                padding: 8px;
                line-height: 1.4;
            }
            QProgressBar {
                border: 2px solid #ced4da;
                border-radius: 4px;
                text-align: center;
                background: #e9ecef;
                color: #495057;
            }
            QProgressBar::chunk {
                background: #007bff;
                border-radius: 2px;
            }
            QFrame#cameraFrame {
                border: 2px solid #007bff;
                border-radius: 8px;
                background: #2c3e50;
            }
            QFrame#statusFrame {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                background: #ffffff;
                padding: 10px;
            }
        """)
    
    # Tạo panel bên trái chứa các controls chính
    # Input: không có
    # Output: QFrame chứa các controls
    def create_left_panel(self):
        """Create left control panel"""
        panel = QFrame()
        panel.setFixedWidth(300)
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setFrameShadow(QFrame.Raised)
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Control group
        control_group = QGroupBox("Điều khiển chính")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(15, 15, 15, 15)
        
        # Start/Stop buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Bắt đầu")
        self.start_button.setObjectName("startButton")
        self.start_button.setMinimumHeight(40)
        self.stop_button = QPushButton("Dừng")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.setMinimumHeight(40)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        control_layout.addLayout(button_layout)
        
        # Trigger button (tính toán 1 lần và xuất tọa độ)
        self.trigger_button = QPushButton("Trigger xuất tọa độ")
        self.trigger_button.setMinimumHeight(36)
        self.trigger_button.setEnabled(False)
        control_layout.addWidget(self.trigger_button)

        # Move robot button
        self.move_robot_button = QPushButton("Di chuyển Robot")
        self.move_robot_button.setObjectName("moveRobotButton")
        self.move_robot_button.setEnabled(False)
        self.move_robot_button.setMinimumHeight(40)
        control_layout.addWidget(self.move_robot_button)
        
        # Auto move checkbox
        self.auto_move_checkbox = QCheckBox("Tự động di chuyển")
        control_layout.addWidget(self.auto_move_checkbox)
        
        layout.addWidget(control_group)
        
        # Model settings group
        model_group = QGroupBox("Cài đặt Model YOLO")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(10)
        model_layout.setContentsMargins(15, 15, 15, 15)
        
        # Weights path
        weights_layout = QHBoxLayout()
        self.weights_line_edit = QLineEdit("best.pt")
        self.weights_line_edit.setPlaceholderText("Đường dẫn file model (.pt)")
        self.browse_weights_button = QPushButton("...")
        self.browse_weights_button.setMaximumWidth(30)
        
        weights_layout.addWidget(self.weights_line_edit)
        weights_layout.addWidget(self.browse_weights_button)
        model_layout.addLayout(weights_layout)
        
        # Confidence
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setMinimum(0.1)
        self.confidence_spinbox.setMaximum(1.0)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setValue(0.25)
        confidence_layout.addWidget(self.confidence_spinbox)
        model_layout.addLayout(confidence_layout)
        
        # IoU
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU:"))
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setMinimum(0.1)
        self.iou_spinbox.setMaximum(1.0)
        self.iou_spinbox.setSingleStep(0.05)
        self.iou_spinbox.setValue(0.45)
        iou_layout.addWidget(self.iou_spinbox)
        model_layout.addLayout(iou_layout)
        
        layout.addWidget(model_group)
        
        # Robot settings group
        robot_group = QGroupBox("Cài đặt Robot")
        robot_layout = QVBoxLayout(robot_group)
        robot_layout.setSpacing(10)
        robot_layout.setContentsMargins(15, 15, 15, 15)
        
        # Robot IP
        robot_ip_layout = QHBoxLayout()
        robot_ip_layout.addWidget(QLabel("Robot IP:"))
        self.robot_ip_line_edit = QLineEdit("192.168.58.2")
        robot_ip_layout.addWidget(self.robot_ip_line_edit)
        robot_layout.addLayout(robot_ip_layout)
        
        # Grasp type
        grasp_type_layout = QHBoxLayout()
        grasp_type_layout.addWidget(QLabel("Kiểu gắp:"))
        self.grasp_type_combo = QComboBox()
        self.grasp_type_combo.addItems(["center", "top", "side", "pen_grasp"])
        grasp_type_layout.addWidget(self.grasp_type_combo)
        robot_layout.addLayout(grasp_type_layout)
        
        # Tool and User
        tool_user_layout = QHBoxLayout()
        tool_user_layout.addWidget(QLabel("Tool:"))
        self.tool_spinbox = QSpinBox()
        self.tool_spinbox.setMinimum(0)
        self.tool_spinbox.setMaximum(10)
        self.tool_spinbox.setValue(2)
        tool_user_layout.addWidget(self.tool_spinbox)
        
        tool_user_layout.addWidget(QLabel("User:"))
        self.user_spinbox = QSpinBox()
        self.user_spinbox.setMinimum(0)
        self.user_spinbox.setMaximum(10)
        self.user_spinbox.setValue(0)
        tool_user_layout.addWidget(self.user_spinbox)
        robot_layout.addLayout(tool_user_layout)
        
        layout.addWidget(robot_group)
        
        # Status group
        status_group = QGroupBox("Trạng thái hệ thống")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(10)
        status_layout.setContentsMargins(15, 15, 15, 15)
        
        self.status_label = QLabel("Trạng thái: Sẵn sàng")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        self.fps_display_label = QLabel("FPS: 0")
        status_layout.addWidget(self.fps_display_label)
        
        self.detection_count_label = QLabel("TOP Objects: 0")
        status_layout.addWidget(self.detection_count_label)
        
        self.progress_bar = QProgressBar()
        status_layout.addWidget(self.progress_bar)
        
        layout.addWidget(status_group)
        
        # Remove spacer to eliminate empty space
        
        return panel
    
    # Tạo panel giữa chứa camera và depth display
    # Input: không có
    # Output: QFrame chứa camera displays
    def create_center_panel(self):
        """Create center display panel for camera and depth"""
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setFrameShadow(QFrame.Raised)
        
        # Main vertical layout
        main_layout = QVBoxLayout(panel)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Create a container widget for camera displays
        camera_container = QWidget()
        camera_container_layout = QVBoxLayout(camera_container)
        camera_container_layout.setSpacing(5)
        camera_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Camera frame
        self.camera_frame = QFrame()
        self.camera_frame.setObjectName("cameraFrame")
        self.camera_frame.setFixedSize(600, 450)
        self.camera_frame.setFrameShape(QFrame.Box)
        self.camera_frame.setFrameShadow(QFrame.Raised)
        
        camera_layout = QVBoxLayout(self.camera_frame)
        self.camera_label = ROIDrawingLabel()
        self.camera_label.setText("Camera Feed")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("color: #ecf0f1; font-size: 16px; font-weight: bold;")
        camera_layout.addWidget(self.camera_label)
        
        camera_container_layout.addWidget(self.camera_frame)
        
        # Depth frame
        self.depth_frame = QFrame()
        self.depth_frame.setObjectName("cameraFrame")
        self.depth_frame.setFixedSize(600, 450)
        self.depth_frame.setFrameShape(QFrame.Box)
        self.depth_frame.setFrameShadow(QFrame.Raised)
        
        depth_layout = QVBoxLayout(self.depth_frame)
        self.depth_label = QLabel()
        self.depth_label.setText("Depth Feed")
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setStyleSheet("color: #ecf0f1; font-size: 16px; font-weight: bold;")
        depth_layout.addWidget(self.depth_label)
        
        camera_container_layout.addWidget(self.depth_frame)
        
        # Status frame
        self.status_frame = QFrame()
        self.status_frame.setObjectName("statusFrame")
        self.status_frame.setFrameShape(QFrame.Box)
        self.status_frame.setFrameShadow(QFrame.Raised)
        self.status_frame.setFixedHeight(120)  # Tăng từ 100 lên 120 để có chỗ cho grasp pose
        
        status_frame_layout = QVBoxLayout(self.status_frame)
        status_frame_layout.setSpacing(3)
        status_frame_layout.setContentsMargins(8, 8, 8, 8)
        
        self.pose_info_label = QLabel("Tọa độ Vật")
        self.pose_info_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 12px;")
        status_frame_layout.addWidget(self.pose_info_label)
        
        # Thêm label hiển thị tọa độ vật
        self.grasp_pose_label = QLabel("Chưa có tọa độ")
        self.grasp_pose_label.setWordWrap(True)
        self.grasp_pose_label.setStyleSheet("color: #e74c3c; font-size: 10px; font-weight: bold;")
        self.grasp_pose_label.setMaximumHeight(60)
        status_frame_layout.addWidget(self.grasp_pose_label)
        
        camera_container_layout.addWidget(self.status_frame)
        
        # Create horizontal layout to center the camera container
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        horizontal_layout.addWidget(camera_container)
        horizontal_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        main_layout.addLayout(horizontal_layout)
        
        # Add spacer to push everything to top
        main_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        return panel
    
    # Tạo panel bên phải chứa ROI controls và log console
    # Input: không có
    # Output: QFrame chứa các controls phụ
    def create_right_panel(self):
        """Create right control panel"""
        panel = QFrame()
        panel.setFixedWidth(300)
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setFrameShadow(QFrame.Raised)
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # ROI controls group
        roi_group = QGroupBox("ROI Controls")
        roi_group_layout = QVBoxLayout(roi_group)
        roi_group_layout.setSpacing(5)
        roi_group_layout.setContentsMargins(8, 8, 8, 8)
        
        # Row 1
        roi_row1 = QHBoxLayout()
        self.draw_roi_button = QPushButton("Vẽ ROI")
        self.draw_roi_button.setObjectName("drawROIButton")
        self.draw_roi_button.setMinimumHeight(35)
        self.clear_roi_button = QPushButton("Xóa ROI")
        self.clear_roi_button.setObjectName("clearROIButton")
        self.clear_roi_button.setMinimumHeight(35)
        roi_row1.addWidget(self.draw_roi_button)
        roi_row1.addWidget(self.clear_roi_button)
        
        # Row 2
        roi_row2 = QHBoxLayout()
        self.load_roi_button = QPushButton("Tải ROI")
        self.load_roi_button.setObjectName("loadROIButton")
        self.load_roi_button.setMinimumHeight(35)
        self.save_roi_button = QPushButton("Lưu ROI")
        self.save_roi_button.setObjectName("saveROIButton")
        self.save_roi_button.setMinimumHeight(35)
        self.save_roi_button.setEnabled(False)
        roi_row2.addWidget(self.load_roi_button)
        roi_row2.addWidget(self.save_roi_button)
        
        roi_group_layout.addLayout(roi_row1)
        roi_group_layout.addLayout(roi_row2)
        layout.addWidget(roi_group)
        
        # Display settings group
        display_group = QGroupBox("Hiển thị")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(5)
        display_layout.setContentsMargins(8, 8, 8, 8)
        
        self.show_point_cloud_checkbox = QCheckBox("Hiển thị Point Cloud")
        self.show_point_cloud_checkbox.setChecked(True)
        display_layout.addWidget(self.show_point_cloud_checkbox)
        
        self.show_pose_axes_checkbox = QCheckBox("Hiển thị Pose Axes")
        self.show_pose_axes_checkbox.setChecked(True)
        display_layout.addWidget(self.show_pose_axes_checkbox)
        
        self.show_pyramid_checkbox = QCheckBox("Hiển thị Pyramid Camera")
        self.show_pyramid_checkbox.setChecked(True)
        display_layout.addWidget(self.show_pyramid_checkbox)
        
        # Point size
        point_size_layout = QHBoxLayout()
        point_size_layout.addWidget(QLabel("Kích thước điểm:"))
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(10)
        self.point_size_slider.setValue(3)
        point_size_layout.addWidget(self.point_size_slider)
        display_layout.addLayout(point_size_layout)
        
        layout.addWidget(display_group)
        
        # Camera settings group
        camera_group = QGroupBox("Cài đặt Camera")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setSpacing(5)
        camera_layout.setContentsMargins(8, 8, 8, 8)
        
        # Resolution
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Độ phân giải:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "848x480", "1280x720"])
        resolution_layout.addWidget(self.resolution_combo)
        camera_layout.addLayout(resolution_layout)
        
        # FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setMinimum(15)
        self.fps_spinbox.setMaximum(60)
        self.fps_spinbox.setValue(30)
        fps_layout.addWidget(self.fps_spinbox)
        camera_layout.addLayout(fps_layout)
        
        layout.addWidget(camera_group)
        
        # Log group
        log_group = QGroupBox("Log Console")
        log_layout = QVBoxLayout(log_group)
        log_layout.setSpacing(5)
        log_layout.setContentsMargins(8, 8, 8, 8)
        
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setMinimumHeight(300)  # Tăng chiều cao tối thiểu
        self.log_text_edit.setMaximumHeight(400)  # Tăng chiều cao tối đa
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setWordWrapMode(0)  # No wrap để tránh chồng chữ
        log_layout.addWidget(self.log_text_edit)
        
        # Thêm spacer để đẩy nút xuống dưới
        log_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Log buttons
        log_buttons_layout = QHBoxLayout()
        self.clear_log_button = QPushButton("Xóa Log")
        self.clear_log_button.setMinimumWidth(80)
        self.clear_log_button.setMinimumHeight(25)
        self.save_log_button = QPushButton("Lưu Log")
        self.save_log_button.setMinimumWidth(80)
        self.save_log_button.setMinimumHeight(25)
        
        log_buttons_layout.addWidget(self.clear_log_button)
        log_buttons_layout.addWidget(self.save_log_button)
        log_buttons_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        log_layout.addLayout(log_buttons_layout)
        
        layout.addWidget(log_group)
        
        # Remove spacer to eliminate empty space
        
        return panel
    
    # Thiết lập các kết nối signal-slot cho UI controls
    # Input: không có
    # Output: không có
    def setup_connections(self):
        """Setup signal connections"""
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        self.move_robot_button.clicked.connect(self.move_robot)
        self.trigger_button.clicked.connect(self.trigger_export_coordinates)
        self.browse_weights_button.clicked.connect(self.browse_weights)
        self.clear_log_button.clicked.connect(self.clear_log)
        self.save_log_button.clicked.connect(self.save_log)
        
        # ROI connections
        self.draw_roi_button.clicked.connect(self.enable_roi_drawing)
        self.clear_roi_button.clicked.connect(self.clear_roi)
        self.load_roi_button.clicked.connect(self.load_roi)
        self.save_roi_button.clicked.connect(self.save_roi)
        self.camera_label.roi_drawn.connect(self.on_roi_drawn)
        self.camera_label.roi_updated.connect(self.on_roi_updated)
        
        # Timer for FPS update
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)  # Update every second
        
        # Timer for camera display update
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_display)
        self.camera_timer.start(33)  # Update every 33ms (~30 FPS)
        
        self.frame_count = 0
        self.last_fps_time = time.time()

        # Timer theo dõi trigger từ phần mềm ngoài (file-based)
        self.external_trigger_path = r"c:\\Downloads\\trigger.txt"
        self.external_trigger_timer = QTimer()
        self.external_trigger_timer.timeout.connect(self.check_external_trigger)
        self.external_trigger_timer.start(100)  # kiểm tra mỗi 100ms
    
    # Bắt đầu YOLO detection với các tham số đã cấu hình
    # Input: không có (lấy từ UI controls)
    # Output: không có
    def start_detection(self):
        """Start YOLO detection"""
        try:
            # Dừng ROI pipeline và timer nếu đang chạy
            if hasattr(self, 'roi_pipeline') and self.roi_pipeline:
                self.roi_pipeline.stop()
                self.roi_pipeline = None
            
            # Dừng camera timer
            if hasattr(self, 'camera_timer'):
                self.camera_timer.stop()
            
            # Get parameters
            weights_path = self.weights_line_edit.text()
            confidence = self.confidence_spinbox.value()
            iou = self.iou_spinbox.value()
            
            # Get camera settings
            resolution = self.resolution_combo.currentText()
            width, height = map(int, resolution.split('x'))
            fps = self.fps_spinbox.value()
            
            # Get robot settings
            robot_ip = self.robot_ip_line_edit.text()
            tool = self.tool_spinbox.value()
            user = self.user_spinbox.value()
            grasp_type = self.grasp_type_combo.currentText()
            auto_move = self.auto_move_checkbox.isChecked()
            
            # Create detection thread
            self.detection_thread = YOLODetectionThread()
            self.detection_thread.frame_ready.connect(self.update_frame)
            self.detection_thread.status_update.connect(self.update_status)
            self.detection_thread.error_occurred.connect(self.show_error)
            
            # Setup systems
            if not self.detection_thread.setup_model(weights_path, confidence, iou):
                return
            
            if not self.detection_thread.setup_camera(width, height, fps):
                return
            
            if not self.detection_thread.setup_robot(robot_ip, tool, user):
                return
            
            if not self.detection_thread.load_calibration():
                return
            
            # Truyền thêm tham số điều khiển
            self.detection_thread.grasp_type = grasp_type
            self.detection_thread.auto_move = auto_move
            
            # Start detection
            self.detection_thread.start()
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.move_robot_button.setEnabled(True)
            self.trigger_button.setEnabled(True)
            
            self.log_message("Detection started successfully")
            
        except Exception as e:
            self.show_error(f"Failed to start detection: {str(e)}")
    
    # Dừng YOLO detection và khôi phục camera display
    # Input: không có
    # Output: không có
    def stop_detection(self):
        """Stop YOLO detection"""
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.wait()
            self.detection_thread = None
        
        # Khởi động lại camera timer
        if hasattr(self, 'camera_timer'):
            self.camera_timer.start(33)
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.move_robot_button.setEnabled(False)
        self.trigger_button.setEnabled(False)

    # Kích hoạt tính toán một lần và xuất tọa độ
    # Input: không có
    # Output: không có
    def trigger_export_coordinates(self):
        """Kích hoạt tính toán một lần và xuất tọa độ"""
        if hasattr(self, 'detection_thread') and self.detection_thread:
            self.detection_thread.trigger_once = True
            self.log_message("Đã kích hoạt Trigger: sẽ xuất tọa độ ở khung hình kế tiếp")
        else:
            self.log_message("Chưa khởi động detection")
        
    # Kiểm tra trigger ngoài (file c:\Downloads\trigger.txt)
    # Nếu phát hiện file, kích hoạt trigger_once và xóa file
    def check_external_trigger(self):
        try:
            if os.path.exists(self.external_trigger_path):
                if hasattr(self, 'detection_thread') and self.detection_thread:
                    self.detection_thread.trigger_once = True
                    self.log_message("Nhận trigger ngoài: sẽ xuất tọa độ ở khung hình kế tiếp")
                    # Xóa file trigger sau khi nhận
                    os.remove(self.external_trigger_path)
                else:
                    # Chưa chạy detection: giữ file để xử lý sau
                    pass
        except Exception as e:
            # Không popup; chỉ log để không làm phiền người dùng
            self.log_message(f"External trigger error: {str(e)}")
    
    # Di chuyển robot đến vị trí object đã detect
    # Input: không có (sử dụng last_detected_targets)
    # Output: không có
    def move_robot(self):
        """Move robot to detected object"""
        if not self.detection_thread or not self.detection_thread.robot:
            self.show_error("Robot not connected")
            return
        if not self.last_detected_targets or self.last_detected_targets[0][0] is None:
            self.show_error("Chưa có mục tiêu hợp lệ để di chuyển")
            return
        try:
            Rb, Tb = self.last_detected_targets[0]
            desc_pos = [float(Tb[0]), float(Tb[1]), float(Tb[2]), float(Rb[0]), float(Rb[1]), float(Rb[2])]
            
            # Giới hạn góc quay trước khi MoveL
            desc_pos_clamped, warnings = clamp_desc_pos_angles(desc_pos, max_rx=15.0, max_ry=180.0, max_rz=180.0)
            
            # In ra tư thế gốc và tư thế sau khi đã giới hạn
            self.log_message(f"Tư thế gốc: X={desc_pos[0]:.1f}, Y={desc_pos[1]:.1f}, Z={desc_pos[2]:.1f}, RX={desc_pos[3]:.1f}, RY={desc_pos[4]:.1f}, RZ={desc_pos[5]:.1f}")
            self.log_message(f"Tư thế MoveL (sau cưỡng bức Rx=15°): X={desc_pos_clamped[0]:.1f}, Y={desc_pos_clamped[1]:.1f}, Z={desc_pos_clamped[2]:.1f}, RX={desc_pos_clamped[3]:.1f}, RY={desc_pos_clamped[4]:.1f}, RZ={desc_pos_clamped[5]:.1f}")
            
            # Cập nhật tọa độ hiện tại
            self.update_coordinates(desc_pos_clamped)
            
            for warning in warnings:
                self.log_message(f"MoveL WARNING: {warning}")
            
            offset = [0, 0, -20, 0, 0, 0]
            ret1 = self.detection_thread.robot.MoveL(desc_pos_clamped, self.detection_thread.tool, self.detection_thread.user, vel=5, acc=100, offset_flag=2, offset_pos=[0,0,70,0,0,0])
            ret2 = self.detection_thread.robot.MoveL(desc_pos_clamped, self.detection_thread.tool, self.detection_thread.user, vel=5, acc=100, offset_flag=2, offset_pos=offset)
            # Quay về vị trí chụp sau khi gắp xong
            ret3 = self.detection_thread.robot.MoveL(self.detection_thread.capture_pose, self.detection_thread.tool, self.detection_thread.user, vel=5, acc=100, offset_flag=0, offset_pos=[0,0,0,0,0,0])
            self.log_message(f"Movement results: up={ret1}, down={ret2}, back_to_capture={ret3}")
        except Exception as e:
            self.show_error(f"Movement failed: {str(e)}")
    
    # Mở dialog để chọn file weights YOLO
    # Input: không có
    # Output: không có
    def browse_weights(self):
        """Browse for weights file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO weights file", "", "PyTorch files (*.pt)"
        )
        if file_path:
            self.weights_line_edit.setText(file_path)
    
    # Xóa nội dung log console
    # Input: không có
    # Output: không có
    def clear_log(self):
        """Clear log console"""
        self.log_text_edit.clear()
    
    # Lưu log console ra file
    # Input: không có
    # Output: không có
    def save_log(self):
        """Save log to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save log file", "", "Text files (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text_edit.toPlainText())
                self.log_message(f"Log saved to {file_path}")
            except Exception as e:
                self.show_error(f"Failed to save log: {str(e)}")
    
    # Kích hoạt chế độ vẽ ROI bằng chuột
    # Input: không có
    # Output: không có
    def enable_roi_drawing(self):
        """Enable ROI drawing mode"""
        self.camera_label.setCursor(Qt.CrossCursor)
        self.log_message("ROI drawing enabled - Click and drag to draw ROI")
    
    # Xóa ROI hiện tại và reset detection thread
    # Input: không có
    # Output: không có
    def clear_roi(self):
        """Clear ROI"""
        self.camera_label.clear_roi()
        if self.detection_thread:
            self.detection_thread.set_roi(None)
        self.save_roi_button.setEnabled(False)
        self.log_message("ROI cleared")
    
    # Tải ROI đã lưu từ file config
    # Input: không có
    # Output: không có
    def load_roi(self):
        """Load saved ROI"""
        try:
            if os.path.exists(self.roi_config_file):
                with open(self.roi_config_file, 'r') as f:
                    roi_data = json.load(f)
                    if "roi_rect" in roi_data:
                        # Lưu ROI frame và quy đổi sang toạ độ label
                        self.frame_roi_rect = tuple(roi_data["roi_rect"])
                        label_size = self.camera_label.size()
                        label_w = max(1, label_size.width())
                        label_h = max(1, label_size.height())
                        frame_w = max(1, int(self.current_frame_width))
                        frame_h = max(1, int(self.current_frame_height))
                        inv_scale_x = label_w / frame_w
                        inv_scale_y = label_h / frame_h
                        lx1 = int(self.frame_roi_rect[0] * inv_scale_x)
                        ly1 = int(self.frame_roi_rect[1] * inv_scale_y)
                        lx2 = int(self.frame_roi_rect[2] * inv_scale_x)
                        ly2 = int(self.frame_roi_rect[3] * inv_scale_y)
                        self.camera_label.roi_rect = (lx1, ly1, lx2, ly2)
                        self.camera_label.update()
                        
                        # Set ROI cho detection thread nếu đang chạy (dùng toạ độ frame)
                        if self.detection_thread:
                            self.detection_thread.set_roi(self.frame_roi_rect)
                        
                        self.log_message("ROI loaded successfully")
                    else:
                        self.log_message("No ROI data found in config file")
            else:
                self.log_message("ROI config file not found")
        except Exception as e:
            self.log_message(f"Failed to load ROI: {str(e)}")
    
    # Xử lý sự kiện ROI được vẽ xong
    # Input: roi_rect - tọa độ ROI đã vẽ
    # Output: không có
    def on_roi_drawn(self, roi_rect):
        """Handle ROI drawn event"""
        if roi_rect is not None:
            # Lưu ROI theo toạ độ label để người dùng chỉnh sửa
            self.camera_label.roi_rect = roi_rect
            
            # Quy đổi sang toạ độ khung hình thực tế
            label_size = self.camera_label.size()
            label_width = max(1, label_size.width())
            label_height = max(1, label_size.height())
            frame_w = max(1, int(self.current_frame_width))
            frame_h = max(1, int(self.current_frame_height))
            scale_x = frame_w / label_width
            scale_y = frame_h / label_height
            x1 = int(roi_rect[0] * scale_x)
            y1 = int(roi_rect[1] * scale_y)
            x2 = int(roi_rect[2] * scale_x)
            y2 = int(roi_rect[3] * scale_y)
            self.frame_roi_rect = (x1, y1, x2, y2)
            
            if hasattr(self, 'detection_thread') and self.detection_thread:
                self.detection_thread.set_roi(self.frame_roi_rect)
            
            self.log_message(f"ROI set (frame): ({x1}, {y1}) to ({x2}, {y2})")
            # Enable save button
            self.save_roi_button.setEnabled(True)
        else:
            self.log_message("ROI drawing cancelled")
        
        # Reset cursor
        self.camera_label.setCursor(Qt.ArrowCursor)
    
    # Xử lý sự kiện ROI được cập nhật (resize/move)
    # Input: roi_rect - tọa độ ROI đã cập nhật
    # Output: không có
    def on_roi_updated(self, roi_rect):
        """Handle ROI updated event (resize/move)"""
        if roi_rect is not None:
            # Cập nhật ROI theo label để tiếp tục chỉnh sửa
            self.camera_label.roi_rect = roi_rect
            
            # Quy đổi sang toạ độ khung hình thực tế
            label_size = self.camera_label.size()
            label_width = max(1, label_size.width())
            label_height = max(1, label_size.height())
            frame_w = max(1, int(self.current_frame_width))
            frame_h = max(1, int(self.current_frame_height))
            scale_x = frame_w / label_width
            scale_y = frame_h / label_height
            x1 = int(roi_rect[0] * scale_x)
            y1 = int(roi_rect[1] * scale_y)
            x2 = int(roi_rect[2] * scale_x)
            y2 = int(roi_rect[3] * scale_y)
            self.frame_roi_rect = (x1, y1, x2, y2)
            
            if hasattr(self, 'detection_thread') and self.detection_thread:
                self.detection_thread.set_roi(self.frame_roi_rect)
            
            self.log_message(f"ROI updated (frame): ({x1}, {y1}) to ({x2}, {y2})")
            # Enable save button
            self.save_roi_button.setEnabled(True)
    
    # Lưu ROI hiện tại vào file config
    # Input: không có
    # Output: không có
    def save_roi(self):
        """Save current ROI configuration"""
        if getattr(self, 'frame_roi_rect', None) is not None:
            self.save_roi_config()
            self.log_message("ROI saved successfully")
            self.save_roi_button.setEnabled(False)
        else:
            self.log_message("No ROI to save")
    
    # Cập nhật hiển thị camera frame với kết quả detection
    # Input: color_frame, depth_frame, poses, targets - dữ liệu từ detection thread
    # Output: không có
    def update_frame(self, color_frame, depth_frame, poses, targets):
        """Update camera frame display"""
        # Lưu mục tiêu mới nhất phục vụ nút di chuyển
        self.last_detected_targets = targets if targets is not None else []
        # Cập nhật kích thước khung hình hiện tại
        if color_frame is not None:
            self.current_frame_height, self.current_frame_width = color_frame.shape[:2]
        
        # Vẽ ROI lên color frame nếu có (dùng toạ độ frame)
        if getattr(self, 'frame_roi_rect', None) is not None:
            x1, y1, x2, y2 = self.frame_roi_rect
            cv2.rectangle(color_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Update color frame
        height, width, channel = color_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(color_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale to fit label (fill entire space)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(), 
            Qt.IgnoreAspectRatio, 
            Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)
        
        # Update depth frame
        if depth_frame is not None:
            # Vẽ ROI lên depth frame nếu có
            if getattr(self, 'frame_roi_rect', None) is not None:
                x1, y1, x2, y2 = self.frame_roi_rect
                cv2.rectangle(depth_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            depth_height, depth_width, depth_channel = depth_frame.shape
            depth_bytes_per_line = 3 * depth_width
            depth_q_image = QImage(depth_frame.data, depth_width, depth_height, depth_bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # Scale to fit depth label (fill entire space)
            depth_pixmap = QPixmap.fromImage(depth_q_image)
            depth_scaled_pixmap = depth_pixmap.scaled(
                self.depth_label.size(), 
                Qt.IgnoreAspectRatio, 
                Qt.SmoothTransformation
            )
            self.depth_label.setPixmap(depth_scaled_pixmap)
        
        # Update frame count for FPS
        self.frame_count += 1
        
        # Update TOP objects count
        top_count = len([t for t in targets if t[0] is not None]) if targets else 0
        self.detection_count_label.setText(f"TOP Objects: {top_count}")
        
        # Hiển thị tọa độ vật
        if self.current_coordinates is not None:
            coord_text = f"X={self.current_coordinates[0]:.1f}mm, Y={self.current_coordinates[1]:.1f}mm, Z={self.current_coordinates[2]:.1f}mm"
            coord_text += f"\nRX={self.current_coordinates[3]:.1f}°, RY={self.current_coordinates[4]:.1f}°, RZ={self.current_coordinates[5]:.1f}°"
            self.grasp_pose_label.setText(coord_text)
            self.grasp_pose_label.setStyleSheet("color: #27ae60; font-size: 10px; font-weight: bold;")
        else:
            self.grasp_pose_label.setText("Chưa có tọa độ")
            self.grasp_pose_label.setStyleSheet("color: #e74c3c; font-size: 10px; font-weight: bold;")
    
    # Cập nhật tọa độ vật
    # Input: coordinates - tọa độ [X, Y, Z, RX, RY, RZ]
    # Output: không có
    def update_coordinates(self, coordinates):
        """Update current coordinates"""
        self.current_coordinates = coordinates.copy()
        # Xuất tọa độ ra file JSON
        self.export_coordinates_to_json(coordinates)
    
    # Xuất tọa độ ra file JSON
    # Input: coordinates - tọa độ [X, Y, Z, RX, RY, RZ]
    # Output: không có
    def export_coordinates_to_json(self, coordinates):
        """Export coordinates to JSON file"""
        try:
            # Tạo dữ liệu JSON với format "coords" array
            data = {
                "coords": [
                    round(coordinates[0], 1),
                    round(coordinates[1], 1), 
                    round(coordinates[2], 1),
                    round(coordinates[3], 1),
                    round(coordinates[4], 1),
                    round(coordinates[5], 1)
                ]
            }
            
            # Đường dẫn file JSON
            json_file_path = r"c:\Downloads\coordinates.json"
            
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
            
            # Ghi đè file JSON
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Log thông báo thành công
            self.log_message(f"Đã xuất tọa độ ra file: {json_file_path}")
            print(f"Đã xuất tọa độ ra file: {json_file_path}")  # In ra console để debug
            
        except Exception as e:
            error_msg = f"Lỗi xuất tọa độ ra JSON: {str(e)}"
            self.log_message(error_msg)
            print(error_msg)  # In ra console để debug
    
    # Cập nhật thông báo trạng thái hệ thống
    # Input: message - thông báo trạng thái
    # Output: không có
    def update_status(self, message):
        """Update status message"""
        # Xử lý cập nhật tọa độ
        if message.startswith("UPDATE_COORDINATES:"):
            try:
                # Lấy tọa độ từ message
                coords_str = message.replace("UPDATE_COORDINATES:", "")
                # Parse tọa độ từ string
                coords = eval(coords_str)  # Chuyển đổi string thành list
                self.update_coordinates(coords)
                return
            except Exception as e:
                self.log_message(f"Lỗi cập nhật tọa độ: {str(e)}")
                return
        
        self.status_label.setText(f"Trạng thái: {message}")
        self.log_message(message)
    
    # Hiển thị thông báo lỗi trong dialog box
    # Input: error_message - thông báo lỗi
    # Output: không có
    def show_error(self, error_message):
        """Show error message"""
        QMessageBox.critical(self, "Error", error_message)
        self.log_message(f"ERROR: {error_message}")
    
    # Thêm thông báo vào log console với timestamp
    # Input: message - thông báo cần log
    # Output: không có
    def log_message(self, message):
        """Add message to log console"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text_edit.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        scrollbar = self.log_text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    # Lưu cấu hình ROI vào file JSON
    # Input: không có
    # Output: không có
    def save_roi_config(self):
        """Lưu cấu hình ROI vào file"""
        try:
            if getattr(self, 'frame_roi_rect', None) is not None:
                roi_data = {"roi_rect": list(self.frame_roi_rect), "timestamp": time.time()}
                with open(self.roi_config_file, 'w') as f:
                    json.dump(roi_data, f)
                self.log_message("ROI configuration saved (frame coords)")
        except Exception as e:
            self.log_message(f"Failed to save ROI config: {str(e)}")
    
    # Tải cấu hình ROI từ file JSON
    # Input: không có
    # Output: không có
    def load_roi_config(self):
        """Tải cấu hình ROI từ file"""
        try:
            if os.path.exists(self.roi_config_file):
                with open(self.roi_config_file, 'r') as f:
                    roi_data = json.load(f)
                    if "roi_rect" in roi_data:
                        # Lưu ROI theo toạ độ frame và suy ra ROI cho label để hiển thị
                        self.frame_roi_rect = tuple(roi_data["roi_rect"])
                        # Suy ra ROI label theo kích thước hiện tại của label
                        label_size = self.camera_label.size()
                        label_w = max(1, label_size.width())
                        label_h = max(1, label_size.height())
                        frame_w = max(1, int(self.current_frame_width))
                        frame_h = max(1, int(self.current_frame_height))
                        inv_scale_x = label_w / frame_w
                        inv_scale_y = label_h / frame_h
                        lx1 = int(self.frame_roi_rect[0] * inv_scale_x)
                        ly1 = int(self.frame_roi_rect[1] * inv_scale_y)
                        lx2 = int(self.frame_roi_rect[2] * inv_scale_x)
                        ly2 = int(self.frame_roi_rect[3] * inv_scale_y)
                        self.camera_label.roi_rect = (lx1, ly1, lx2, ly2)
                        self.log_message("ROI configuration loaded")
        except Exception as e:
            self.log_message(f"Failed to load ROI config: {str(e)}")
    
    def save_roi_config(self):
        """Lưu cấu hình ROI vào file"""
        try:
            roi_data = {}
            if getattr(self, 'frame_roi_rect', None) is not None:
                roi_data["roi_rect"] = list(self.frame_roi_rect)
                with open(self.roi_config_file, 'w') as f:
                    json.dump(roi_data, f)
                self.log_message("ROI configuration saved (frame coords)")
        except Exception as e:
            self.log_message(f"Failed to save ROI config: {str(e)}")
    
    # Tự động mở camera khi khởi động để vẽ ROI và hiển thị depth
    # Input: không có
    # Output: không có
    def start_camera(self):
        """Tự động mở camera khi khởi động để vẽ ROI và hiển thị depth"""
        try:
            # Khởi tạo camera để vẽ ROI và hiển thị cả color và depth
            self.initialize_camera_for_display()
            self.log_message("Camera initialized for display")
        except Exception as e:
            self.log_message(f"Failed to initialize camera: {str(e)}")
    
    # Khởi tạo camera để hiển thị cả color và depth
    # Input: không có
    # Output: không có
    def initialize_camera_for_display(self):
        """Khởi tạo camera để hiển thị cả color và depth"""
        try:
            # Setup camera pipeline với cả color và depth
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipeline.start(config)
            align = rs.align(rs.stream.color)
            
            # Lấy frames để hiển thị
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame:
                # Convert color frame
                color_image = np.asanyarray(color_frame.get_data())
                # Cập nhật kích thước khung hình
                self.current_frame_height, self.current_frame_width = color_image.shape[:2]
                
                # Hiển thị color frame
                height, width, channel = color_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(color_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(
                    self.camera_label.size(), 
                    Qt.IgnoreAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.camera_label.setPixmap(scaled_pixmap)
            
            if depth_frame:
                # Convert depth frame
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Tạo depth visualization
                depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                
                # Hiển thị depth frame
                depth_height, depth_width, depth_channel = depth_colored.shape
                depth_bytes_per_line = 3 * depth_width
                depth_q_image = QImage(depth_colored.data, depth_width, depth_height, depth_bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                
                depth_pixmap = QPixmap.fromImage(depth_q_image)
                depth_scaled_pixmap = depth_pixmap.scaled(
                    self.depth_label.size(), 
                    Qt.IgnoreAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.depth_label.setPixmap(depth_scaled_pixmap)
                
                # Lưu pipeline để sử dụng sau
                self.roi_pipeline = pipeline
                self.camera_initialized = True
                
                # Load saved ROI nếu có
                if getattr(self, 'frame_roi_rect', None) is not None:
                    self.log_message("Loaded saved ROI")
                
            else:
                self.log_message("Failed to get camera frames")
                
        except Exception as e:
            self.log_message(f"Camera initialization error: {str(e)}")
            self.camera_initialized = False
    
    # Cập nhật hiển thị camera liên tục cho ROI drawing
    # Input: không có
    # Output: không có
    def update_camera_display(self):
        """Cập nhật hiển thị camera liên tục"""
        if hasattr(self, 'roi_pipeline') and self.roi_pipeline:
            try:
                # Lấy frames mới
                frames = self.roi_pipeline.wait_for_frames()
                align = rs.align(rs.stream.color)
                frames = align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if color_frame:
                    # Convert color frame
                    color_image = np.asanyarray(color_frame.get_data())
                    # Cập nhật kích thước khung hình
                    self.current_frame_height, self.current_frame_width = color_image.shape[:2]
                    
                    # Vẽ ROI lên color frame nếu có (dùng toạ độ frame)
                    if getattr(self, 'frame_roi_rect', None) is not None:
                        x1, y1, x2, y2 = self.frame_roi_rect
                        cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Hiển thị color frame
                    height, width, channel = color_image.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(color_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                    
                    pixmap = QPixmap.fromImage(q_image)
                    scaled_pixmap = pixmap.scaled(
                        self.camera_label.size(), 
                        Qt.IgnoreAspectRatio, 
                        Qt.SmoothTransformation
                    )
                    self.camera_label.setPixmap(scaled_pixmap)
                
                if depth_frame:
                    # Convert depth frame
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    # Tạo depth visualization
                    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                    
                    # Vẽ ROI lên depth frame nếu có (dùng toạ độ frame)
                    if getattr(self, 'frame_roi_rect', None) is not None:
                        x1, y1, x2, y2 = self.frame_roi_rect
                        cv2.rectangle(depth_colored, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Hiển thị depth frame
                    depth_height, depth_width, depth_channel = depth_colored.shape
                    depth_bytes_per_line = 3 * depth_width
                    depth_q_image = QImage(depth_colored.data, depth_width, depth_height, depth_bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                    
                    depth_pixmap = QPixmap.fromImage(depth_q_image)
                    depth_scaled_pixmap = depth_pixmap.scaled(
                        self.depth_label.size(), 
                        Qt.IgnoreAspectRatio, 
                        Qt.SmoothTransformation
                    )
                    self.depth_label.setPixmap(depth_scaled_pixmap)
                    
            except Exception as e:
                # Không log lỗi để tránh spam console
                pass

    # Cập nhật hiển thị FPS
    # Input: không có
    # Output: không có
    def update_fps(self):
        """Update FPS display"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            self.fps_display_label.setText(f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = current_time
    
    # Xử lý sự kiện đóng cửa sổ để dọn dẹp resources
    # Input: event - sự kiện đóng cửa sổ
    # Output: không có
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_detection()
        # Dọn dẹp ROI pipeline
        if hasattr(self, 'roi_pipeline') and self.roi_pipeline:
            self.roi_pipeline.stop()
        # Dừng camera timer
        if hasattr(self, 'camera_timer'):
            self.camera_timer.stop()
        event.accept()

# Hàm main để khởi chạy ứng dụng PyQt5
# Input: không có
# Output: không có
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
