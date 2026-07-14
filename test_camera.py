import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import time
from ultralytics import YOLO

# Import functions from app_removeRobot to use the exact same logic
from app_removeRobot import (
    CameraInfo, 
    PoseEstimation,
    build_point_cloud_from_detection
)

def estimate_pose_with_pca_stable(point_cloud, mask_2d, depth_m, camera_info, bbox_center) -> PoseEstimation:
    # 2. TÍNH GÓC QUAY X, Y VÀ TÂM TỪ MASK 2D (Chống lỗi Point Cloud bị móp méo do mất depth)
    contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    (cx_mask, cy_mask), (width, height), angle = rect
    box = cv2.boxPoints(rect)
    
    # 1. TÍNH TÂM 3D SIÊU CHUẨN TỪ TÂM MASK VÀ MEDIAN DEPTH
    # Giải quyết triệt để lỗi RealSense bị mất chiều sâu ở 1 nửa phôi
    valid_depths = depth_m[(mask_2d > 0) & (depth_m > 0.1) & (depth_m < 1.0)]
    if len(valid_depths) < 10: return None
    
    # Lấy trung vị (median) của độ sâu
    cz3d = float(np.median(valid_depths))
    cx3d = (cx_mask - camera_info.cx) * cz3d / camera_info.fx
    cy3d = -(cy_mask - camera_info.cy) * cz3d / camera_info.fy # [FIX] OPEN3D Y HƯỚNG LÊN!
    center = np.array([cx3d, cy3d, cz3d])
    
    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]
    if np.linalg.norm(edge1) > np.linalg.norm(edge2): dir_x_2d = edge1 / np.linalg.norm(edge1)
    else: dir_x_2d = edge2 / np.linalg.norm(edge2)
        
    # 3. KẾT HỢP GÓC 2D VÀ PHÁP TUYẾN 3D (Cố định Z vuông góc camera)
    # Trái với PCA bị nhiễu làm trục xoay lung tung, ta ép trục Z của phôi
    # luôn song song với trục Z của camera (vuông góc với mặt phẳng ảnh)
    normal = np.array([0.0, 0.0, -1.0]) # Z hướng về camera
    
    # Trục X lấy chuẩn từ ảnh 2D
    dir_x_3d = np.array([dir_x_2d[0], -dir_x_2d[1], 0.0]) 
    
    # Đảm bảo trục X luôn hướng về nửa phải của camera
    if dir_x_3d[0] < 0:
        dir_x_3d = -dir_x_3d
    
    # Trục Y = Z cross X
    dir_y_3d = np.cross(normal, dir_x_3d)
    
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[:, 0] = dir_x_3d
    rotation_matrix[:, 1] = dir_y_3d
    rotation_matrix[:, 2] = normal
    
    # Chuẩn hoá hệ toạ độ thuận tay phải
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 1] = -rotation_matrix[:, 1]

    return PoseEstimation(center=center, rotation_matrix=rotation_matrix, axes=rotation_matrix)

def main():
    print("[INFO] Khởi tạo Camera RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure resolution just like the app
    width, height, fps = 640, 480, 30
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    profile = pipeline.start(config)
    
    # Căn chỉnh (Align) depth frame theo color frame
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # Lấy thông số Camera
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    camera_info = CameraInfo(
        width=intrinsics.width,
        height=intrinsics.height,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.ppx,
        cy=intrinsics.ppy,
        scale=1.0 / depth_scale
    )
    
    print("[INFO] Thông số Camera:", camera_info)
    
    # Load YOLO Model
    weights_path = 'C:/Users/DELL/Downloads/BinPicking_old/BinPicking/best.pt'
    print(f"[INFO] Load YOLO Model từ {weights_path}...")
    try:
        model = YOLO(weights_path)
        print("[INFO] YOLO Model load thành công!")
    except Exception as e:
        print(f"[ERROR] Không thể load model: {e}")
        return

    print("[INFO] Bắt đầu lấy frame và quét vật (Nhấn 'q' để thoát)...")
    
    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Predict
            results = model.predict(source=color_image, conf=0.5, iou=0.45, verbose=False)
            
            display_img = color_image.copy()
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                result = results[0]
                boxes = result.boxes
                masks = result.masks
                
                if masks is not None:
                    # Parse results
                    bxyxy = boxes.xyxy.cpu().numpy()
                    bconf = boxes.conf.cpu().numpy()
                    bcls = boxes.cls.cpu().numpy().astype(int)
                    
                    # Convert depth to meters
                    depth_m = depth_image.astype(np.float32) * (1.0 / camera_info.scale)
                    color_rgb = color_image[:, :, ::-1].astype(np.float32) / 255.0
                    
                    for i in range(len(bxyxy)):
                        x1, y1, x2, y2 = map(int, bxyxy[i])
                        cls = int(bcls[i])
                        conf = float(bconf[i])
                        
                        # Chỉ check object class top (class 0 hoặc 1 thường là object top)
                        if cls not in [0, 1]:  
                            continue
                            
                        # Tâm của bounding box 2D
                        bbox_center_x = int((x1 + x2) / 2)
                        bbox_center_y = int((y1 + y2) / 2)
                        
                        # [FIX CỰC KỲ QUAN TRỌNG] 
                        # YOLO masks.data trả về kích thước bị thu nhỏ (thường là 160x160) và có viền đen (padding)
                        # Nếu resize trực tiếp lên 640x480 sẽ bị méo và lệch mask so với thực tế!
                        # Cách chuẩn xác nhất là dùng masks.xy (toạ độ viền mask đã được YOLO scale chuẩn về ảnh gốc)
                        polygon = masks.xy[i]
                        mk = np.zeros(depth_m.shape[:2], dtype=np.uint8)
                        if len(polygon) > 0:
                            cv2.fillPoly(mk, [polygon.astype(np.int32)], 1)
                            
                        # Dùng thuật toán Erode (xói mòn) để thu nhỏ mask lại một chút
                        # Việc này giúp loại bỏ các điểm ảnh thuộc mặt sàn bị lem vào mask của YOLO
                        kernel = np.ones((5,5), np.uint8)
                        mask_for_pcl = cv2.erode(mk, kernel, iterations=2)
                        
                        # Tính PointCloud & Pose
                        detection_pcd = build_point_cloud_from_detection(
                            color_rgb, depth_m, mask_for_pcl, camera_info,
                            min_distance_m=0.1, max_distance_m=1.0
                        )
                        
                        pose = None
                        if detection_pcd is not None and len(detection_pcd.points) > 10:
                            bbox_center = (bbox_center_x, bbox_center_y)
                            pose = estimate_pose_with_pca_stable(detection_pcd, mask_for_pcl, depth_m, camera_info, bbox_center)
                        
                        # Vẽ Bounding Box
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        
                        # Vẽ Mask Overlay (phủ lớp màu khít với phôi)
                        overlay = display_img.copy()
                        mask_color = (0, 0, 255) if i == 0 else (144, 238, 144) # Object 0 màu đỏ, còn lại màu xanh
                        if np.any(mk > 0):
                            overlay[mk > 0] = (overlay[mk > 0] * 0.6 + np.array(mask_color) * 0.4).astype(np.uint8)
                        display_img = overlay
                        
                        # Vẽ tâm Bbox (Màu Vàng)
                        cv2.circle(display_img, (bbox_center_x, bbox_center_y), 5, (0, 255, 255), -1)
                        
                        if pose is not None:
                            # Tọa độ 3D Tâm Phôi
                            cx3d, cy3d, cz3d = pose.center
                            
                            # Chiếu Tâm 3D về ảnh 2D để hiển thị
                            cam_center = np.array([cx3d, -cy3d, cz3d])
                            pca_center_2d = None
                            if cam_center[2] != 0:
                                pca_center_2d = (
                                    int(cam_center[0] * camera_info.fx / cam_center[2] + camera_info.cx),
                                    int(cam_center[1] * camera_info.fy / cam_center[2] + camera_info.cy)
                                )
                                # Vẽ Tâm PCA (Màu Đỏ)
                                cv2.circle(display_img, pca_center_2d, 5, (0, 0, 255), -1)
                                
                                # Vẽ Trục Tọa Độ từ Pose PCA
                                axis_length = 0.05
                                axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # BGR
                                for j in range(3):
                                    end_point_3d_o3d = pose.center + pose.axes[:, j] * axis_length
                                    end_point_3d = np.array([end_point_3d_o3d[0], -end_point_3d_o3d[1], end_point_3d_o3d[2]])
                                    if end_point_3d[2] != 0:
                                        end_point_2d = (
                                            int(end_point_3d[0] * camera_info.fx / end_point_3d[2] + camera_info.cx),
                                            int(end_point_3d[1] * camera_info.fy / end_point_3d[2] + camera_info.cy)
                                        )
                                        cv2.line(display_img, pca_center_2d, end_point_2d, axis_colors[j], 3)
                            
                            # Print thông số ra Terminal
                            print(f"\n[DETECT] Object {i} - Class {cls} - Conf: {conf:.2f}")
                            print(f"  -> BBox Center (2D) : ({bbox_center_x}, {bbox_center_y})")
                            if pca_center_2d is not None:
                                print(f"  -> PCA Center (2D)  : ({pca_center_2d[0]}, {pca_center_2d[1]})")
                                dx = pca_center_2d[0] - bbox_center_x
                                dy = pca_center_2d[1] - bbox_center_y
                                print(f"  -> Độ lệch tâm (px) : dX={dx}, dY={dy}")
                            print(f"  -> Tọa độ 3D (Camera) : X={cx3d:.4f}, Y={cy3d:.4f}, Z={cz3d:.4f}")
                        else:
                            print(f"\n[DETECT] Object {i} - Không tính được Pose (ít điểm PCL).")

            # Hiển thị
            cv2.imshow("Camera Test - Center Detection", display_img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
                
            frame_count += 1
            if frame_count % 30 == 0:
                print(".", end="", flush=True)
                
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n[INFO] Đã đóng camera.")

if __name__ == "__main__":
    main()
