import json
import time
from fairino import Robot

def get_robot_flange_pose():
    """
    Lấy tọa độ hiện tại của end flange robot và ghi vào file poserobot.json
    """
    try:

        
        # Kết nối robot với IP mặc định
        robot = Robot.RPC("192.168.58.2")  # IP mặc định của robot
        
        # Lấy tọa độ end flange hiện tại
        errorcode, flange_pose = robot.GetActualToolFlangePose(flag=1)
        
        if errorcode == 0:  # Thành công
            # Tạo dữ liệu theo định dạng yêu cầu
            robot_data = {
                "robot_pose": [
                    errorcode,
                    flange_pose
                ]
            }
            
            # Ghi vào file poserobot.json
            with open("C:\\Downloads\\poserobot.json", "w", encoding="utf-8") as f:
                json.dump(robot_data, f, indent=2, ensure_ascii=False)
            
            print(f"Đã ghi tọa độ robot vào C:\\Downloads\\poserobot.json")
            print(f"Tọa độ: {flange_pose}")
            return True
            
        else:
            print(f"Lỗi khi lấy tọa độ robot. Error code: {errorcode}")
            return False
            
    except Exception as e:
        print(f"Lỗi: {e}")
        return False
    
    finally:
        # Đóng kết nối robot
        try:
            robot.CloseRPC()
        except:
            pass

def read_robot_pose():
    """
    Đọc tọa độ robot từ file poserobot.json
    """
    try:
        with open("C:\\Downloads\\poserobot.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("File C:\\Downloads\\poserobot.json không tồn tại")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None

if __name__ == "__main__":
    # Lấy và ghi tọa độ robot
    success = get_robot_flange_pose()
    
    if success:
        # Đọc và hiển thị dữ liệu vừa ghi
        data = read_robot_pose()
        if data:
            print("\nDữ liệu trong file C:\\Downloads\\poserobot.json:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
