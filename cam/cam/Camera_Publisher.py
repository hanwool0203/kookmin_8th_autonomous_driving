#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import subprocess

def set_manual_camera_parameter(
    device='/dev/video0',
    exposure=50,
    gain=0,
    contrast=32,
    gamma=150,
    white_balance_temperature=3500
):
    try:
        # 1. 수동 노출 관련 설정
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'auto_exposure=1'], check=True)
        subprocess.run(['v4l2-ctl', '-d', device, '-c', f'exposure_time_absolute={exposure}'], check=True)

        subprocess.run(['v4l2-ctl', '-d', device, '-c', f'gain={gain}'], check=True)
        subprocess.run(['v4l2-ctl', '-d', device, '-c', f'contrast={contrast}'], check=True)

        # 2. 감마 수동 설정
        subprocess.run(['v4l2-ctl', '-d', device, '-c', f'gamma={gamma}'], check=True)

        # 3. 수동 화이트 밸런스 설정
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'white_balance_automatic=0'], check=True)
        subprocess.run(['v4l2-ctl', '-d', device, '-c', f'white_balance_temperature={white_balance_temperature}'], check=True)

        print(f'✅ Manual settings applied:')
        print(f'   exposure={exposure}, gain={gain}, contrast={contrast}, gamma={gamma}, white_balance_temperature={white_balance_temperature}')
    except subprocess.CalledProcessError as e:
        print(f'❌ Failed to set manual camera parameters: {e}')

class Camera_Publisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.bridge = CvBridge()

        # ------------------------------- 시각화 설정 -------------------------------

        self.show = True # 시각화 여부

        # ------------------------------- 시각화 설정 종료 -------------------------------

        # ROS2 이미지 퍼블리셔 생성
        self.publisher = self.create_publisher(Image, '/image_raw', 10)

        # OpenCV로 카메라 열기
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("❌ Failed to open /dev/video0")
            return

        # ✅ 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ✅ 왜곡 보정용 내부 파라미터 설정
        self.img_size = (640, 480)
        self.mtx = np.array([[329.7890410051319, 0.000000, 320.6057545861696],[0.000000, 329.76662478183584, 229.13605526487663],[0.000000, 0.000000, 1.000000]])
        self.dist = np.array([-0.00845590148493767, -0.020724735072096084, 0.007699567026492894, 0.004404855849468008])
        self.cal_mtx = self.mtx
        self.new_mtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.mtx, self.dist, self.img_size, np.eye(3), balance=0)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.mtx, self.dist, np.eye(3), self.new_mtx, self.img_size, cv2.CV_32FC1)

        # ✅ OpenCV 창 설정
        # cv2.namedWindow("Camera_Publisher - undistorted /image_raw", cv2.WINDOW_NORMAL)
        # cv2.moveWindow("Camera_Publisher - undistorted /image_raw", 0, 0)

        # ✅ 30fps 기준 타이머 설정
        self.timer = self.create_timer(0.033, self.timer_callback)

        self.get_logger().info("🚀 FastCameraPublisher running...")
    
    ### ===== 어안 렌즈 이미지의 왜곡을 보정한다 =====
    def to_calibrated(self, img):
        # 어안 렌즈 이미지 왜곡 보정
        # cv2.remap 함수와 initUndistortRectifyMap에서 계산된 맵을 사용하여 왜곡 보정 수행
        img_undistorted = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return img_undistorted

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("⚠️ Failed to read frame from camera.")
            return

        # 이미지 ROS 메시지로 변환 후 퍼블리시
        frame = self.to_calibrated(frame)
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        # 1. 헤더에 현재 시간과 좌표계 정보 채우기
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        self.publisher.publish(msg)

        # 시각화
        if self.show:
            # cv2.imshow("Camera_Publisher - undistorted /image_raw", frame)
            if cv2.waitKey(1) == 27:  # ESC 키로 종료
                self.get_logger().info("🛑 ESC pressed. Shutting down...")
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    # ✅ 카메라 수동 노출/감도 설정
    set_manual_camera_parameter('/dev/video0')

    node = Camera_Publisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("🛑 KeyboardInterrupt received.")
    finally:
        if hasattr(node, 'cap'):
            node.cap.release()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
