import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import yaml
import os
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory

# 직접 생성한 커스텀 메시지를 임포트합니다.
from projection_interfaces.msg import PixelCoords, Point2D
from std_msgs.msg import Header


class Lidar2CamProjectorNode(Node):
    """
    LiDAR 포인트 클라우드를 카메라 이미지 평면에 투영하고,
    결과 픽셀 좌표를 '/scan_pixels' 토픽으로 발행하는 노드.
    (카메라 파라미터를 YAML 파일에서 로드)
    """
    def __init__(self):
        super().__init__('lidar2cam_projector_node')

        # --- 센서 데이터 저장을 위한 변수 ---
        self.last_scan_points_3d = None
        
        # --- [MODIFIED] 캘리브레이션 파라미터를 YAML 파일에서 로드 ---
        pkg_share = get_package_share_directory('lidar2cam_projector')
        intrinsic_file = os.path.join(pkg_share, 'config', 'fisheye_calib.yaml')
        extrinsic_file = os.path.join(pkg_share, 'config', 'extrinsic.yaml')

        self.get_logger().info(f"Loading intrinsic calibration from: {intrinsic_file}")
        self.get_logger().info(f"Loading extrinsic calibration from: {extrinsic_file}")

        with open(intrinsic_file, 'r') as f:
            intrinsic_calib = yaml.safe_load(f)
        
        ## [MODIFIED] 새로운 YAML 형식에 맞게 파싱 ##
        self.img_size = (intrinsic_calib['image_width'], intrinsic_calib['image_height'])
        self.mtx = np.array(intrinsic_calib['K'])   

        # 2. 외부 파라미터 (Extrinsic) 로드 (변경 없음)
        with open(extrinsic_file, 'r') as f:
            extrinsic_calib = yaml.safe_load(f)
        self.R = np.array(extrinsic_calib['R'])
        self.t = np.array(extrinsic_calib['t']).reshape((3, 1))
        
        # # 3. 로드된 파라미터로 왜곡 보정 맵 계산
        # self.new_mtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.mtx, self.dist, self.img_size, np.eye(3), balance=0)
        # self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.mtx, self.dist, np.eye(3), self.new_mtx, self.img_size, cv2.CV_32FC1)
        # --- 수정 완료 ---

        self.bridge = CvBridge()
        # --- LiDAR와 카메라 이미지만 구독 ---
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        
        # --- 투영된 픽셀 좌표와 시각화 이미지를 발행할 Publisher ---
        self.pixel_pub = self.create_publisher(PixelCoords, '/scan_pixels', 10)
        self.projected_img_pub = self.create_publisher(Image, '/projected_image', 10)
        
        self.get_logger().info("LiDAR to Camera Projector Node has been started.")

    def scan_callback(self, msg):
        """LaserScan 데이터를 3D 포인트 클라우드로 변환하여 저장합니다."""
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        mask = (ranges > msg.range_min) & (ranges < 3.0)
        x = ranges[mask] * np.cos(angles[mask])
        y = ranges[mask] * np.sin(angles[mask])
        self.last_scan_points_3d = np.stack([x, y, np.zeros_like(x)], axis=1)

    def image_callback(self, msg):
        """
        이미지 프레임이 들어올 때마다 저장된 LiDAR 포인트를 투영하고 결과를 발행합니다.
        """
        if self.last_scan_points_3d is None:
            self.get_logger().warn("Waiting for LiDAR data...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # calibrated_img = self.to_calibrated(cv_image)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return
            
        projected_image, pixel_coords_msg = self.project_points_and_create_msg(
            cv_image.copy(), 
            self.last_scan_points_3d,
            msg.header
        )

        self.pixel_pub.publish(pixel_coords_msg)

        try:
            self.projected_img_pub.publish(self.bridge.cv2_to_imgmsg(projected_image, "bgr8"))
            cv2.imshow("LiDAR Points on Image", projected_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to publish or show projected image: {e}")

    def project_points_and_create_msg(self, image, points_3d, header):
        """
        3D LiDAR 포인트를 2D 이미지 좌표로 투영하고,
        결과를 시각화하며 발행할 메시지를 생성합니다.
        """
        h, w, _ = image.shape
        pixel_coords_msg = PixelCoords()
        pixel_coords_msg.header = header
        
        if points_3d.shape[0] == 0:
            return image, pixel_coords_msg

        cam_pts = (self.R @ points_3d.T + self.t).T
        img_pts, _ = cv2.projectPoints(cam_pts, np.zeros(3), np.zeros(3), self.mtx, None)
        img_pts = img_pts.squeeze(axis=1)

        for pt in img_pts:
            px, py = int(pt[0]), int(pt[1])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(image, (px, py), 2, (0, 0, 255), -1)
                point2d = Point2D()
                point2d.x = px
                point2d.y = py
                pixel_coords_msg.pixels.append(point2d)
                
        return image, pixel_coords_msg

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Lidar2CamProjectorNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"An error occurred in Lidar2CamProjectorNode: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node:
            cv2.destroyAllWindows()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()