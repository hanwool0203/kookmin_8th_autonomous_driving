import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32,String
import numpy as np
import cv2
import yaml
import os
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory
import message_filters

# YOLO 노드에서 발행하는 기존 커스텀 메시지만 임포트합니다.
from cam_interfaces.msg import Detections

class YoloLidarFusionNode(Node):
    def __init__(self):
        super().__init__('yolo_lidar_fusion_node')

        # --- 센서 데이터 저장을 위한 변수 ---
        self.last_scan_points_3d = None
        self.last_detections = None
        
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

        self.bridge = CvBridge()

        # --- 3개의 토픽을 구독 (변경 없음) ---
        self.image_subscription = message_filters.Subscriber(self, Image, '/image_raw')
        self.scan_subscription = message_filters.Subscriber(self, LaserScan, '/scan',qos_profile=qos_profile_sensor_data)
        self.detection_subscription = message_filters.Subscriber(self, Detections, '/yolo_detections')

        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_subscription, self.scan_subscription, self.detection_subscription],
            queue_size=10,
            slop=0.1
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)
        
        # --- 시각화된 이미지를 발행할 Publisher만 남김 ---
        
        self.distance_pub = self.create_publisher(Float32, '/obstacle_distance', 10)
        self.position_pub = self.create_publisher(String, '/obstacle_position', 10)
        self.get_logger().info("YOLO-LiDAR Fusion Node has been started.")

    def synchronized_callback(self, image_msg, scan_msg, detection_msg):
        #self.get_logger().info('<<<<< SYNCHRONIZED CALLBACK IS RUNNING! >>>>>') # 콜백 호출 디버깅
        self.image_callback(image_msg)
        self.scan_callback(scan_msg)
        self.detections_callback(detection_msg)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        mask = (ranges > msg.range_min) & (ranges < 3.0)
        x = ranges[mask] * np.cos(angles[mask])
        y = ranges[mask] * np.sin(angles[mask])
        self.last_scan_points_3d = np.stack([x, y, np.zeros_like(x)], axis=1)

    def detections_callback(self, msg):
        self.last_detections = msg

    def image_callback(self, msg):
        if self.last_scan_points_3d is None or self.last_detections is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return
            
        # 메인 로직 함수 호출
        fused_image = self.fuse_and_visualize(
            cv_image.copy(), 
            self.last_detections, 
            self.last_scan_points_3d
        )

        cv2.imshow("YOLO-LiDAR Fusion", fused_image)
        cv2.waitKey(1)
        
        # 시각화된 이미지를 토픽으로 발행
        
    # --- [MODIFIED] 평균 거리 계산 및 시각화 전용 함수 ---
    def fuse_and_visualize(self, image, detections_msg, points_3d):
        h, w, _ = image.shape
        vehicle_detected = False
        
        if points_3d.shape[0] == 0:
            return image

        # 1. 3D 라이다 포인트를 2D 이미지 좌표로 투영
        cam_pts = (self.R @ points_3d.T + self.t).T
        img_pts, _ = cv2.projectPoints(cam_pts, np.zeros(3), np.zeros(3), self.mtx, None)
        img_pts = img_pts.squeeze(axis=1)

        for det in detections_msg.detections:
            # "obstacle_vehicle" 클래스만 대상으로 함
            if det.class_name == "obstacle_vehicle":
                vehicle_detected = True
                xmin, ymin, xmax, ymax = det.xmin, det.ymin, det.xmax, det.ymax
                # [NEW] Bbox 중심 좌표 계산 및 위치 판별 로직
                center_x = (xmin + xmax) / 2
                position_str = ""
                if center_x < 4.5*w / 10:
                    position_str = "left"
                elif center_x < 5.5 * w / 10:
                    position_str = "center"
                else:
                    position_str = "right"
                
                # [NEW] 위치 정보를 String 메시지로 발행
                position_msg = String()
                position_msg.data = position_str
                self.position_pub.publish(position_msg)

                # BBox 그리기
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, f"Pos: {position_str}", (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                # BBox 내부에 있는 3D 포인트들의 인덱스를 찾음
                indices_in_bbox = np.where(
                    (img_pts[:, 0] >= xmin) & (img_pts[:, 0] < xmax) &
                    (img_pts[:, 1] >= ymin) & (img_pts[:, 1] < ymax)
                )[0]

                if len(indices_in_bbox) > 0:
                    # 해당 인덱스의 3D 포인트들을 추출
                    points_3d_in_bbox = points_3d[indices_in_bbox]
                    
                    # 거리(range) 계산 및 평균값 산출
                    ranges_in_bbox = np.sqrt(points_3d_in_bbox[:, 0]**2 + points_3d_in_bbox[:, 1]**2)
                    avg_distance = np.mean(ranges_in_bbox)

                    distance_msg = Float32()
                    distance_msg.data = float(avg_distance)
                    self.distance_pub.publish(distance_msg)

                    # 시각화: BBox 내부의 포인트들 그리기
                    for pt_idx in indices_in_bbox:
                        px, py = int(img_pts[pt_idx, 0]), int(img_pts[pt_idx, 1])
                        cv2.circle(image, (px, py), 2, (0, 0, 255), -1)

                    # 시각화: 평균 거리 텍스트 추가
                    text = f"Dist: {avg_distance:.2f}m"
                    cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if not vehicle_detected:
            distance_msg = Float32()
            distance_msg.data = 999.0  # Large value to indicate no obstacle
            self.distance_pub.publish(distance_msg)

            position_msg = String()
            position_msg.data = "none"
            self.position_pub.publish(position_msg)
            
        return image

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = YoloLidarFusionNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"An error occurred in YoloLidarFusionNode: {e}")
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