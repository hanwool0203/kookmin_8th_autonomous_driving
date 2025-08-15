import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
import numpy as np
import math

class ScanRotatorNode(Node):
    def __init__(self):
        super().__init__('scan_rotator_node')
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, qos_profile_sensor_data
        )
        self.rotated_scan_pub = self.create_publisher(LaserScan, '/scan_rotated', 10)
        self.get_logger().info('스캔 회전 노드가 시작되었습니다.')

    def lidar_callback(self, msg: LaserScan):
        # --- 핵심 로직: 각도 보정 ---
        
        # 1. 원본 LaserScan 데이터를 데카르트 좌표(x, y)로 변환합니다.
        scan_data = np.array(msg.ranges, dtype=np.float32)
        intensities_data = np.array(msg.intensities, dtype=np.float32) if msg.intensities else np.array([])
        angle_start = msg.angle_min
        num_points = len(scan_data)
        angles = np.linspace(angle_start, angle_start + msg.angle_increment * (num_points - 1), num_points)

        xs = scan_data * np.cos(angles)
        ys = scan_data * np.sin(angles)

        # 2. 좌표를 -120도 회전시킵니다.
        theta = np.deg2rad(-120.0)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        x_rotated = xs * cos_t - ys * sin_t
        y_rotated = xs * sin_t + ys * cos_t

        # 3. 회전된 좌표를 다시 LaserScan 메시지 형태로 변환하여 발행합니다.
        self.publish_rotated_scan(msg, x_rotated, y_rotated,intensities_data)

    def publish_rotated_scan(self, original_msg, x_all, y_all, intensities_all):
        """회전된 데카르트 좌표를 강도(intensity)와 함께 LaserScan 메시지로 변환하여 발행"""
        rotated_msg = LaserScan()
        rotated_msg.header = original_msg.header
        rotated_msg.header.frame_id = 'laser_frame'

        # 데카르트 -> 폴라 좌표로 변환
        rotated_ranges = np.hypot(x_all, y_all)
        rotated_angles = np.arctan2(y_all, x_all)
        
        num_points = len(original_msg.ranges)
        new_ranges = np.full(num_points, np.inf, dtype=np.float32)
        
        # 강도 배열도 동일한 크기로 초기화
        new_intensities = np.zeros(num_points, dtype=np.float32)
        has_intensities = len(intensities_all) == num_points

        # 회전된 각도가 원본 각도 슬롯의 어디에 해당하는지 계산
        indices = ((rotated_angles - original_msg.angle_min) / original_msg.angle_increment).astype(int)
        
        for i in range(len(indices)):
            idx = indices[i]
            if 0 <= idx < num_points:
                # 같은 각도 슬롯에 여러 점이 매핑될 경우, 더 가까운 점으로 갱신
                if rotated_ranges[i] < new_ranges[idx]:
                    new_ranges[idx] = rotated_ranges[i]
                    # 해당 거리의 강도 값도 함께 갱신
                    if has_intensities:
                        new_intensities[idx] = intensities_all[i]
        
        # 최종 메시지 설정
        rotated_msg.angle_min = original_msg.angle_min
        rotated_msg.angle_max = original_msg.angle_max
        rotated_msg.angle_increment = original_msg.angle_increment
        rotated_msg.time_increment = original_msg.time_increment
        rotated_msg.scan_time = original_msg.scan_time
        rotated_msg.range_min = original_msg.range_min
        rotated_msg.range_max = original_msg.range_max
        rotated_msg.ranges = new_ranges.tolist()
        
        # 강도 데이터가 있었을 경우에만 채워넣기
        if has_intensities:
            rotated_msg.intensities = new_intensities.tolist()
        
        self.rotated_scan_pub.publish(rotated_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanRotatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
