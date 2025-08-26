import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from std_msgs.msg import String
from custom_interfaces.msg import ClusterData # 외부 인터페이스 패키지 사용

import numpy as np
from sklearn.cluster import DBSCAN
import math

# --- 상수 정의 ---
MAX_RANGE = 1.5
MIN_RANGE_THRESHOLD = 0.18
DBSCAN_EPS = 0.15
DBSCAN_MIN_SAMPLES = 5
MAX_CONE_DIAMETER = 0.3
FILTER_DEGREE = 36 # 각도 필터링을 위한 각도 구간 (36도)

def filter_front_clusters_by_angle(cluster_centers, degree_bin):
    def angle_bin(theta):
        return int(np.rad2deg(theta) // degree_bin)

    used_bins = set()
    front_cone_clusters = []

    # 1. 차량과의 거리가 가까운 순서대로 클러스터를 정렬합니다.
    for center in sorted(cluster_centers, key=lambda p: np.hypot(p[0], p[1])):
        x, y = center
        theta = math.atan2(y, x)
        bin_idx = angle_bin(theta)

        # 2. 해당 각도 구역(bin)이 아직 사용되지 않았다면, 이 클러스터를 선택합니다.
        if bin_idx not in used_bins:
            front_cone_clusters.append(center)
            # 3. 이 각도 구역을 '사용됨'으로 표시합니다.
            used_bins.add(bin_idx)
            # 가까운 순으로 정렬했기 때문에, 같은 각도 구역의 더 먼 클러스터는 무시됩니다.

    return front_cone_clusters

class PreprocessingNode(Node):
    def __init__(self):
        super().__init__('preprocessing_node')
        self.is_active = False
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan_rotated', self.lidar_callback, qos_profile_sensor_data
        )
        self.activation_sub = self.create_subscription(String, '/sign_color', self.activation_callback, 10)
        self.cluster_pub = self.create_publisher(ClusterData, '/clusters', 10)
        self.get_logger().info('전처리(클러스터링) 노드가 시작되었습니다.')

    def activation_callback(self, msg: String):
        if msg.data == 'green' and not self.is_active:
            self.get_logger().info('!!! Preprocessing activated by green signal !!!')
            self.is_active = True
    
    def lidar_callback(self, msg: LaserScan):
        if not self.is_active:
            return
        # 1. Lidar 데이터 좌표 변환
        scan_data = np.array(msg.ranges, dtype=np.float32)
        angle_start = msg.angle_min
        num_points = len(scan_data)
        angles = np.linspace(angle_start, angle_start + msg.angle_increment * (num_points - 1), num_points)

        x_all = scan_data * np.cos(angles)
        y_all = scan_data * np.sin(angles)

        # 2. 데이터 필터링
        front_mask = (angles >= -np.pi/2) & (angles <= np.pi/2)

        valid = np.isfinite(scan_data) & (scan_data <= MAX_RANGE) & (scan_data > MIN_RANGE_THRESHOLD) & front_mask
        if not np.any(valid):
            self.cluster_pub.publish(ClusterData()) # 데이터가 없으면 빈 메시지 발행
            return
        
        x, y = x_all[valid], y_all[valid]
        points = np.vstack((x, y)).T
        
        # 3. 클러스터링 (DBSCAN)
        if points.shape[0] < DBSCAN_MIN_SAMPLES:
            self.cluster_pub.publish(ClusterData())
            return
            
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points)
        labels = db.labels_

        # 4. 클러스터 대표점 계산 및 지름 필터링
        cluster_centers = []
        unique_labels = set(labels)
        for k in unique_labels:
            if k == -1: continue # 노이즈는 제외
            class_member_mask = (labels == k)
            cluster_points = points[class_member_mask]
            
            if len(cluster_points) > 1:
                min_x, max_x = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])
                min_y, max_y = np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1])
                diameter = np.hypot(max_x - min_x, max_y - min_y)
                if diameter > MAX_CONE_DIAMETER:
                    continue # 지름이 너무 크면 콘이 아닌 것으로 간주
            
            # 클러스터 내에서 차량에 가장 가까운 점을 대표점으로 선택
            dists = np.sqrt(cluster_points[:, 0]**2 + cluster_points[:, 1]**2)
            min_dist_idx = np.argmin(dists)
            center = tuple(cluster_points[min_dist_idx])
            cluster_centers.append(center)

        # 5. 각도 기반 필터링 적용
        filtered_centers = filter_front_clusters_by_angle(cluster_centers, degree_bin=FILTER_DEGREE)
        
        # 6. 필터링된 최종 결과를 발행
        cluster_msg = ClusterData()
        cluster_msg.clusters = [Point(x=c[0], y=c[1], z=0.0) for c in filtered_centers]
        self.cluster_pub.publish(cluster_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PreprocessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
