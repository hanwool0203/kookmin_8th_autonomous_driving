import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import XycarMotor
from scipy.interpolate import CubicSpline
from sklearn.cluster import DBSCAN

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration

# --- 상수 정의 ---
MAX_RANGE = 1.5
MIN_CLUSTER_SIZE = 3
DBSCAN_EPS = 0.15
DBSCAN_MIN_SAMPLES = 2
MIN_RANGE_THRESHOLD = 0.15

def filter_front_clusters_by_angle(cluster_centers, degree_bin):

    def angle_bin(theta):
        return int(np.rad2deg(theta) // degree_bin)

    used_bins = set()
    front_cone_clusters = []

    for center in sorted(cluster_centers, key=lambda p: np.hypot(p[0], p[1])):
        x, y = center
        theta = math.atan2(x, y)  # 차량 기준 각도(라디안)
        bin_idx = angle_bin(theta)
        if bin_idx not in used_bins:
            front_cone_clusters.append(center)
            used_bins.add(bin_idx)

    return front_cone_clusters

class PurePursuitVisualizerNode(Node):
    """
    Lidar 데이터를 구독하여 경로 추종 알고리즘을 수행하고,
    처리 과정을 Matplotlib으로 실시간 시각화하는 ROS2 노드.
    """
    def __init__(self):
        super().__init__('move2rviz')
        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, qos_profile_sensor_data
        )
        self.prev_interp_x = None
        self.prev_interp_y = None

        # 클러스터들을 시각화하기 위한 MarkerArray 퍼블리셔
        self.cones_pub = self.create_publisher(MarkerArray, 'cone_clusters', 10)
        # 경로(midline)를 시각화하기 위한 Marker 퍼블리셔
        self.path_pub = self.create_publisher(Marker, 'interpolated_path', 10)
        # Pure Pursuit 타겟 포인트를 시각화하기 위한 Marker 퍼블리셔
        self.target_pub = self.create_publisher(Marker, 'pursuit_target', 10)

    def lidar_callback(self, msg: LaserScan):
        self.process_and_visualize(msg)

    def process_and_visualize(self, msg: LaserScan):
        
        # --- 1. 초기화 ---
        motor_msg = XycarMotor() # 모터에 전달할 메시지 객체 생성

        # --- 2. Lidar 데이터 수신 및 좌표 변환 ---

        scan_data = np.array(msg.ranges[1:505], dtype=np.float32)
        angle_start = msg.angle_min + msg.angle_increment
        num_points = len(scan_data)
        angles = np.linspace(angle_start, angle_start + msg.angle_increment * (num_points - 1), num_points)
        x_all = scan_data * np.cos(angles)
        y_all = scan_data * np.sin(angles)

        # --- 3. 데이터 필터링 (전처리) ---

        valid = np.isfinite(scan_data) & (scan_data <= MAX_RANGE) & (scan_data > MIN_RANGE_THRESHOLD)
        if np.sum(valid) == 0:
            self.lidar_pts.set_data([], [])
            motor_msg.speed = 0.0
            motor_msg.angle = 0.0
            self.motor_pub.publish(motor_msg)
            self.fig.canvas.draw_idle()
            return
        
        x, y = x_all[valid], y_all[valid]
        
        # --- 4. 클러스터링 (DBSCAN) ---

        points = np.vstack((x, y)).T
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='euclidean').fit(points)
        labels = db.labels_

        # --- 5. 클러스터 대표점 계산 ---

        cluster_centers = []
        unique_labels = set(labels)
        for k in unique_labels:
            if k == -1: continue
            class_member_mask = (labels == k)
            cluster_points = points[class_member_mask]
            if len(cluster_points) < MIN_CLUSTER_SIZE: continue
            
            dists = np.sqrt(cluster_points[:, 0]**2 + cluster_points[:, 1]**2)
            min_dist_idx = np.argmin(dists)
            center_x, center_y = cluster_points[min_dist_idx, 0], cluster_points[min_dist_idx, 1]
            cluster_centers.append((center_x, center_y))

        # --- 6. 각도 기반 필터링 --- (각도 구역(bin)별로 가장 가까운 클러스터만 남겨 배경 사물을 제거)
        
        cluster_centers = filter_front_clusters_by_angle(cluster_centers, degree_bin=36)

        # --- 7. 좌우 클러스터 군집 형성 ---

        left_candidates = [pt for pt in cluster_centers if pt[1] > 0]
        right_candidates = [pt for pt in cluster_centers if pt[1] < 0]

        # 각 후보군에서 차량에 가장 가까운 점을 '씨앗(seed)'으로 선택 (유클리드 거리 기준)

        left_seed = min(left_candidates, key=lambda p: np.hypot(p[0], p[1]), default=None)
        right_seed = min(right_candidates, key=lambda p: np.hypot(p[0], p[1]), default=None)

        # 씨앗으로부터 가까운 클러스터들을 연결하여 최종 라인을 '성장'시킴

        if left_seed:
            left_cones = self.grow_clusters([left_seed], cluster_centers)
        else:
            left_cones = []

        if right_seed:
            right_cones = self.grow_clusters([right_seed], cluster_centers)
        else:
            right_cones = []

        # --- 8. 주행 경로 생성 ---

        left_cones = sorted(left_cones, key=lambda t: t[0])
        right_cones = sorted(right_cones, key=lambda t: t[0])
            
        midpoints = []

        used_ridx = set()
        for lpt in left_cones:
            candidates = [(abs(lpt[0]-rpt[0]), idx, rpt) for idx, rpt in enumerate(right_cones) if idx not in used_ridx]
            if not candidates: continue
            _, ridx, best_rpt = min(candidates)
            used_ridx.add(ridx)
            mx, my = (lpt[0] + best_rpt[0]) / 2, (lpt[1] + best_rpt[1]) / 2
            midpoints.append((mx, my))

        # --- 9. 스플라인 보간 (경로 부드럽게 만들기) ---

        if midpoints:
            midpoints = sorted(midpoints, key=lambda p: p[0])
            mxs = np.array([p[0] for p in midpoints])
            mys = np.array([p[1] for p in midpoints])
            mxs, unique_idx = np.unique(mxs, return_index=True)
            mys = mys[unique_idx]

            if len(mxs) >= 2:
                spline_fn = CubicSpline(mxs, mys)
                interp_x = np.linspace(mxs.min(), mxs.max(), 100)
                interp_y = spline_fn(interp_x)
                self.prev_interp_x, self.prev_interp_y = interp_x, interp_y
            elif self.prev_interp_x is not None:
                interp_x, interp_y = self.prev_interp_x, self.prev_interp_y
            else:
                interp_x, interp_y = None, None
        else:
            interp_x, interp_y = None, None

        # --- 10. 조향각 계산 (Pure Pursuit) ---

        if interp_x is None:
            angle_cmd = 0
        else:
            lookahead_dist = self.dynamic_lookahead_from_path(interp_x, interp_y)
            angle_cmd, target = self.pure_pursuit(interp_x, interp_y, lookahead_dist)

        # --- 11. 모터 제어 명령 전송 ---

        motor_msg.speed = self.compute_speed(angle_cmd)
        motor_msg.angle = float(angle_cmd)
        self.motor_pub.publish(motor_msg)

        # --- 12. 시각화 업데이트 (RViz2) ---
        # Matplotlib 관련 코드를 모두 삭제하고 아래 코드로 대체합니다.
        
        # 좌/우 콘 클러스터를 하나의 MarkerArray 메시지로 만들어 발행
        self.publish_cone_markers(left_cones, right_cones, msg.header)
        
        # 생성된 경로(Path)를 Marker 메시지로 만들어 발행
        if interp_x is not None:
            self.publish_path_marker(interp_x, interp_y, msg.header)
        
        # Pure Pursuit 타겟 포인트를 Marker 메시지로 만들어 발행
        if 'target' in locals() and target is not None:
            self.publish_target_marker(target, msg.header)

        # PurePursuitVisualizerNode 클래스 내부에 아래 함수들을 추가합니다.

    def publish_cone_markers(self, left_cones, right_cones, header):
        """좌/우 콘 클러스터를 MarkerArray로 만들어 발행합니다."""
        marker_array = MarkerArray()
        
        # 왼쪽 콘 마커
        for i, cone in enumerate(left_cones):
            marker = Marker()
            marker.header = header
            marker.ns = "left_cones"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = cone[0]
            marker.pose.position.y = cone[1]
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = (0.2, 0.2, 0.2)
            marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0) # Yellow
            marker.lifetime = Duration(sec=0, nanosec=200000000) # 0.2초
            marker_array.markers.append(marker)
            
        # 오른쪽 콘 마커
        for i, cone in enumerate(right_cones):
            marker = Marker()
            marker.header = header
            marker.ns = "right_cones"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = cone[0]
            marker.pose.position.y = cone[1]
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = (0.2, 0.2, 0.2)
            marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0) # Cyan
            marker.lifetime = Duration(sec=0, nanosec=200000000) # 0.2초
            marker_array.markers.append(marker)
            
        self.cones_pub.publish(marker_array)

    def publish_path_marker(self, path_x, path_y, header):
        """보간된 경로를 LINE_STRIP 타입의 Marker로 만들어 발행합니다."""
        marker = Marker()
        marker.header = header
        marker.ns = "path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # 라인 두께
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0) # Magenta
        marker.lifetime = Duration(sec=0, nanosec=200000000)
        
        for i in range(len(path_x)):
            p = Point()
            p.x = path_x[i]
            p.y = path_y[i]
            marker.points.append(p)
            
        self.path_pub.publish(marker)

    def publish_target_marker(self, target, header):
        """Pure Pursuit 타겟 포인트를 SPHERE 타입의 Marker로 만들어 발행합니다."""
        marker = Marker()
        marker.header = header
        marker.ns = "target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = target[0]
        marker.pose.position.y = target[1]
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = (0.3, 0.3, 0.3)
        marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0) # Yellow
        marker.lifetime = Duration(sec=0, nanosec=200000000)
        
        self.target_pub.publish(marker)



    def grow_clusters(self, seed_list, all_clusters, threshold=0.5):
        grown = seed_list.copy()
        queue = seed_list.copy()
        visited = set(tuple(p) for p in seed_list)
        while queue:
            base = queue.pop(0)
            bx, by = base
            for pt in all_clusters:
                pt_tuple = tuple(pt)
                if pt_tuple in visited: continue
                px, py = pt
                dist = np.hypot(px - bx, py - by)
                if dist <= threshold:
                    grown.append(pt)
                    queue.append(pt)
                    visited.add(pt_tuple)
        return grown

    def pure_pursuit(self, interp_x, interp_y, lookahead_dist):
        dists = np.sqrt(interp_x**2 + interp_y**2)
        candidates = np.where(dists > lookahead_dist)[0]
        if len(candidates) == 0:
            tx, ty = interp_x[-1], interp_y[-1]
        else:
            best_idx = candidates[0]
            for idx in candidates:
                if interp_y[idx] > 0:
                    best_idx = idx
                    break
            tx, ty = interp_x[best_idx], interp_y[best_idx]
        angle_rad = math.atan2(ty, tx)
        angle_deg = math.degrees(angle_rad)
        angle_cmd = float(np.clip(angle_deg / 0.2, -100, 100))
        return angle_cmd, (tx, ty)

    def dynamic_lookahead_from_path(self, interp_x, interp_y, scale=0.3, min_ld=0.5, max_ld=5.0):
        dx, dy = np.diff(interp_x), np.diff(interp_y)
        dists = np.sqrt(dx**2 + dy**2)
        total_length = np.sum(dists)
        return np.clip(scale * total_length, min_ld, max_ld)

    def compute_speed(self, angle_cmd):
        MAX_SPEED, MIN_SPEED = 25.0, 15.0
        angle_abs = abs(angle_cmd)
        norm = angle_abs / 100.0
        return float(MAX_SPEED - (MAX_SPEED - MIN_SPEED) * norm)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()