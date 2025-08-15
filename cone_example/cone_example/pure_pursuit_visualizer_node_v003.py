import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import XycarMotor
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import threading
from sklearn.cluster import DBSCAN

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
        super().__init__('pure_pursuit_visualizer_node_v003')
        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, qos_profile_sensor_data
        )
        self._lock = threading.Lock()
        self.prev_interp_x = None
        self.prev_interp_y = None
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.setup_plot()

    def setup_plot(self):
        """Matplotlib 플롯의 초기 설정을 수행합니다."""
        self.ax.set_xlim(-MAX_RANGE, MAX_RANGE)
        self.ax.set_ylim(0, MAX_RANGE)
        self.ax.set_aspect("equal")

        self.lidar_pts, = self.ax.plot([], [], "bo", ms=5)
        self.left_cone_dots, = self.ax.plot([], [], 'ys', ms=5, label="Left Cluster")
        self.right_cone_dots, = self.ax.plot([], [], 'cs', ms=5, label="Right Cluster")
        self.interp_line, = self.ax.plot([], [], 'm--', lw=2, label="Interpolated Midline")
        self.pp_target_dot, = self.ax.plot([], [], 'y*', ms=15, label="Pursuit Target")
        self.steering_line, = self.ax.plot([], [], "r-", lw=2, label="Steering direction")
        self.angle_text = self.ax.text(0.05, 0.95, "", transform=self.ax.transAxes,
            fontsize=10, color='red', verticalalignment='top', horizontalalignment='left'
        )

        self.ax.legend(loc="upper right")
        for d in range(0, 361, 20):
            t = np.deg2rad(d) + np.pi/2
            self.ax.plot([0, MAX_RANGE*np.cos(t)], [0, MAX_RANGE*np.sin(t)], "--", lw=0.5)

    def lidar_callback(self, msg: LaserScan):
        with self._lock:
            self.process_and_visualize(msg)

    def process_and_visualize(self, msg: LaserScan):
        
        # --- 1. 초기화 ---
        motor_msg = XycarMotor() # 모터에 전달할 메시지 객체 생성

        # --- 2. Lidar 데이터 수신 및 좌표 변환 ---

        scan_data = np.array(msg.ranges[1:505], dtype=np.float32)
        angle_start = msg.angle_min + msg.angle_increment
        num_points = len(scan_data)
        angles = np.linspace(angle_start, angle_start + msg.angle_increment * (num_points - 1), num_points)
        x_all = scan_data * -np.sin(angles)
        y_all = scan_data * np.cos(angles)

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
        self.lidar_pts.set_data(x, y)
        
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

        left_candidates = [pt for pt in cluster_centers if pt[0] < 0]
        right_candidates = [pt for pt in cluster_centers if pt[0] > 0]

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

        left_cones = sorted(left_cones, key=lambda t: t[1])
        right_cones = sorted(right_cones, key=lambda t: t[1])
            
        midpoints = []

        used_ridx = set()
        for lpt in left_cones:
            candidates = [(abs(lpt[1]-rpt[1]), idx, rpt) for idx, rpt in enumerate(right_cones) if idx not in used_ridx]
            if not candidates: continue
            _, ridx, best_rpt = min(candidates)
            used_ridx.add(ridx)
            mx, my = (lpt[0] + best_rpt[0]) / 2, (lpt[1] + best_rpt[1]) / 2
            midpoints.append((mx, my))

        # --- 9. 스플라인 보간 (경로 부드럽게 만들기) ---

        if midpoints:
            midpoints = sorted(midpoints, key=lambda p: p[1])
            mxs = np.array([p[0] for p in midpoints])
            mys = np.array([p[1] for p in midpoints])
            mys, unique_idx = np.unique(mys, return_index=True)
            mxs = mxs[unique_idx]

            if len(mxs) >= 2:
                spline_fn = CubicSpline(mys, mxs)
                interp_y = np.linspace(mys.min(), mys.max(), 100)
                interp_x = spline_fn(interp_y)
                self.prev_interp_x, self.prev_interp_y = interp_x, interp_y
            elif self.prev_interp_x is not None:
                interp_x, interp_y = self.prev_interp_x, self.prev_interp_y
            else:
                interp_x, interp_y = None, None
        else:
            interp_x, interp_y = None, None

        # --- 10. 조향각 계산 (Pure Pursuit) ---

        if interp_x is None:
            self.interp_line.set_data([], [])
            angle_cmd = 0
        else:
            self.interp_line.set_data(interp_x, interp_y)
            lookahead_dist = self.dynamic_lookahead_from_path(interp_x, interp_y)
            angle_cmd, target = self.pure_pursuit(interp_x, interp_y, lookahead_dist)
            self.pp_target_dot.set_data(*target)

        # --- 11. 모터 제어 명령 전송 ---

        motor_msg.speed = self.compute_speed(angle_cmd)
        motor_msg.angle = float(angle_cmd)
        self.motor_pub.publish(motor_msg)

        # --- 12. 시각화 업데이트 ---

        if left_cones: self.left_cone_dots.set_data(*zip(*left_cones))
        else: self.left_cone_dots.set_data([], [])
        
        if right_cones: self.right_cone_dots.set_data(*zip(*right_cones))
        else: self.right_cone_dots.set_data([], [])

        restored_rad = math.radians(angle_cmd * 0.2)
        steer_dx, steer_dy = 5.0 * math.sin(restored_rad), 5.0 * math.cos(restored_rad)
        self.steering_line.set_data([0, steer_dx], [0, steer_dy])
        self.angle_text.set_text(f"angle_cmd: {angle_cmd}")
        
        self.fig.canvas.draw_idle()

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
        angle_rad = math.atan2(tx, ty)
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
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        plt.show()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join()

if __name__ == '__main__':
    main()