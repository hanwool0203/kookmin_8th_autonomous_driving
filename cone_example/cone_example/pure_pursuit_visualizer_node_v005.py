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
FILTER_DEGREE = 36
SEED_MAX_DISTANCE = 1.0

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
        super().__init__('pure_pursuit_visualizer_node')
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
        self.cluster_center_dots, = self.ax.plot([], [], 'rs', ms=5, label="All Clusters")
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
            # if len(cluster_points) < MIN_CLUSTER_SIZE: continue
            
            dists = np.sqrt(cluster_points[:, 0]**2 + cluster_points[:, 1]**2)
            min_dist_idx = np.argmin(dists)
            center_x, center_y = cluster_points[min_dist_idx, 0], cluster_points[min_dist_idx, 1]
            cluster_centers.append((center_x, center_y))

        # --- 6. 각도 기반 필터링 --- (각도 구역(bin)별로 가장 가까운 클러스터만 남겨 배경 사물을 제거)
        
        cluster_centers = filter_front_clusters_by_angle(cluster_centers, degree_bin=FILTER_DEGREE)

        if cluster_centers:
            # zip(*)을 사용하여 x좌표 리스트와 y좌표 리스트로 분리
            cx, cy = zip(*cluster_centers)
            self.cluster_center_dots.set_data(cx, cy)
        else:
            self.cluster_center_dots.set_data([], [])

        # --- 7. 좌우 클러스터 군집 형성 ---

        # 1. x좌표 기준으로 초기 후보군 생성
        initial_left_candidates = [pt for pt in cluster_centers if pt[0] < 0]
        initial_right_candidates = [pt for pt in cluster_centers if pt[0] > 0]

        # 2. 거리 필터링: 차량으로부터 SEED_MAX_DISTANCE 이내에 있는 후보만 최종 후보군으로 선택
        left_candidates = [
            pt for pt in initial_left_candidates
            if np.hypot(pt[0], pt[1]) < SEED_MAX_DISTANCE
        ]
        right_candidates = [
            pt for pt in initial_right_candidates
            if np.hypot(pt[0], pt[1]) < SEED_MAX_DISTANCE
        ]

        # 3. 필터링된 최종 후보군 내에서 가장 가까운 점을 '씨앗(seed)'으로 선택
        left_seed = min(left_candidates, key=lambda p: np.hypot(p[0], p[1]), default=None)
        right_seed = min(right_candidates, key=lambda p: np.hypot(p[0], p[1]), default=None)

        if left_seed:
            left_cones = self.grow_clusters([left_seed], cluster_centers)
        else:
            left_cones = []

        if right_seed:
            right_cones = self.grow_clusters([right_seed], cluster_centers)
        else:
            right_cones = []

    # --- 8, 9, 10 단계: 주행 경로 및 조향각 결정 (통합된 조건문) ---

        angle_cmd = 0.0
        interp_x, interp_y = None, None
        target = None
        midpoints = [] 

        # 조건 1: 정상 주행 (양쪽에 2개 이상의 콘이 있을 때)
        if len(left_cones) >= 2 and len(right_cones) >= 2:
            midpoints = self.calculate_midpoints(left_cones, right_cones)
            interp_x, interp_y = self.interpolate_path(midpoints)

        # 조건 2: 양쪽에 콘이 하나씩만 있을 때
        elif len(left_cones) == 1 and len(right_cones) == 1:
            l_cone = left_cones[0]
            r_cone = right_cones[0]
            mid_x = (l_cone[0] + r_cone[0]) / 2
            mid_y = (l_cone[1] + r_cone[1]) / 2
            
            target = (mid_x, mid_y) # 중앙점을 타겟으로 직접 설정
            angle_rad = math.atan2(mid_x, mid_y)
            angle_deg = math.degrees(angle_rad)
            angle_cmd = float(np.clip(angle_deg / 0.2, -100, 100))

        # --- 신규 추가된 조건 3: 비대칭 상황 처리 ---
        # 왼쪽은 1개, 오른쪽은 여러 개일 때
        elif len(left_cones) == 1 and len(right_cones) > 1:
            l_cone = left_cones[0]
            for r_cone in right_cones:
                mx, my = (l_cone[0] + r_cone[0]) / 2, (l_cone[1] + r_cone[1]) / 2
                midpoints.append((mx, my))
            interp_x, interp_y = self.interpolate_path(midpoints)
        
        # 오른쪽은 1개, 왼쪽은 여러 개일 때
        elif len(right_cones) == 1 and len(left_cones) > 1:
            r_cone = right_cones[0]
            for l_cone in left_cones:
                mx, my = (l_cone[0] + r_cone[0]) / 2, (l_cone[1] + r_cone[1]) / 2
                midpoints.append((mx, my))
            interp_x, interp_y = self.interpolate_path(midpoints)

        elif len(left_cones) == 0 and len(right_cones) > 0:
            # (-0.5, 0.1)에 가상의 왼쪽 콘을 생성
            virtual_left_cone = (-0.5, 0.1)
            
            # *** 바로 이 위치에 아래 한 줄을 추가합니다 ***
            left_cones.append(virtual_left_cone) # 시각화를 위해 left_cones 리스트에 추가
            # *** 추가 완료 ***

            # 이 가상 콘과 감지된 모든 오른쪽 콘을 연결하여 중간점 생성
            for r_cone in right_cones:
                mx, my = (virtual_left_cone[0] + r_cone[0]) / 2, (virtual_left_cone[1] + r_cone[1]) / 2
                midpoints.append((mx, my))
            
            # 생성된 중간점들로 경로 보간
            interp_x, interp_y = self.interpolate_path(midpoints)
        
        # --- 최종 조향각 계산 및 모터 제어 ---
        # 위 조건들에서 보간된 경로(interp_x)가 생성되었으면 Pure Pursuit 실행
        if interp_x is not None:
            lookahead_dist = self.dynamic_lookahead_from_path(interp_x, interp_y)
            angle_cmd, target = self.pure_pursuit(interp_x, interp_y, lookahead_dist)

        # Pure Pursuit 실행 (보간된 경로가 있을 경우에만)
        if interp_x is not None:
            self.interp_line.set_data(interp_x, interp_y)
            lookahead_dist = self.dynamic_lookahead_from_path(interp_x, interp_y)
            angle_cmd, target = self.pure_pursuit(interp_x, interp_y, lookahead_dist)
            self.pp_target_dot.set_data(*target)

        # 조건 3을 제외한 나머지 불확실한 상황에서는 interp_x가 None이므로 angle_cmd는 0을 유지
        elif not (len(left_cones) == 1 and len(right_cones) == 1):
             self.interp_line.set_data([], [])
             self.pp_target_dot.set_data([], [])

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
    
    def calculate_midpoints(self, left_cones, right_cones):
        """
        좌/우 콘 리스트로부터 중간점 리스트를 계산합니다.
        각 왼쪽 콘을 가장 가까운 오른쪽 콘 및 그 다음 콘과 연결하여
        더 많은 중간점을 생성합니다.
        """
        midpoints = []
        
        # 경로가 연속되도록 y 좌표(전방 거리) 기준으로 정렬
        left_sorted = sorted(left_cones, key=lambda p: p[1])
        right_sorted = sorted(right_cones, key=lambda p: p[1])
        
        if not left_sorted or not right_sorted:
            return []

        for l_cone in left_sorted:
            # 1. 각 왼쪽 콘에 대해 y좌표가 가장 가까운 오른쪽 콘 찾기
            candidates = [(abs(l_cone[1] - r_cone[1]), j, r_cone) for j, r_cone in enumerate(right_sorted)]
            if not candidates:
                continue
            
            _, best_j, best_r_cone = min(candidates)

            # 2. 가장 가까운 콘과 중간점 생성
            mx1, my1 = (l_cone[0] + best_r_cone[0]) / 2, (l_cone[1] + best_r_cone[1]) / 2
            midpoints.append((mx1, my1))

            # 3. 그 다음(+1) 오른쪽 콘과도 중간점 생성 (존재하는 경우)
            if best_j + 1 < len(right_sorted):
                next_r_cone = right_sorted[best_j + 1]
                mx2, my2 = (l_cone[0] + next_r_cone[0]) / 2, (l_cone[1] + next_r_cone[1]) / 2
                midpoints.append((mx2, my2))
        
        return midpoints

    
    def interpolate_path(self, midpoints):
        if not midpoints or len(midpoints) < 2:
            if self.prev_interp_x is not None:
                return self.prev_interp_x, self.prev_interp_y
            return None, None

        midpoints = sorted(midpoints, key=lambda p: p[1])
        mxs = np.array([p[0] for p in midpoints])
        mys = np.array([p[1] for p in midpoints])
        
        mys, unique_idx = np.unique(mys, return_index=True)
        mxs = mxs[unique_idx]

        if len(mxs) < 2:
            if self.prev_interp_x is not None:
                return self.prev_interp_x, self.prev_interp_y
            return None, None

        spline_fn = CubicSpline(mys, mxs)
        interp_y = np.linspace(mys.min(), mys.max(), 100)
        interp_x = spline_fn(interp_y)
        
        self.prev_interp_x, self.prev_interp_y = interp_x, interp_y
        return interp_x, interp_y

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
        MAX_SPEED, MIN_SPEED = 20.0,20.0
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