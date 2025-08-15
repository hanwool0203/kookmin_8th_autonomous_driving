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

# --- ?? ?? ---
WHEELBASE = 0.33  # [m] ??? ?? ????
LIDAR_TO_REAR_AXLE = 0.42 # [m] Lidar? ??? ? ??? ??
MAX_RANGE = 1.5
MIN_CLUSTER_SIZE = 3
DBSCAN_EPS = 0.15
DBSCAN_MIN_SAMPLES = 2
MIN_RANGE_THRESHOLD = 0.15
FILTER_DEGREE = 36
SEED_MAX_DISTANCE = 1.0
MAX_CONE_DIAMETER = 0.3 # ??? ??? ?? ?? (m)

def filter_front_clusters_by_angle(cluster_centers, degree_bin):

    def angle_bin(theta):
        return int(np.rad2deg(theta) // degree_bin)

    used_bins = set()
    front_cone_clusters = []

    for center in sorted(cluster_centers, key=lambda p: np.hypot(p[0], p[1])):
        x, y = center
        theta = math.atan2(x, y)  # ?? ?? ??(???)
        bin_idx = angle_bin(theta)
        if bin_idx not in used_bins:
            front_cone_clusters.append(center)
            used_bins.add(bin_idx)

    return front_cone_clusters

class PurePursuitVisualizerNode(Node):

    def __init__(self):
        super().__init__('pure_pursuit_visualizer_node_v007')
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
        self.ax.set_xlim(-MAX_RANGE, MAX_RANGE)
        self.ax.set_ylim(0, MAX_RANGE)
        self.ax.set_aspect("equal")

        self.lidar_pts, = self.ax.plot([], [], "bo", ms=5)
        self.cluster_center_dots, = self.ax.plot([], [], 'rs', ms=5, label="All Clusters")
        self.left_cone_dots, = self.ax.plot([], [], 'ys', ms=5, label="Left Cluster")
        self.right_cone_dots, = self.ax.plot([], [], 'cs', ms=5, label="Right Cluster")
        self.midpoints_dots, = self.ax.plot([], [], 'o', color='orange', ms=6, label="Midpoints")
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
        
        # --- 1. ??? ---
        motor_msg = XycarMotor()

        scan_data = np.array(msg.ranges[1:505], dtype=np.float32)
        angle_start = msg.angle_min + msg.angle_increment
        num_points = len(scan_data)
        angles = np.linspace(angle_start, angle_start + msg.angle_increment * (num_points - 1), num_points)
        x_all = scan_data * -np.sin(angles)
        y_all = scan_data * np.cos(angles)


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
        
        # --- 4. ????? (DBSCAN) ---

        points = np.vstack((x, y)).T
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='euclidean').fit(points)
        labels = db.labels_

        # --- 5. ???? ??? ?? ---
        cluster_centers = []
        unique_labels = set(labels)
        for k in unique_labels:
            if k == -1: continue
            class_member_mask = (labels == k)
            cluster_points = points[class_member_mask]
            
            # --- ???? ?? ??? ?? ---
            if len(cluster_points) > 1:
                min_x, max_x = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])
                min_y, max_y = np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1])
                diameter = np.hypot(max_x - min_x, max_y - min_y)
                
                # ????? ??? ?? ?? ?? ?? ??? ???? ??
                if diameter > MAX_CONE_DIAMETER:
                    continue

                
            dists = np.sqrt(cluster_points[:, 0]**2 + cluster_points[:, 1]**2)
            min_dist_idx = np.argmin(dists)
            center_x, center_y = cluster_points[min_dist_idx, 0], cluster_points[min_dist_idx, 1]
            cluster_centers.append((center_x, center_y))


        # --- 6. ?? ?? ??? --- (?? ??(bin)?? ?? ??? ????? ?? ?? ??? ??)
        
        cluster_centers = filter_front_clusters_by_angle(cluster_centers, degree_bin=FILTER_DEGREE)

        if cluster_centers:
            # zip(*)? ???? x?? ???? y?? ???? ??
            cx, cy = zip(*cluster_centers)
            self.cluster_center_dots.set_data(cx, cy)
        else:
            self.cluster_center_dots.set_data([], [])

        # --- 7. ?? ???? ?? ?? ---

        # 1. x?? ???? ?? ??? ??
        initial_left_candidates = [pt for pt in cluster_centers if pt[0] < 0]
        initial_right_candidates = [pt for pt in cluster_centers if pt[0] > 0]

        # 2. ?? ???: ?????? SEED_MAX_DISTANCE ??? ?? ??? ?? ????? ??
        left_candidates = [
            pt for pt in initial_left_candidates
            if np.hypot(pt[0], pt[1]) < SEED_MAX_DISTANCE
        ]
        right_candidates = [
            pt for pt in initial_right_candidates
            if np.hypot(pt[0], pt[1]) < SEED_MAX_DISTANCE
        ]

        # 3. ???? ?? ??? ??? ?? ??? ?? '??(seed)'?? ??
        left_seed = min(left_candidates, key=lambda p: np.hypot(p[0], p[1]), default=None)
        right_seed = min(right_candidates, key=lambda p: np.hypot(p[0], p[1]), default=None)

        used_clusters = set()

        if right_seed:
            right_cones = self.grow_clusters([right_seed], cluster_centers, used_clusters)
        else:
            right_cones = []
        if left_seed:
            left_cones = self.grow_clusters([left_seed], cluster_centers, used_clusters)
        else:
            left_cones = []

        # --- 8, 9, 10 ??: ?? ?? ? ??? ?? (???? ?? ??) ---

        # 1. ?? ???
        angle_cmd = 0.0
        midpoints_lidar = []      # Lidar ?? ???
        target_lidar = None       # Lidar ?? ?? ??

        # 2. ?? ?? ? Lidar ?? ??/?? ??
        if len(left_cones) >= 2 and len(right_cones) >= 2:
            midpoints_lidar = self.calculate_midpoints(left_cones, right_cones)
        elif len(left_cones) == 1 and len(right_cones) > 1:
            l_cone = left_cones[0]
            for r_cone in right_cones:
                midpoints_lidar.append(((l_cone[0] + r_cone[0]) / 2, (l_cone[1] + r_cone[1]) / 2))
        elif len(right_cones) == 1 and len(left_cones) > 1:
            r_cone = right_cones[0]
            for l_cone in left_cones:
                midpoints_lidar.append(((l_cone[0] + r_cone[0]) / 2, (l_cone[1] + r_cone[1]) / 2))
        elif len(left_cones) == 0 and len(right_cones) > 0:
            virtual_left_cone = (-0.5, 0.1)
            left_cones.append(virtual_left_cone)
            for r_cone in right_cones:
                midpoints_lidar.append(((virtual_left_cone[0] + r_cone[0]) / 2, (virtual_left_cone[1] + r_cone[1]) / 2))
        elif len(left_cones) == 1 and len(right_cones) == 1:
            target_lidar = ((left_cones[0][0] + right_cones[0][0]) / 2, (left_cones[0][1] + right_cones[0][1]) / 2)

        # 3. ??? ? ?? ???? ?? ? ??? ??
        interp_x_ra, interp_y_ra = None, None
        target_ra = None

        if midpoints_lidar:
            # ????? ??? ? ???? ??
            rear_axle_midpoints = [(p[0], p[1] + LIDAR_TO_REAR_AXLE) for p in midpoints_lidar]
            interp_x_ra, interp_y_ra = self.interpolate_path(rear_axle_midpoints)
        elif target_lidar is not None:
            # ?? ??? ??? ? ???? ??
            target_ra = (target_lidar[0], target_lidar[1] + LIDAR_TO_REAR_AXLE)
        
        # 4. ?? ??? ??
        if interp_x_ra is not None:
            lookahead_dist = self.dynamic_lookahead_from_path(interp_x_ra, interp_y_ra)
            angle_cmd, target_ra = self.pure_pursuit(interp_x_ra, interp_y_ra, lookahead_dist)
        elif target_ra is not None:
            alpha = math.atan2(target_ra[0], target_ra[1])
            ld = np.hypot(target_ra[0], target_ra[1])
            delta_rad = math.atan2(2.0 * WHEELBASE * math.sin(alpha), ld)
            angle_deg = math.degrees(delta_rad)
            angle_cmd = float(np.clip(angle_deg / 0.2, -100, 100))

        # --- 11. ?? ?? ?? ?? ---

        motor_msg.speed = self.compute_speed(angle_cmd)
        motor_msg.angle = float(angle_cmd)
        self.motor_pub.publish(motor_msg)

        # --- 12. ??? ???? ---

        if left_cones: self.left_cone_dots.set_data(*zip(*left_cones))
        else: self.left_cone_dots.set_data([], [])
        
        if right_cones: self.right_cone_dots.set_data(*zip(*right_cones))
        else: self.right_cone_dots.set_data([], [])

        if midpoints_lidar:
            mx, my = zip(*midpoints_lidar)
            self.midpoints_dots.set_data(mx, my)
        else:
            self.midpoints_dots.set_data([], [])

        # ?? ???
        if interp_x_ra is not None:
            interp_x_lidar = interp_x_ra
            interp_y_lidar = [y - LIDAR_TO_REAR_AXLE for y in interp_y_ra]
            self.interp_line.set_data(interp_x_lidar, interp_y_lidar)
        else:
            self.interp_line.set_data([], [])

        # ?? ??? ???
        if target_ra is not None:
            target_x_lidar = target_ra[0]
            target_y_lidar = target_ra[1] - LIDAR_TO_REAR_AXLE
            self.pp_target_dot.set_data(target_x_lidar, target_y_lidar)
        elif target_lidar is not None:
            self.pp_target_dot.set_data(*target_lidar)
        else:
            self.pp_target_dot.set_data([], [])


        restored_rad = math.radians(angle_cmd * 0.2)
        steer_dx, steer_dy = 5.0 * math.sin(restored_rad), 5.0 * math.cos(restored_rad)
        self.steering_line.set_data([0, steer_dx], [0, steer_dy])
        self.angle_text.set_text(f"angle_cmd: {angle_cmd}")
        
        self.fig.canvas.draw_idle()

    def grow_clusters(self, seed_list, all_clusters, used_clusters_set, threshold=0.5):
        # ??? ?? ?? ??? ??????, ?? ??? ???? ??
        if tuple(seed_list[0]) in used_clusters_set:
            return []

        grown = []
        queue = []
        
        # ??? ?? ???? ?? ??
        initial_seed_tuple = tuple(seed_list[0])
        grown.append(seed_list[0])
        queue.append(seed_list[0])
        used_clusters_set.add(initial_seed_tuple)

        head = 0
        while head < len(queue):
            base = queue[head]
            head += 1
            
            # ?? ????? ???? ?? ??
            for pt in all_clusters:
                pt_tuple = tuple(pt)
                # ?? ??? ????? ???
                if pt_tuple in used_clusters_set:
                    continue

                dist = np.hypot(pt[0] - base[0], pt[1] - base[1])

                if dist <= threshold:
                    grown.append(pt)
                    queue.append(pt)
                    # ?? ?? ??? ??? ??
                    used_clusters_set.add(pt_tuple)
        return grown

    
    def calculate_midpoints(self, left_cones, right_cones):
        midpoints = []
        
        # ??? ? ??? ?? ?? (?? ??? ?? ?? ??)
        MAX_VALID_DISTANCE = 1.3 

        left_sorted = sorted(left_cones, key=lambda p: p[1])
        right_sorted = sorted(right_cones, key=lambda p: p[1])
        
        if not left_sorted or not right_sorted:
            return []

        for l_cone in left_sorted:
            # y??? ?? ??? ??? ? ?? (?? ??)
            candidates = [(abs(l_cone[1] - r_cone[1]), r_cone) for r_cone in right_sorted]
            if not candidates: continue
            
            _, best_r_cone = min(candidates)

            # --- ?? ??? ?? ?? ---
            # ?? ????? ?? ??? ??
            dist = np.hypot(l_cone[0] - best_r_cone[0], l_cone[1] - best_r_cone[1])
            
            # ??? ?? ??, ? ??? ???? ?? ?? ??? ???
            if dist > MAX_VALID_DISTANCE:
                continue
            # --- ?? ?? ---

            # ??? ??? ??? ??? ???? ??
            mx, my = (l_cone[0] + best_r_cone[0]) / 2, (l_cone[1] + best_r_cone[1]) / 2
            midpoints.append((mx, my))
        
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
        
        alpha = math.atan2(tx,ty)
        ld = np.hypot(tx, ty)
        delta_rad = math.atan2(2.0 * WHEELBASE * math.sin(alpha), ld)
        angle_deg = math.degrees(delta_rad)
        angle_cmd = float(np.clip(angle_deg / 0.2, -100, 100))
        
        return angle_cmd, (tx, ty)


    def dynamic_lookahead_from_path(self, interp_x, interp_y, scale=0.25, min_ld=0.7, max_ld=2.5):
        dx, dy = np.diff(interp_x), np.diff(interp_y)
        dists = np.sqrt(dx**2 + dy**2)
        total_length = np.sum(dists)
        return np.clip(scale * total_length, min_ld, max_ld)

    def compute_speed(self, angle_cmd):
        MAX_SPEED, MIN_SPEED = 45.0,45.0
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
