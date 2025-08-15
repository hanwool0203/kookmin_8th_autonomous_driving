import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped
from mission_cone_interfaces.msg import ClusterData, ConeData
from std_msgs.msg import Float32MultiArray

import numpy as np
import matplotlib.pyplot as plt
import threading
import math

# --- 상수 정의 ---
MAX_RANGE = 1.5
LIDAR_TO_REAR_AXLE = 0.42

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, qos_profile_sensor_data
        )
        self.cluster_sub = self.create_subscription(
            ClusterData, '/clusters', self.cluster_callback, 10
        )
        self.cone_sub = self.create_subscription(
            ConeData, '/cone_clusters', self.cone_callback, 10
        )
        self.path_sub = self.create_subscription(Path, '/path', self.path_callback, 10)
        self.target_sub = self.create_subscription(
            PointStamped, '/target_point', self.target_callback, 10
        )
        self.midpoint_sub = self.create_subscription(
            ClusterData, '/midpoints', self.midpoint_callback, 10
        )
        self.motor_sub = self.create_subscription(Float32MultiArray, 'xycar_motor', 10)
        
        self._lock = threading.Lock()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.setup_plot()

    def setup_plot(self):
        self.ax.set_xlim(0, MAX_RANGE) # 전방
        self.ax.set_ylim(-MAX_RANGE, MAX_RANGE) # 좌우
        self.ax.set_aspect("equal")
        self.lidar_pts, = self.ax.plot([], [], "bo", ms=2, label="Lidar Points")
        self.cluster_dots, = self.ax.plot([], [], 'rs', ms=5, label="Clusters")
        self.left_cone_dots, = self.ax.plot([], [], 'ys', ms=5, label="Left Cones")
        self.right_cone_dots, = self.ax.plot([], [], 'cs', ms=5, label="Right Cones")
        self.path_line, = self.ax.plot([], [], 'm--', lw=2, label="Path")
        self.target_dot, = self.ax.plot([], [], 'y*', ms=15, label="Target")
        # 새로 추가된 플롯 요소들
        self.midpoints_dots, = self.ax.plot([], [], 'o', color='orange', ms=6, label="Midpoints")
        self.steering_line, = self.ax.plot([], [], "r-", lw=2, label="Steering direction")
        self.angle_text = self.ax.text(0.05, 0.95, "", transform=self.ax.transAxes,
            fontsize=12, color='red', verticalalignment='top', horizontalalignment='left')
        
        # 각도 격자 추가
        for d in range(0, 361, 20):
            t = np.deg2rad(d) + np.pi/2
            self.ax.plot([0, MAX_RANGE*np.cos(t)], [0, MAX_RANGE*np.sin(t)], "k--", lw=0.5, alpha=0.3)
        self.ax.legend(loc="upper right")

    def midpoint_callback(self, msg: ClusterData):
        with self._lock:
            if msg.clusters:
                mx, my = zip(*[(p.x, p.y) for p in msg.clusters])
                self.midpoints_dots.set_data(mx, my)
            else:
                self.midpoints_dots.set_data([], [])

    def motor_callback(self, msg: Float32MultiArray):
        with self._lock:
            angle_cmd = msg.data[0]
            speed = msg.data[1]
            # 조향각 텍스트 업데이트
            self.angle_text.set_text(f"Angle Cmd: {angle_cmd:.2f}")
            # 조향선 업데이트
            original_angle_cmd = -msg.angle
            restored_rad = math.radians(original_angle_cmd * 0.2)
            steer_dx = 1.0 * math.cos(restored_rad)
            steer_dy = 1.0 * math.sin(restored_rad)
            self.steering_line.set_data([0, steer_dx], [0, steer_dy])
        
    def lidar_callback(self, msg: LaserScan):
        with self._lock:
            scan_data = np.array(msg.ranges[1:505], dtype=np.float32)
            angles = np.linspace(msg.angle_min + msg.angle_increment, msg.angle_min + msg.angle_increment * len(scan_data), len(scan_data))
            x = scan_data * np.cos(angles)
            y = scan_data * np.sin(angles)
            valid = np.isfinite(scan_data) & (scan_data <= MAX_RANGE)
            self.lidar_pts.set_data(x[valid], y[valid])
            self.fig.canvas.draw_idle()

    def cluster_callback(self, msg: ClusterData):
        with self._lock:
            if msg.clusters:
                cx, cy = zip(*[(p.x, p.y) for p in msg.clusters])
                self.cluster_dots.set_data(cx, cy)
            else:
                self.cluster_dots.set_data([], [])

    def cone_callback(self, msg: ConeData):
        with self._lock:
            if msg.left_cones: self.left_cone_dots.set_data(*zip(*[(p.x,p.y) for p in msg.left_cones]))
            else: self.left_cone_dots.set_data([], [])
            if msg.right_cones: self.right_cone_dots.set_data(*zip(*[(p.x,p.y) for p in msg.right_cones]))
            else: self.right_cone_dots.set_data([], [])

    def path_callback(self, msg: Path):
        with self._lock:
            if msg.poses:
                # 경로의 기준이 'rear_axle'이므로, 시각화를 위해 다시 'lidar' 기준으로 변환
                path_x = [p.pose.position.x - LIDAR_TO_REAR_AXLE for p in msg.poses]
                path_y = [p.pose.position.y for p in msg.poses]
                self.path_line.set_data(path_x, path_y)
            else:
                self.path_line.set_data([], [])

    def target_callback(self, msg: PointStamped):
        with self._lock:
            # 타겟의 기준이 'rear_axle'이므로, 'lidar' 기준으로 변환
            tx = msg.point.x - LIDAR_TO_REAR_AXLE
            ty = msg.point.y 
            self.target_dot.set_data(tx, ty)

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
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
