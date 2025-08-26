import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

import numpy as np
import math

WHEELBASE = 0.33  # [m]

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.is_active = False # 비활성 상태로 시작
        self.mode_sub = self.create_subscription(String, '/driving_mode', self.mode_callback, 10)
        self.activation_sub = self.create_subscription(String, '/sign_color', self.activation_callback, 10)
        self.path_sub = self.create_subscription(Path, '/path', self.path_callback, 10)
        self.motor_pub = self.create_publisher(Float32MultiArray, 'xycar_motor', 10)
        self.target_pub = self.create_publisher(PointStamped, '/target_point', 10)

    def activation_callback(self, msg: String):
        if msg.data == 'green' and self.is_active == False:
            self.get_logger().warn('!!! Motor control activated by green signal !!!')
            self.is_active = True

    def mode_callback(self, msg):
        if msg.data == "STANLEY":
            self.get_logger().info("pure_pursuit 비활성화, 곧 노드를 종료합니다.")
            self.is_active = False
            self.create_timer(3.0, self.shutdown_later, callback_group=None)  # 3초 후 종료

    def shutdown_later(self):
        self.get_logger().info("pure_pursuit 노드를 shutdown합니다.")
        self.destroy_node()

    def path_callback(self, msg: Path):

        if not self.is_active:
            # self.publish_motor_command(0.0, 0.0)
            return

        angle_cmd = 0.0
        speed = 10.0
        target_ra = None

        if len(msg.poses) > 1:  # 보간된 경로가 있을 때
            interp_x = np.array([p.pose.position.x for p in msg.poses])
            interp_y = np.array([p.pose.position.y for p in msg.poses])
            
            lookahead_dist = self.dynamic_lookahead_from_path(interp_x, interp_y)
            angle_cmd, target_ra = self.pure_pursuit_control(interp_x, interp_y, lookahead_dist)
            speed = self.compute_speed(angle_cmd)

        elif len(msg.poses) == 1:  # 단일 목표점이 있을 때
            target_ra = (msg.poses[0].pose.position.x, msg.poses[0].pose.position.y)
            angle_cmd = self.single_target_control(target_ra)
            speed = self.compute_speed(angle_cmd)
        
        self.publish_motor_command(angle_cmd, speed)
        
        if target_ra is not None:
            self.publish_target_point(target_ra)

    def single_target_control(self, target_ra):
        alpha = math.atan2(target_ra[1], target_ra[0])
        ld = np.hypot(target_ra[0], target_ra[1])
        if ld < 1e-6:
            return 0.0
            
        delta_rad = math.atan2(2.0 * WHEELBASE * math.sin(alpha), ld)
        angle_deg = -math.degrees(delta_rad)
        return float(np.clip(angle_deg / 0.2, -100, 100))

    def pure_pursuit_control(self, interp_x, interp_y, lookahead_dist):
        dists = np.sqrt(interp_x**2 + interp_y**2)
        candidates = np.where(dists > lookahead_dist)[0]
        
        if not candidates.any():
            tx, ty = interp_x[-1], interp_y[-1]
        else:
            best_idx = candidates[0]
            tx, ty = interp_x[best_idx], interp_y[best_idx]
        
        angle_cmd = self.single_target_control((tx, ty))
        return angle_cmd, (tx, ty)

    def dynamic_lookahead_from_path(self, interp_x, interp_y, scale=0.25, min_ld=0.7, max_ld=2.5):
        dx, dy = np.diff(interp_x), np.diff(interp_y)
        dists = np.sqrt(dx**2 + dy**2)
        total_length = np.sum(dists)
        return np.clip(scale * total_length, min_ld, max_ld)

    def compute_speed(self, angle_cmd):
        MAX_SPEED, MIN_SPEED = 20.0, 10.0
        angle_abs = abs(angle_cmd)
        norm = angle_abs / 100.0
        return float(MAX_SPEED - (MAX_SPEED - MIN_SPEED) * norm)

    def publish_motor_command(self, angle_cmd, speed):

        motor_msg = Float32MultiArray()
        motor_msg.data = [float(angle_cmd), float(speed)]
        self.motor_pub.publish(motor_msg)

    def publish_target_point(self, target_ra):
        target_msg = PointStamped()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = "rear_axle"
        target_msg.point.x = target_ra[0]
        target_msg.point.y = target_ra[1]
        self.target_pub.publish(target_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
