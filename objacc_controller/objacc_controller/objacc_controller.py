import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Float32
import numpy as np

class ObstacleFollowerNode(Node):
    def __init__(self):
        super().__init__('obstacle_follower_node')

        # --- 튜닝 파라미터 ---
        self.TARGET_TIME_GAP = 4.0
        self.OBSTACLE_DETECTION_THRESHOLD = 3.0 # 감지 거리 약간 늘림
        self.KP_SPEED_CONTROL = 0.5 # 제어 게인 약간 늘림
        self.MIN_SPEED = 1.0
        # [수정] 부드러운 가감속을 위한 파라미터
        self.ACCEL_STEP = 0.4
        self.DECEL_STEP = 0.7

        # --- 내부 변수 ---
        self.relative_distance = self.OBSTACLE_DETECTION_THRESHOLD + 1.0
        self.is_obstacle_detected = False
        # [수정] 노드가 직접 제어하고 기억할 현재 속도 변수 추가
        self.current_speed = self.MIN_SPEED 

        # --- ROS2 인터페이스 ---
        self.distance_sub = self.create_subscription(
            Float32, '/obstacle_distance', self.distance_callback, 10
        )
        self.command_sub = self.create_subscription(
            Float32MultiArray, '/lane_following_cmd', self.command_callback, 10
        )
        self.motor_pub = self.create_publisher(
            Float32MultiArray, '/xycar_motor', 10
        )

        self.get_logger().info('Obstacle Follower Node has been started.')

    def distance_callback(self, msg: Float32):
        self.relative_distance = msg.data
        self.is_obstacle_detected = self.relative_distance < self.OBSTACLE_DETECTION_THRESHOLD

    def command_callback(self, msg: Float32MultiArray):

        steering_angle = msg.data[0]
        target_speed = msg.data[1] 
        
        final_speed = self.current_speed

        if self.is_obstacle_detected:
            current_speed_mps = self.current_speed / 20.0 
            safe_speed_mps = max(current_speed_mps, 0.1)
            
            time_gap = self.relative_distance / safe_speed_mps
            
            error = time_gap - self.TARGET_TIME_GAP
            speed_adjustment = self.KP_SPEED_CONTROL * error
            
            # [수정] 'original_speed'가 아닌 'self.current_speed'에 제어 값을 더함
            adjusted_speed = self.current_speed + speed_adjustment
            
            # [수정] 속도 상한을 Stanley의 목표 속도로 제한
            final_speed = np.clip(adjusted_speed, self.MIN_SPEED, target_speed)
            
            self.get_logger().info(f"Vehicle Detected! Dist: {self.relative_distance:.2f}m, "
                                   f"TimeGap: {time_gap:.2f}s -> Adjusted Speed: {final_speed:.2f}")
        else:
            # ==========================================================
            # 여기가 바로 새로운 가감속 실행부입니다.
            # ==========================================================
            speed_diff = target_speed - self.current_speed

            if speed_diff > 0: # 목표 속도가 현재보다 높으면 -> 가속
                final_speed = self.current_speed + self.ACCEL_STEP
                final_speed = min(final_speed, target_speed) # 목표 속도를 넘지 않도록
                self.get_logger().info(f"No vehicle. Accelerating to {target_speed:.2f} -> Current: {final_speed:.2f}")

            elif speed_diff < 0: # 목표 속도가 현재보다 낮으면 -> 감속
                final_speed = self.current_speed - self.DECEL_STEP
                final_speed = max(final_speed, target_speed) # 목표 속도보다 더 많이 감속하지 않도록
                self.get_logger().info(f"No vehicle. Decelerating to {target_speed:.2f} -> Current: {final_speed:.2f}")

        # 최종 결정된 속도로 현재 속도를 업데이트 (이것이 피드백 루프의 핵심)
        self.current_speed = final_speed
        
        motor_msg = Float32MultiArray()
        motor_msg.data = [steering_angle, float(self.current_speed)]
        self.motor_pub.publish(motor_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleFollowerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()