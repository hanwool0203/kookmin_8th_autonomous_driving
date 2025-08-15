#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import numpy as np
from std_msgs.msg import Float32MultiArray, String, Float32
from collections import deque

class IntegratedStanleyController(Node):
    def __init__(self):
        super().__init__('integrated_stanley_controller')

        # ==============================================================
        #                       ROS2 인터페이스
        # ==============================================================
        # 구독자들
        self.lane_sub = self.create_subscription(Point, '/lane_point', self.lane_callback, 10)
        self.distance_sub = self.create_subscription(Float32, '/obstacle_distance', self.distance_callback, 10)
        self.mode_sub = self.create_subscription(String, '/lane_mode', self.mode_callback, 10)
        self.position_sub = self.create_subscription(String, '/obstacle_position', self.position_callback, 10)
        # 퍼블리셔
        self.motor_pub = self.create_publisher(Float32MultiArray, '/xycar_motor', 10)
        self.override_pub = self.create_publisher(String, '/lane_override_cmd', 10)
        self.last_time = self.get_clock().now()
        self.get_logger().info('🎯 Integrated Stanley Controller Node Successfully Started❗')

        # ==============================================================
        #                   Stanley Controller 파라미터
        # ==============================================================
        # 1. compute_steering_angle_stanley
        self.image_center_x = 320                   # 이미지 중앙 x좌표
        self.pixel_to_meter = 1.9 / 650.0           # 픽셀 → 미터 환산계수
        self.heading_weight = 0.4                   # 헤딩오차 반영 비율
        self.deg_to_servo_scale = 5.0 / 3.0
        self.deg_to_servo_offset = 0.0

        # 2. steering angle utilities
        self.max_delta_angle = 10.0
        self.window = 3

        # 3. speed and k(gain) utilities
        self.steering_threshold = 3.0              # curve구간 판단 임계 조향각

        self.k_straight = 1.0                      
        self.k_curve = 1.2              
        self.k_obstacle = 1.2          

        self.speed_straight = 35.0                      
        self.speed_curve = 20.0     

        # ==============================================================
        #                   Obstacle Follower 파라미터
        # ==============================================================
        self.TARGET_TIME_GAP = 4.0
        self.OBSTACLE_DETECTION_THRESHOLD = 2.0    # 감지 거리
        self.KP_SPEED_CONTROL = 0.5                 # 제어 게인
        self.MIN_SPEED = 1.0
        # 부드러운 가감속을 위한 파라미터
        self.ACCEL_STEP = 0.4
        self.DECEL_STEP = 0.7

        # ==============================================================
        #                       상태 변수들
        # ==============================================================
        # Stanley Controller 변수들
        self.last_steering_deg = 0.0
        self.angle_buffer = deque(maxlen=self.window)
        self.current_k = self.k_straight                       
        self.current_mode = 'center'               # 초기 타겟 포인트

        # Obstacle Follower 변수들
        self.relative_distance = self.OBSTACLE_DETECTION_THRESHOLD + 1.0
        self.is_obstacle_detected = False
        self.current_speed = self.MIN_SPEED        # 현재 실제 속도
        self.obstacle_position = 'none'
        # 기타
        self.min_allowed_speed = min(self.speed_curve, self.speed_straight)
        self.max_allowed_speed = max(self.speed_curve, self.speed_straight)

    # ==============================================================
    #                   Stanley Controller 메서드들
    # ==============================================================
    @staticmethod
    def compute_steering_angle_stanley(lane_center_x, lane_center_y, lane_angle_rad,
                                       image_center_x, pixel_to_meter, heading_weight, 
                                       deg_to_servo_scale, deg_to_servo_offset, k):
        cte_pixel = lane_center_x - image_center_x
        cte = cte_pixel * pixel_to_meter
        heading_error = lane_angle_rad
        cte_term = np.arctan2(k * cte, 3.0)
        steering_angle_rad = heading_weight * heading_error + cte_term
        servo_angle = np.degrees(steering_angle_rad) * deg_to_servo_scale + deg_to_servo_offset
        return servo_angle, heading_error, cte_term

    @staticmethod
    def clamp_angle(angle):
        return max(min(angle, 100.0), -100.0)

    def limit_angle_change(self, current, last, max_delta_angle):
        delta = current - last
        delta = max(-max_delta_angle, min(delta, max_delta_angle))
        return last + delta

    def smooth_angle(self, new_angle):
        self.angle_buffer.append(new_angle)
        return sum(self.angle_buffer) / len(self.angle_buffer)

    def get_target_speed_and_k(self, servo_angle):
        """Stanley Controller의 목표 속도와 k값을 계산"""
        # 장애물 감지 시에는 k값만 설정하고 속도는 동적으로 제어됨
        if self.is_obstacle_detected:
            self.current_k = self.k_obstacle
            # 장애물 모드에서는 target_speed를 반환하지 않음 (동적 제어)
            if abs(servo_angle) > self.steering_threshold:
                target_speed = self.speed_curve
            else:
                target_speed = self.speed_straight
        else:
            if abs(servo_angle) > self.steering_threshold:
                target_speed = self.speed_curve
                self.current_k = self.k_curve
            else:
                target_speed = self.speed_straight
                self.current_k = self.k_straight
        
        return target_speed, self.current_k

    # ==============================================================
    #                   Obstacle Follower 메서드들
    # ==============================================================
    def calculate_final_speed(self, target_speed):
        if self.is_obstacle_detected and self.current_mode == self.obstacle_position:
            current_speed_mps = self.current_speed / 20.0 
            safe_speed_mps = max(current_speed_mps, 0.1)
            
            time_gap = self.relative_distance / safe_speed_mps
            error = time_gap - self.TARGET_TIME_GAP
            speed_adjustment = self.KP_SPEED_CONTROL * error
            
            adjusted_speed = self.current_speed + speed_adjustment
            final_speed = np.clip(adjusted_speed, self.MIN_SPEED, target_speed)
            
            return final_speed, f"🚧 FOLLOWING [{self.obstacle_position.upper()}]"
        
        else: # 장애물이 없거나 다른 차선에 있는 경우
            speed_diff = target_speed - self.current_speed
            if speed_diff > 0:
                final_speed = self.current_speed + self.ACCEL_STEP
                return min(final_speed, target_speed), "🔼 ACCELERATING"
            elif speed_diff < 0:
                final_speed = self.current_speed - self.DECEL_STEP
                return max(final_speed, target_speed), "🔽 DECELERATING"
            else:
                return self.current_speed, "➡️ MAINTAINING"

    # ==============================================================
    #                           콜백 함수들
    # ==============================================================
    def position_callback(self, msg: String):
        """장애물 위치 정보 콜백"""
        self.obstacle_position = msg.data.strip().lower()

    def mode_callback(self, msg: String):
        """차선 모드 변경 콜백"""
        mode = msg.data.strip().lower()
        if mode in ('center', 'left', 'right'):
            self.current_mode = mode

    def distance_callback(self, msg: Float32):
        """장애물 거리 정보 콜백"""
        self.relative_distance = msg.data
        self.is_obstacle_detected = self.relative_distance < self.OBSTACLE_DETECTION_THRESHOLD
        # [수정] 거리 정보를 받을 때마다 회피 명령을 보낼지 체크
        if self.is_obstacle_detected and self.obstacle_position == 'right' and self.current_mode != 'left':
            self.get_logger().warn("🚨 Condition met! Sending 'go_left' command to Lane Detector.")
            cmd_msg = String()
            cmd_msg.data = 'go_left'
            self.override_pub.publish(cmd_msg)

        if self.is_obstacle_detected and self.obstacle_position == 'left' and self.current_mode != 'right':
            self.get_logger().warn("🚨 Condition met! Sending 'go_right' command to Lane Detector.")
            cmd_msg = String()
            cmd_msg.data = 'go_right'
            self.override_pub.publish(cmd_msg)
        
    def lane_callback(self, msg: Point):
        """차선 정보 콜백 - 메인 제어 로직"""
        lane_center_x = msg.x
        lane_center_y = msg.y
        lane_angle_rad = msg.z

        # 1. Stanley Controller로 조향각 계산
        servo_angle, heading_error_rad, cte_term = self.compute_steering_angle_stanley(
            lane_center_x=lane_center_x,
            lane_center_y=lane_center_y,
            lane_angle_rad=lane_angle_rad,
            image_center_x=self.image_center_x,
            pixel_to_meter=self.pixel_to_meter,
            heading_weight=self.heading_weight,
            deg_to_servo_scale=self.deg_to_servo_scale,
            deg_to_servo_offset=self.deg_to_servo_offset,
            k=self.current_k
        )

        # 2. 조향각 후처리 (rate limiting, smoothing, clamping)
        raw_delta = servo_angle - self.last_steering_deg
        rate_limited = abs(raw_delta) > self.max_delta_angle

        limited_angle = self.limit_angle_change(servo_angle, self.last_steering_deg, self.max_delta_angle)
        smoothed_angle = self.smooth_angle(limited_angle)
        final_angle = self.clamp_angle(smoothed_angle)
        self.last_steering_deg = final_angle

        # 3. Stanley Controller의 목표 속도 계산
        target_speed, current_k = self.get_target_speed_and_k(smoothed_angle)

        # 4. Obstacle Follower로 최종 속도 결정
        final_speed, speed_status = self.calculate_final_speed(target_speed)
        self.current_speed = final_speed  # 현재 속도 업데이트

        # 5. 모터 명령 발행
        motor_msg = Float32MultiArray()
        motor_msg.data = [final_angle, float(self.current_speed)]
        self.motor_pub.publish(motor_msg)

        # 6. 로깅
        self.log_status(heading_error_rad, cte_term, current_k, final_angle, 
                       target_speed, final_speed, speed_status, rate_limited)

    def log_status(self, heading_error_rad, cte_term, current_k, final_angle, 
                   target_speed, final_speed, speed_status, rate_limited):
        """상태 로깅"""
        # 차선 타겟 모드 표시
        if self.current_mode == 'center':
            lane_target_str = "🟡 CENTER"
        elif self.current_mode == 'left':
            lane_target_str = "🔵 LEFT"
        elif self.current_mode == 'right':
            lane_target_str = "🔴 RIGHT"
        else:
            lane_target_str = "⚪ UNKNOWN"

        # 주행 모드 표시 (장애물 감지 기반)
        if self.is_obstacle_detected:
            driving_mode = "🚧 OBSTACLE MODE"
        else:
            driving_mode = "🔄 CURVE MODE" if abs(final_angle) > self.steering_threshold else "➡️ STRAIGHT MODE"

        # Rate limiting 표시
        rate_limit_marker = "⛔" if rate_limited else ""
        
        # 장애물 상태 표시
        obstacle_status = f"🚗 Dist: {self.relative_distance:.2f}m" if self.is_obstacle_detected else "🟢 Clear"

        # 로그 출력 (매 프레임)
        self.get_logger().info(
            "\n📐 [Integrated Stanley Controller Log] ---------------\n"
            f"  • Lane Target       : {lane_target_str}\n"
            f"  • Driving Mode      : {driving_mode}\n"
            f"  • Obstacle Status   : {obstacle_status}\n"
            f"  • Stanley K Value   : {current_k:>6.2f}\n"
            f"  • Heading Error     : {np.degrees(heading_error_rad):>6.2f}°\n"
            f"  • Cross Track Term  : {np.degrees(cte_term):>6.2f}°\n"
            f"  • Servo Angle       : {final_angle:>6.2f} {rate_limit_marker}\n"
            f"  • Target Speed      : {target_speed:>6.2f}\n"
            f"  • Final Speed       : {final_speed:>6.2f} {speed_status}\n"
            "--------------------------------------------------------"
        )

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedStanleyController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()