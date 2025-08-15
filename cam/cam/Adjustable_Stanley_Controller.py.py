#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import numpy as np
from std_msgs.msg import Float32MultiArray
import cv2

class Stanley_Controller(Node):
    """
    Stanley 제어 기법을 사용하여 차선 추종 주행을 수행하는 ROS2 노드.
    OpenCV GUI를 통해 실시간으로 주요 파라미터를 튜닝할 수 있다.
    """
    def __init__(self):
        """노드 및 파라미터 초기화."""
        super().__init__('stanley_controller')

        # ==================================================
        # 1. 동적 제어 파라미터 (GUI 트랙바로 실시간 조절)
        # ==================================================
        self.k = 1.0                          # 횡방향 오차(cross-track error) 보정을 위한 제어 이득.
        self.speed = 20.0                     # 차량의 목표 속도.
        self.heading_weight = 0.4             # 헤딩 오차(heading error)가 조향각에 미치는 영향의 가중치.
        self.speed_denominator_const = 3.0    # 횡방향 오차 항의 분모 값. 속도 대신 사용되며, 제어 민감도를 조절.
        self.max_delta_servo_angle = 7.0      # 한 스텝 당 조향각의 최대 변화량 (조향 안정화).
        self.smoothing_window = 7             # 이동 평균 필터를 위한 윈도우 크기 (조향각 스무딩).

        # ==================================================
        # 2. 정적 환경/하드웨어 파라미터
        # ==================================================
        self.image_center_x = 320             # 카메라 이미지의 가로 중앙 픽셀 좌표.
        self.pixel_to_meter = 1.9 / 650.0     # 이미지 픽셀 단위를 미터 단위로 변환하기 위한 계수.
        self.deg_to_servo_scale = 5.0 / 3.0   # 계산된 조향각(degree)을 서보 모터 제어값으로 변환하는 스케일.

        # ==================================================
        # 3. 내부 상태 변수
        # ==================================================
        self.last_steering_deg = 0.0          # 이전 스텝의 조향각.
        self.angle_buffer = []                # 조향각 스무딩을 위한 버퍼.

        # ==================================================
        # 4. ROS2 퍼블리셔 및 서브스크라이버 설정
        # ==================================================
        self.lane_sub = self.create_subscription(
            Point,
            '/lane_point',      # 차선 정보 토픽
            self.lane_callback,
            10
        )
        self.motor_pub = self.create_publisher(
            Float32MultiArray,
            '/xycar_motor',     # 모터 제어 토픽
            10
        )

        # ==================================================
        # 5. GUI 및 타이머 설정
        # ==================================================
        self.setup_gui()
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback) # 30Hz 주기로 GUI 이벤트 처리
        
        self.get_logger().info('Stanley_Controller Node with GUI Successfully!')

    ## ----------------------------------------------------------------
    ## GUI 트랙바 콜백 함수
    ## ----------------------------------------------------------------
    
    def on_k_trackbar(self, val):
        # 트랙바는 정수 값만 다루므로, 소수점을 표현하기 위해 100을 곱한 값을 사용하고 콜백에서 다시 100으로 나눕니다.
        self.k = val / 100.0

    def on_speed_trackbar(self, val):
        self.speed = float(val)

    def on_heading_weight_trackbar(self, val):
        # 트랙바는 정수 값만 다루므로, 소수점을 표현하기 위해 100을 곱한 값을 사용하고 콜백에서 다시 100으로 나눕니다.
        self.heading_weight = val / 100.0

    def on_speed_denom_trackbar(self, val):
        # 트랙바는 정수 값만 다루므로, 소수점을 표현하기 위해 10을 곱한 값을 사용하고 콜백에서 다시 10으로 나눕니다.
        # 0으로 나누는 것을 방지합니다.
        self.speed_denominator_const = val / 10.0 if val > 0 else 0.1

    def on_max_delta_trackbar(self, val):
        self.max_delta_servo_angle = float(val)

    def on_smoothing_window_trackbar(self, val):
        # 윈도우 크기는 최소 1이어야 합니다.
        self.smoothing_window = val if val > 0 else 1

    ## ----------------------------------------------------------------
    ## 초기 설정 함수
    ## ----------------------------------------------------------------

    def setup_gui(self):
        """파라미터 튜닝을 위한 OpenCV GUI 창과 트랙바를 생성한다."""
        cv2.namedWindow('Stanley Params')
        # 트랙바 생성: createTrackbar(트랙바 이름, 창 이름, 초기값, 최대값, 콜백함수)
        # 클래스 속성에 저장된 초기값을 트랙바의 기본값으로 사용한다.
        cv2.createTrackbar('k * 100', 'Stanley Params', int(self.k * 100), 500, self.on_k_trackbar)
        cv2.createTrackbar('speed', 'Stanley Params', int(self.speed), 100, self.on_speed_trackbar)
        cv2.createTrackbar('heading_weight * 100', 'Stanley Params', int(self.heading_weight * 100), 100, self.on_heading_weight_trackbar)
        cv2.createTrackbar('speed_denom * 10', 'Stanley Params', int(self.speed_denominator_const * 10), 100, self.on_speed_denom_trackbar)
        cv2.createTrackbar('max_delta_angle', 'Stanley Params', int(self.max_delta_servo_angle), 30, self.on_max_delta_trackbar)
        cv2.createTrackbar('smoothing_window', 'Stanley Params', self.smoothing_window, 30, self.on_smoothing_window_trackbar)
    
    def timer_callback(self):
        """GUI 이벤트 처리를 위해 주기적으로 호출되는 타이머 콜백."""
        cv2.waitKey(1)

    ## ----------------------------------------------------------------
    ## 제어 로직 및 유틸리티 함수
    ## ----------------------------------------------------------------

    @staticmethod
    def compute_steering_angle_stanley(lane_center_x, lane_angle_rad,
                                     image_center_x, pixel_to_meter, k, heading_weight,
                                     speed_denominator_const, deg_to_servo_scale):
        """
        Stanley 제어 공식에 따라 목표 조향각을 계산한다.

        Args:
            lane_center_x: 감지된 차선의 중심 x좌표 (픽셀).
            lane_angle_rad: 감지된 차선의 기울기 (라디안).
            image_center_x: 이미지의 가로 중앙 x좌표 (픽셀).
            pixel_to_meter: 픽셀-미터 변환 계수.
            k: 횡방향 오차 제어 이득.
            heading_weight: 헤딩 오차 가중치.
            speed_denominator_const: 횡방향 오차 항의 분모 값.
            deg_to_servo_scale: 각도-서보값 변환 스케일.

        Returns:
            Tuple: (서보 제어값, 조향각(도), 헤딩 오차(도), 횡방향 오차 항(도))
        """
        # 횡방향 오차(Cross-track Error) 계산
        cte_pixel = lane_center_x - image_center_x
        cte = cte_pixel * pixel_to_meter
        
        # 헤딩 오차(Heading Error)
        heading_error = lane_angle_rad
        heading_error_deg = np.degrees(heading_error)
        
        # 횡방향 오차에 대한 조향각 계산 (arctan(k*e/v))
        cte_term = np.arctan2(k * cte, speed_denominator_const)
        cte_term_deg = np.degrees(cte_term)
        
        # 최종 조향각 = 헤딩 오차 + 횡방향 오차
        steering_angle_rad = heading_weight * heading_error + cte_term
        steering_angle_deg = np.degrees(steering_angle_rad)
        
        # 서보모터 제어값으로 변환
        servo_angle = steering_angle_deg * deg_to_servo_scale
        return servo_angle, steering_angle_deg, heading_error_deg, cte_term_deg

    @staticmethod
    def clamp_angle(angle):
        """조향각의 최대/최소값을 제한한다."""
        return max(min(angle, 100.0), -100.0)

    def limit_angle_change(self, current, last, max_delta_servo_angle):
        """조향각의 급격한 변화를 방지하기 위해 변화율을 제한한다."""
        delta = current - last
        delta = max(-max_delta_servo_angle, min(delta, max_delta_servo_angle))
        return last + delta

    def smooth_angle(self, new_angle, window):
        """이동 평균 필터를 사용하여 조향각을 부드럽게 만든다."""
        self.angle_buffer.append(new_angle)
        if len(self.angle_buffer) > window:
            self.angle_buffer.pop(0)
        return sum(self.angle_buffer) / len(self.angle_buffer)

    ## ----------------------------------------------------------------
    ## 메인 콜백 함수
    ## ----------------------------------------------------------------

    def lane_callback(self, msg: Point):
        """
        차선 정보를 수신하여 제어 로직을 수행하고 모터 명령을 발행하는 메인 콜백.
        """
        # 입력 메시지에서 차선 데이터 추출
        lane_center_x = msg.x
        lane_angle_rad = msg.z

        # 1. Stanley 제어 알고리즘을 통해 목표 조향각 계산
        servo_angle, steering_angle_deg, heading_error_deg, cte_term_deg = self.compute_steering_angle_stanley(
            lane_center_x=lane_center_x,
            lane_angle_rad=lane_angle_rad,
            image_center_x=self.image_center_x,
            pixel_to_meter=self.pixel_to_meter,
            k=self.k,
            heading_weight=self.heading_weight,
            speed_denominator_const=self.speed_denominator_const,
            deg_to_servo_scale=self.deg_to_servo_scale
        )

        # 2. 조향각 안정화를 위한 후처리 (변화량 제한 및 스무딩)
        clamped_angle = self.clamp_angle(servo_angle)
        limited_angle = self.limit_angle_change(clamped_angle, self.last_steering_deg, self.max_delta_servo_angle)
        smoothed_angle = self.smooth_angle(limited_angle, self.smoothing_window)
        self.last_steering_deg = smoothed_angle

        # 3. 최종 제어값 (조향각, 속도)을 모터 토픽으로 발행
        motor_msg = Float32MultiArray()
        motor_msg.data = [smoothed_angle, self.speed]
        self.motor_pub.publish(motor_msg)

        # 4. 현재 제어 상태 로깅
        self.get_logger().info(
            "\n [Stanley Controller Log] ---------------------------\n"
            f"  • Heading Error     : {heading_error_deg:>6.2f}°\n"
            f"  • Cross Track Term  : {cte_term_deg:>6.2f}°\n"
            f"  • Steering Angle    : {steering_angle_deg:>6.2f}°\n"
            f"  • Servo Angle (Out) : {smoothed_angle:>6.2f}\n"
            f"  • Speed (Out)       : {self.speed:>6.1f}\n"
            "--------------------------------------------------------"
        )
    
    def destroy_node(self):
        """노드 종료 시 OpenCV 창을 안전하게 닫는다."""
        super().destroy_node()
        cv2.destroyAllWindows()


def main(args=None):
    """메인 함수: ROS2 노드를 초기화하고 실행한다."""
    rclpy.init(args=args)
    node = Stanley_Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT) received. Shutting down...')
    finally:
        # 노드 종료 및 자원 해제
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()