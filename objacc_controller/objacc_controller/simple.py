import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import time

class SimpleMotorControlNode(Node):
    """
    단순히 지정된 조향각과 속도를 /xycar_motor 토픽으로 계속 발행하는 노드입니다.
    """
    def __init__(self):
        super().__init__('simple_motor_control_node')

        # --- 발행할 고정 값 ---
        self.target_angle = 0.0  # 목표 조향각
        self.target_speed = 10.0 # 목표 속도

        # --- ROS2 인터페이스 ---
        # '/xycar_motor' 토픽에 Float32MultiArray 메시지를 발행하는 퍼블리셔 생성
        self.motor_pub = self.create_publisher(
            Float32MultiArray,
            '/xycar_motor',
            10)

        # 0.1초마다 publish_motor_command 함수를 호출하는 타이머 생성
        self.timer = self.create_timer(0.1, self.publish_motor_command)

        self.get_logger().info('Simple Motor Control Node has been started.')
        self.get_logger().info(f'Publishing Angle: {self.target_angle}, Speed: {self.target_speed}')

    def publish_motor_command(self):
        """
        지정된 조향각과 속도 값을 메시지에 담아 발행합니다.
        """
        # Float32MultiArray 메시지 객체 생성
        motor_msg = Float32MultiArray()

        # 메시지의 data 필드에 목표 조향각과 속도를 리스트 형태로 할당
        motor_msg.data = [self.target_angle, self.target_speed]

        # 메시지 발행
        self.motor_pub.publish(motor_msg)


def main(args=None):
    """
    메인 함수: ROS2를 초기화하고 노드를 실행합니다.
    """
    rclpy.init(args=args)
    node = SimpleMotorControlNode()
    try:
        # 노드가 종료될 때까지 계속 실행 (콜백 함수들이 호출될 수 있도록)
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C로 종료 시 로그 출력
        node.get_logger().info('Node stopped by user.')
    finally:
        # 노드 소멸 및 ROS2 종료
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
