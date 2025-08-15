# 파일명: test.py

import rclpy
from dqn_stanley_test.stanley_env import StanleyEnv

def main(args=None):
    """
    DQN 에이전트의 학습된 정책을 평가하기 위한 메인 실행 스크립트입니다.
    
    이 스크립트는 StanleyEnv 노드를 평가 모드(test_mode=True)로
    초기화하고 실행하여, 에이전트가 학습 없이 오직 학습된 정책에 따라
    주행하는 성능을 확인할 수 있도록 합니다.
    """
    # ROS2 시스템을 초기화합니다.
    rclpy.init(args=args)

    # 평가 모드(test_mode=True)로 환경 노드를 생성합니다.
    test_node = StanleyEnv(test_mode=True)

    try:
        # 노드를 스핀(spin)시켜 ROS2 이벤트 및 콜백 함수들이 지속적으로 처리되도록 합니다.
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        # 사용자가 Ctrl+C로 종료를 요청할 경우, 정보 메시지를 출력합니다.
        test_node.get_logger().info("Test stopped by user.")
    finally:
        # 프로그램 종료 시, 노드와 관련된 자원을 안전하게 해제합니다.
        test_node.get_logger().info("Shutting down...")
        test_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    # 이 스크립트가 직접 실행되었을 때 main 함수를 호출합니다.
    main()