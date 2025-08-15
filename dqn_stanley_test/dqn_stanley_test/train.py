# 파일명: train.py

import rclpy
from dqn_stanley_test.stanley_env import StanleyEnv

def main(args=None):
    """
    메인 함수: 학습 모드로 설정된 ROS2 노드를 초기화하고 실행합니다.
    실제 학습 루프는 StanleyEnv 클래스의 lane_callback에서 동기적으로 실행됩니다.
    """
    rclpy.init(args=args)

    # test_mode=False로 환경 및 트레이너 노드 생성
    env_and_trainer_node = StanleyEnv(test_mode=False)

    try:
        # 노드를 스핀시켜 콜백 함수들이 활성화되도록 함
        rclpy.spin(env_and_trainer_node)
    except KeyboardInterrupt:
        env_and_trainer_node.get_logger().info("Training stopped by user.")
    finally:
        # 노드 종료 및 자원 해제
        env_and_trainer_node.get_logger().info("Shutting down...")
        env_and_trainer_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()