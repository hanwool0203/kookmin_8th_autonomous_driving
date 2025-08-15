# 파일명: stanley_env_refactored.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import numpy as np
from std_msgs.msg import Float32MultiArray, Bool
import threading
import sys
import termios
import tty
from collections import deque
import torch
import math

from dqn_stanley_test.dqn_agent import DQNAgent

class StanleyEnv(Node):
    """
    강화학습 '환경(Environment)'과 ROS2 '노드(Node)'의 역할을 동시에 수행하는 클래스입니다.
    로봇(시뮬레이터)으로부터 센서 데이터를 수신하여 '상태(State)'를 정의하고,
    학습 에이전트가 결정한 '행동(Action)'을 통해 Stanley 제어 파라미터를 동적으로 결정합니다.
    """
    def __init__(self, test_mode=False):
        node_name = 'stanley_test_node' if test_mode else 'stanley_env_and_trainer'
        super().__init__(node_name)

        self.is_test_mode = test_mode

        # --- 강화학습(RL) 핵심 요소 ---
        self.state_dim = 6
        self.action_dim = 7  # 6개의 파라미터 조정 + 1개의 아무것도 안 함(no-op)
        self.agent = DQNAgent(state_dim=self.state_dim, action_dim=self.action_dim)
        
        ### 변경점 1: 제어 파라미터 중앙 관리 ###
        # Stanley 제어에 사용될 파라미터들을 딕셔너리로 묶어 관리합니다.
        # 이 파라미터들이 RL 에이전트의 '학습 대상'이 됩니다.
        self.control_params = {
            'k': 1.2,           # CTE에 대한 게인
            'speed': 15.0,      # 목표 속도
            'heading_weight': 0.4 # 전방 주시각 오차에 대한 가중치
        }
        
        ### 변경점 2: 명시적인 행동(Action) 정의 ###
        # action 정수 값에 따라 어떤 파라미터를 어떻게 변경할지 명시적으로 매핑합니다.
        # lambda 함수를 사용하여 코드의 가독성과 확장성을 높입니다.
        self.action_map = [
            lambda p: self._update_param(p, 'k', -0.1),      # action 0: k 감소
            lambda p: self._update_param(p, 'k', 0.1),       # action 1: k 증가
            lambda p: self._update_param(p, 'heading_weight', -0.05), # action 2: heading_weight 감소
            lambda p: self._update_param(p, 'heading_weight', 0.05),  # action 3: heading_weight 증가
            lambda p: self._update_param(p, 'speed', -2.0),     # action 4: speed 감소
            lambda p: self._update_param(p, 'speed', 1.0),      # action 5: speed 증가
            lambda p: p # action 6: No-op (아무것도 변경하지 않음)
        ]

        # --- 조향각 스무딩 파라미터 ---
        self.last_steering_deg = 0.0
        self.angle_buffer = deque(maxlen=3)

        self.lane_center_x = None
        self.lane_center_y = None
        self.lane_angle_rad = None
        
        # --- 보상 함수 가중치 ---
        self.W_STABILITY = 1.0
        self.W_SPEED = 0.2
        
        # --- 정적 시스템 파라미터 ---
        self.image_center_x = 320
        self.pixel_to_meter = 1.9 / 650.0
        self.deg_to_servo_scale = 5.0 / 3.0
        self.deg_to_servo_offset = 2.0
        self.max_delta_angle = 7.0

        # --- 실시간 상태 변수 ---
        self.current_cte = 0.0
        self.current_heading_error = 0.0
        
        # --- RL 경험(Transition) 구성 요소 ---
        self.last_state = None
        self.last_action = None
        
        # --- 에피소드 흐름 제어 플래그 ---
        self.episode_done = True
        self.goal_reached = False
        self.goal_shutdown_imminent = False
        self.reset_requested = False
        self.emergency_stop_activated = False

        # --- 학습 통계 기록 ---
        self.episode_count = 0
        self.step_count = 0
        self.current_episode_score = 0
        self.scores_window = deque(maxlen=100)
        
        # --- ROS2 통신 인터페이스 ---
        self.lane_sub = self.create_subscription(Point, '/lane_point', self.lane_callback, 1)
        self.motor_pub = self.create_publisher(Float32MultiArray, '/xycar_motor', 10)
        self.checker_sub = self.create_subscription(Bool, '/checkerboard_detect', self.checkerboard_callback, 10)
        
        # --- 비동기 키보드 입력 처리 ---
        self.key_listener_thread = threading.Thread(target=self._keyboard_listener)
        self.key_listener_thread.daemon = True
        self.key_listener_thread.start()

        # --- 노드 시작 안내 메시지 (이전과 동일) ---
        log_info = "Stanley Controller - TEST MODE" if test_mode else "🎯 Stanley RL Trainer Node Ready"
        self.get_logger().info("==============================================")
        self.get_logger().info(f"<<<<< {log_info} >>>>>")
        self.get_logger().info("   Press 'r' to start/reset an episode.")
        self.get_logger().info("   Press 'f' for EMERGENCY STOP.")
        self.get_logger().info("==============================================")

    def _keyboard_listener(self):
        if not sys.stdin.isatty():
            self.get_logger().warn("키보드 입력을 위한 터미널(TTY)이 없습니다. 키보드 제어가 비활성화됩니다.")
            return
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                char = sys.stdin.read(1)
                if char == 'r': self.reset_requested = True
                elif char == 'f': self.emergency_stop_activated = True
        except termios.error: pass
        finally: termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def checkerboard_callback(self, msg: Bool):
        if msg.data and not self.goal_reached:
            self.goal_reached = True
            self.get_logger().info("🏁 Goal (Checkerboard) Detected! Preparing to stop.")

    def _get_state(self) -> np.ndarray:
        """
        환경의 여러 변수들을 조합하여, 에이전트가 이해할 수 있는 형태의
        하나의 '상태(State)' 벡터로 변환합니다.
        """
        ### 변경점 3: 중앙 관리 파라미터를 사용하여 상태 구성 ###
        state = [
            self.current_cte,
            self.current_heading_error,
            self.last_steering_deg,
            self.control_params['speed'],
            self.control_params['k'],
            self.control_params['heading_weight']
        ]
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self) -> float:
        """
        에이전트의 행동 결과가 얼마나 좋았는지를 정량적인 '보상(Reward)' 값으로 계산합니다.
        """
        stability_reward_norm = np.tanh(1.0 / (abs(self.current_cte) + 0.1))
        ### 변경점 4: 중앙 관리 파라미터를 사용하여 보상 계산 ###
        speed_reward_norm = self.control_params['speed'] / 20.0 # 최대 속도 기준으로 정규화
        reward = self.W_STABILITY * stability_reward_norm + self.W_SPEED * speed_reward_norm
        return reward
    
    def reset(self) -> np.ndarray:
        """
        하나의 에피소드를 끝내고, 다음 에피소드를 위해 모든 관련 변수를 초기화합니다.
        """
        ### 변경점 5: 중앙 관리 파라미터 초기화 ###
        self.control_params = {'k': 1.0, 'speed': 10.0, 'heading_weight': 0.4}
        
        self.last_steering_deg, self.current_cte, self.current_heading_error = 0.0, 0.0, 0.0
        self.last_state, self.last_action = None, None
        
        self.episode_done, self.goal_reached, self.goal_shutdown_imminent = False, False, False
        self.reset_requested, self.emergency_stop_activated = False, False

        if not self.is_test_mode:
            self.episode_count += 1
            self.current_episode_score = 0
            self.step_count = 0
            self.get_logger().info(f"--- Episode {self.episode_count} Started ---")
        else:
            self.get_logger().info("--- Test Episode Started ---")

        return self._get_state()
    
    @staticmethod
    def compute_steering_angle_stanley(lane_center_x, lane_center_y, lane_angle_rad,
                                       image_center_x, pixel_to_meter, heading_weight, 
                                       deg_to_servo_scale, deg_to_servo_offset, k):
        cte_pixel = lane_center_x - image_center_x
        cte = cte_pixel * pixel_to_meter
        heading_error = lane_angle_rad
        
        cte_term = np.arctan2(k * cte, 3.0) # 속도 항은 단순화를 위해 상수로 가정 (혹은 speed 파라미터 사용 가능)
        steering_angle_rad = heading_weight * heading_error + cte_term
        
        steering_angle_deg = np.degrees(steering_angle_rad)
        servo_angle = steering_angle_deg * deg_to_servo_scale + deg_to_servo_offset
        return servo_angle, heading_error, cte
    
    # 조향각 처리 유틸리티 함수 (이전과 동일)
    @staticmethod
    def clamp_angle(angle): return max(min(angle, 100.0), -100.0)
    def limit_angle_change(self, current, last): return last + max(-self.max_delta_angle, min(current - last, self.max_delta_angle))
    def smooth_angle(self, new_angle):
        self.angle_buffer.append(new_angle)
        return sum(self.angle_buffer) / len(self.angle_buffer)

    ### 변경점 6: `_update_param` 헬퍼 함수 추가 ###
    def _update_param(self, params, key, delta):
        """행동에 따라 제어 파라미터를 업데이트하고 범위를 제한하는 헬퍼 함수"""
        params[key] += delta
        if key == 'k':
            params[key] = np.clip(params[key], 0.1, 3.0)
        elif key == 'heading_weight':
            params[key] = np.clip(params[key], 0.0, 1.0)
        elif key == 'speed':
            params[key] = np.clip(params[key], 5, 20)
        return params

    def _execute_action(self, action: int):
        """
        에이전트가 선택한 행동에 따라 제어 파라미터를 업데이트하고,
        새 파라미터를 사용하여 Stanley 제어 조향각을 계산 및 발행합니다.
        """
        ### 변경점 7: `action_map`을 이용한 파라미터 업데이트 ###
        # action_map에서 해당 action에 맞는 업데이트 함수를 실행합니다.
        self.control_params = self.action_map[action](self.control_params)

        # 업데이트된 파라미터를 사용하여 Stanley 제어기법으로 조향각 계산
        servo_angle, _, _ = self.compute_steering_angle_stanley(
            lane_center_x=self.lane_center_x,
            lane_center_y=self.lane_center_y,
            lane_angle_rad=self.lane_angle_rad,
            image_center_x=self.image_center_x,
            pixel_to_meter=self.pixel_to_meter,
            deg_to_servo_scale=self.deg_to_servo_scale,
            deg_to_servo_offset=self.deg_to_servo_offset,
            # 중앙 관리되는 파라미터 사용
            k=self.control_params['k'],
            heading_weight=self.control_params['heading_weight']
        )
        
        # 조향각 스무딩 및 제한 (이전과 동일한 로직)
        limited_angle = self.limit_angle_change(servo_angle, self.last_steering_deg)
        smoothed_angle = self.smooth_angle(limited_angle)
        final_angle = self.clamp_angle(smoothed_angle)
        
        # 모터 명령 발행
        motor_msg = Float32MultiArray()
        motor_msg.data = [final_angle, self.control_params['speed']]
        self.motor_pub.publish(motor_msg)
        
        self.last_steering_deg = final_angle

    def _execute_stop_command(self):
        motor_msg = Float32MultiArray()
        motor_msg.data = [0.0, 0.0]
        self.motor_pub.publish(motor_msg)

    def lane_callback(self, msg: Point):
        """
        센서 데이터가 수신될 때마다 호출되는 메인 루프(Main Loop)입니다.
        RL의 (State -> Action -> Reward -> next_State) 사이클이 여기서 실행됩니다.
        """
        self.lane_center_x = msg.x
        self.lane_center_y = msg.y
        self.lane_angle_rad = msg.z
        
        # 비상 정지 또는 수동 리셋 처리 (이전과 동일)
        if self.emergency_stop_activated or self.reset_requested:
            is_manual_reset = self.reset_requested
            self._execute_stop_command()
            if not self.episode_done and not self.is_test_mode:
                reason = "MANUAL RESET" if is_manual_reset else "EMERGENCY STOP"
                self.get_logger().info(f"--- Episode Finished by {reason} ---")
                self.agent.save_model()
            
            self.episode_done = True
            if is_manual_reset: self.reset()
            return
            
        if self.episode_done: return
            
        # 현재 상태 오차 계산
        self.current_cte = (msg.x - self.image_center_x) * self.pixel_to_meter
        self.current_heading_error = msg.z
        
        # 테스트 모드 로직 (이전과 거의 동일)
        if self.is_test_mode:
            state = self._get_state()
            action = self.agent.select_action(state, eval_mode=True).item()
            self._execute_action(action)
            self.get_logger().info(f"CTE: {self.current_cte:.3f}, Speed: {self.control_params['speed']:.1f}, Action: {action}", throttle_duration_sec=0.2)
            # ... (이하 생략)
            return

        # --- 학습 모드 메인 로직 ---
        self.step_count += 1
        device = next(self.agent.policy_net.parameters()).device
        
        current_state = self._get_state()
        
        if self.last_state is not None:
            reward = self._calculate_reward()
            self.current_episode_score += reward
            self.agent.memory.push(self.last_state, self.last_action, current_state, torch.tensor([reward], device=device))
        
        is_off_track = abs(self.current_cte) > 1.0
        done = is_off_track or self.goal_shutdown_imminent

        if done:
            self.episode_done = True
            final_reward = 200.0 if self.goal_reached else -100.0
            self.current_episode_score += final_reward
            if self.last_state is not None:
                self.agent.memory.push(current_state, self.last_action, None, torch.tensor([final_reward], device=device))

            self.scores_window.append(self.current_episode_score)
            avg_score = np.mean(self.scores_window)
            result_msg = 'Success!' if self.goal_reached else 'Off Track'
            self.get_logger().info(
                f"--- Episode {self.episode_count} Finished ---\n"
                f"Score: {self.current_episode_score:.2f}, Avg Score: {avg_score:.2f}\n"
                f"Result: {result_msg}"
            )
            if not self.is_test_mode: self.agent.save_model()
            return

        if self.goal_reached and not self.goal_shutdown_imminent:
            self._execute_stop_command()
            self.goal_shutdown_imminent = True
            action_tensor = torch.tensor([[self.action_dim - 1]], device=device, dtype=torch.long) # No-op
        else:
            action_tensor = self.agent.select_action(current_state, eval_mode=False)
            self._execute_action(action_tensor.item())

        self.last_state = current_state
        self.last_action = action_tensor
        
        self.agent.optimize_model()
        self.agent.update_target_net()

        if self.agent.steps_done % 20 == 0:
            eps = self.agent.EPS_END + (self.agent.EPS_START - self.agent.EPS_END) * \
                  math.exp(-1. * self.agent.steps_done / self.agent.EPS_DECAY)
            
            ### 변경점 8: 로그 메시지에서 중앙 관리 파라미터 사용 ###
            log_msg = (
                f"[Ep.{self.episode_count:03d} | Step.{self.step_count:04d}] "
                f"CTE: {self.current_cte:+.3f} | "
                f"Action: {action_tensor.item()} | "
                f"Eps: {eps:.3f} | "
                f"Params(k/speed/h_w): {self.control_params['k']:.2f}/{self.control_params['speed']:.1f}/{self.control_params['heading_weight']:.2f}"
            )
            self.get_logger().info(log_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        # 리팩토링된 클래스를 사용합니다.
        node = StanleyEnv(test_mode=False) 
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info("Training stopped by user (Ctrl+C).")
    except termios.error: pass
    finally:
        if node:
            if not node.is_test_mode:
                node.get_logger().info("Saving model on shutdown...")
                node.agent.save_model()
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()