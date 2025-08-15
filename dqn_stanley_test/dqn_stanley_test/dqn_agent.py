# 파일명: dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import numpy as np
import os

from dqn_stanley_test.dqn_model import DQN
from dqn_stanley_test.replay_buffer import ReplayBuffer, Transition

# 연산에 사용할 장치를 설정합니다. (GPU 사용 가능 시 "cuda", 아닐 경우 "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    """
    DQN(Deep Q-Network) 알고리즘의 학습 및 행동 선택 로직을 총괄하는 에이전트 클래스입니다.
    신경망, 리플레이 버퍼, 학습 로직 등을 포함하여 강화학습의 전반적인 과정을 관리합니다.
    """
    def __init__(self, state_dim, action_dim, model_name='dqn_policy_net.pth'):
        """
        DQNAgent 클래스의 생성자입니다. 하이퍼파라미터, 신경망, 옵티마이저 등을 초기화합니다.

        Args:
            state_dim (int): 상태 벡터의 차원.
            action_dim (int): 행동 공간의 크기.
            model_name (str): 저장하거나 불러올 모델 파일의 이름.
        """
        # --- 강화학습 하이퍼파라미터 ---
        self.BATCH_SIZE = 128      # 학습 시 한 번에 사용할 경험(transition)의 수
        self.GAMMA = 0.99          # 미래 보상에 대한 할인율 (Discount Factor)
        self.EPS_START = 0.9       # Epsilon-Greedy 전략의 초기 탐험 확률
        self.EPS_END = 0.05        # Epsilon-Greedy 전략의 최종 탐험 확률
        self.EPS_DECAY = 1000      # 탐험 확률이 감소하는 속도를 결정하는 계수
        self.TAU = 0.005           # 타겟 네트워크를 부드럽게 업데이트하기 위한 계수 (Soft Update)
        self.LR = 1e-4             # 신경망 학습률 (Learning Rate)

        self.state_dim = state_dim
        self.action_dim = action_dim

        # --- 모델 저장 경로 설정 ---
        # 학습된 모델 가중치가 저장될 절대 경로를 지정합니다.
        # 참고: 이 경로는 특정 환경에 맞춰 하드코딩되어 있으므로,
        #       다른 PC나 워크스페이스에서 실행 시에는 이 부분을 수정해야 할 수 있습니다.
        save_directory = "/home/xytron/xycar_ws/src/dqn_stanley_test"
        self.model_path = os.path.join(save_directory, model_name)
        
        # --- 신경망 초기화 ---
        # 주 행동을 결정하는 정책 네트워크(Policy Network)와 학습 안정화를 위한 타겟 네트워크(Target Network)를 생성합니다.
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.load_model()  # 인스턴스 생성 시 저장된 모델 가중치가 있다면 불러옵니다.
        
        # 정책 네트워크는 학습 중에 가중치가 변하지만, 타겟 네트워크는 평가에만 사용됩니다.
        self.target_net.eval()

        # --- 학습 관련 요소 초기화 ---
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayBuffer(10000) # 경험 리플레이를 위한 메모리 버퍼
        self.steps_done = 0 # 총 스텝 수를 기록하여 Epsilon 값 감소에 사용

    def select_action(self, state, eval_mode=False):
        """
        Epsilon-Greedy 전략을 사용하여 현재 상태(state)에 대한 행동(action)을 선택합니다.

        Args:
            state (np.ndarray): 현재 상태 정보.
            eval_mode (bool): 평가 모드 여부. True일 경우 탐험을 하지 않고 학습된 최적의 행동만 선택합니다.

        Returns:
            torch.Tensor: 선택된 행동의 인덱스를 담은 텐서.
        """
        # 평가(test) 모드에서는 탐험 없이 항상 최적의 행동을 선택합니다.
        if eval_mode:
            with torch.no_grad():
                state_tensor = torch.tensor(np.array([state]), device=device, dtype=torch.float32)
                # 정책 네트워크를 통해 각 행동의 Q-value를 예측하고, 가장 큰 값을 가진 행동을 선택합니다.
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)

        # 학습 모드에서는 Epsilon-Greedy 전략에 따라 행동을 선택합니다.
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            # 활용(Exploitation): 학습된 정책에 따라 최적의 행동을 선택합니다.
            with torch.no_grad():
                state_tensor = torch.tensor(np.array([state]), device=device, dtype=torch.float32)
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            # 탐험(Exploration): 무작위로 행동을 선택하여 새로운 가능성을 탐색합니다.
            return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)

    def optimize_model(self):
        """
        리플레이 메모리에서 경험의 미니배치를 샘플링하여 정책 네트워크를 1스텝 학습(업데이트)합니다.
        """
        # 메모리에 충분한 경험이 쌓이기 전까지는 학습을 시작하지 않습니다.
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # 종료되지 않은 상태(non-final state)들의 마스크와 그 상태들을 분리합니다.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([torch.tensor(np.array([s]), device=device, dtype=torch.float32)
                                           for s in batch.next_state if s is not None])
        
        state_batch = torch.cat([torch.tensor(np.array([s]), device=device, dtype=torch.float32) for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # 1. 현재 Q-value 계산: Q(s_t, a_t)
        #    정책 네트워크를 통해 현재 상태(state_batch)에서 실제 선택했던 행동(action_batch)의 Q-value를 구합니다.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 2. 기대 Q-value (타겟) 계산: R_t+1 + gamma * max_a Q(s_t+1, a)
        #    다음 상태(next_state)의 최대 Q-value는 타겟 네트워크를 통해 안정적으로 추정합니다.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # 3. 손실(Loss) 계산
        #    현재 Q-value와 기대 Q-value 사이의 오차(손실)를 계산합니다.
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 4. 역전파를 통한 모델 파라미터 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # Gradient Clipping으로 학습 안정성 확보
        self.optimizer.step()
        
    def update_target_net(self):
        """
        타겟 네트워크의 가중치를 정책 네트워크의 가중치로 천천히 업데이트합니다 (Soft Update).
        이를 통해 학습 과정의 안정성을 높입니다.
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def save_model(self):
        """현재 정책 네트워크의 학습된 가중치를 지정된 경로에 파일로 저장합니다."""
        try:
            torch.save(self.policy_net.state_dict(), self.model_path)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """지정된 경로에서 저장된 가중치를 불러와 정책 네트워크와 타겟 네트워크에 적용합니다."""
        if os.path.exists(self.model_path):
            try:
                # CPU/GPU 환경에 맞게 모델을 로드합니다.
                self.policy_net.load_state_dict(torch.load(self.model_path, map_location=device))
                # 타겟 네트워크도 로드된 정책 네트워크와 동일한 가중치로 초기화합니다.
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("No saved model found. Starting from scratch.")