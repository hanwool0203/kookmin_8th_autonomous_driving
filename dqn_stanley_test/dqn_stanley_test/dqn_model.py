# 파일명: dqn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    DQN 에이전트를 위한 신경망 모델.
    """
    def __init__(self, n_observations, n_actions):
        """
        신경망의 계층(layer)들을 초기화합니다.

        Args:
            n_observations (int): 상태 공간의 차원 (입력 뉴런 수).
            n_actions (int): 행동 공간의 크기 (출력 뉴런 수).
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        모델의 순전파 로직을 정의합니다.
        상태(x)를 입력받아 각 행동의 Q-value를 계산하여 반환합니다.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)