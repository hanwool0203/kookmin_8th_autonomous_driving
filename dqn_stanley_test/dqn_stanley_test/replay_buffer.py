# 파일명: replay_buffer.py

import random
from collections import namedtuple, deque

# 학습에 사용할 경험 데이터의 구조를 정의합니다.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    """
    경험 리플레이(Experience Replay)를 위한 고정 크기의 메모리 버퍼.
    """
    def __init__(self, capacity):
        """
        메모리 버퍼를 초기화합니다.

        Args:
            capacity (int): 버퍼가 저장할 수 있는 최대 경험의 수.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """버퍼에 새로운 경험(transition)을 저장합니다."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """버퍼에서 지정된 배치 크기만큼의 경험을 무작위로 샘플링합니다."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """현재 버퍼에 저장된 경험의 수를 반환합니다."""
        return len(self.memory)