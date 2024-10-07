from dataclasses import dataclass
import torch
import sys
import os
import logging

# 상위 폴더 경로를 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold
logging.basicConfig(level=logging.DEBUG)

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    node_xy: torch.Tensor = None
    node_demand: torch.Tensor = None

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    selected_count: int = None
    load: torch.Tensor = None
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None

class CVRPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        self.FLAG__use_saved_problems = False
        self.batch_size = 10
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)

        # 로그 추가
        logging.debug(f"Selected node list after reset: {self.selected_node_list}")

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected

        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)
        logging.debug(f"Selected node list during step: {self.selected_node_list}")
        reward = torch.sum(self.selected_node_list.float())  # 예시: selected_node_list의 합을 보상으로 가정
        done = self.selected_count >= self.problem_size  # 예시: problem_size만큼 스텝을 밟으면 완료로 가정
        return self.selected_node_list, reward, done

if __name__ == '__main__':
    env_params = {
        'problem_size': 10,
        'pomo_size': 5
    }
    env = CVRPEnv(**env_params)
    env.reset()

    # 예시: selected 값을 랜덤으로 설정해 테스트
    selected = torch.randint(0, 10, (env.batch_size, env.pomo_size), dtype=torch.long)
    step_state, reward, done = env.step(selected)

    # 출력 추가
    logging.debug(f"Selected nodes: {selected}")
    logging.debug(f"Step state: {step_state}")
    logging.debug(f"Reward: {reward}")
    logging.debug(f"Done: {done}")
