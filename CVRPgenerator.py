import torch
import os
import sys

# POMO 환경에서 사용할 CVRPEnv import
from CVRPEnv_battery import CVRPEnv as Env

# 상위 폴더 경로를 추가
# 현재 파일이 위치한 디렉토리에서 두 단계 상위 폴더로 이동
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# sys.path에 두 단계 상위 폴더를 추가
sys.path.append(grandparent_dir)
# 문제 설정
env_params = {
    'problem_size': 10,  # 10개의 노드를 가진 문제 설정
    'pomo_size': 10,     # POMO 수 (드론 또는 차량 수)
}

# VRP 환경 초기화
env = Env(**env_params)

# 생성할 배치 사이즈
batch_size = 100  # 한 번에 1000개의 문제 생성

# VRP 문제 생성
env.load_problems(batch_size)  # 문제 데이터 로드
reset_state, _, _ = env.reset()  # 초기 상태로 리셋

# 문제 데이터 추출
scale = 5
depot_xy = scale*reset_state.depot_xy.cpu()  # 디포의 좌표
node_xy = scale*reset_state.node_xy.cpu()    # 각 노드의 좌표
node_demand = reset_state.node_demand.cpu()  # 각 노드의 수요

# 데이터 저장 경로 설정
output_dir = './test_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 문제 데이터를 저장할 파일명
output_filename = os.path.join(output_dir, 'vrp10_test_seed0003.pt')

# 문제 데이터 딕셔너리로 저장
data_dict = {
    'depot_xy': depot_xy,
    'node_xy': node_xy,
    'node_demand': node_demand,
}

# 데이터 저장
torch.save(data_dict, output_filename)

print(f"VRP10 테스트 데이터가 {output_filename}에 저장되었습니다.")
