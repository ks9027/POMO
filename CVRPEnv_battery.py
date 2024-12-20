

from dataclasses import dataclass
import torch
import sys
import os
from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold




@dataclass #dataclass는 데이터 중심의 클래스 정의를 더 간결하고 명확하게 만들어주는 도구임
class Reset_State: #초기화 시의 상태 저장 클래스
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State: #매 단계마다의 상태 저장 클래스
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None #현재까지 남아있는 노드의 수
    load: torch.Tensor = None #현재 남아있는 적재량 (추후에 현재 남아있는 배터리량도 넣어줘야함)
    # shape: (batch, pomo)
    soc: torch.Tensor = None #현재 남아있는 배터리량
    # shape: (batch, pomo)
    current_node: torch.Tensor = None #현재 위치한 노드
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None #방문 불가능한 노드나 이미 방문한 노드 표시
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None #각 그룹의 작업이 완료되었는지 판단
    # shape: (batch, pomo)


class CVRPEnv: #환경을 설정하고 데이터를 관리하며 상태를 추적하고 업데이트하는 역할을 하는 클래스
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params #환경 매개변수 저장
        self.problem_size = env_params['problem_size'] #문제의 크기
        self.pomo_size = env_params['pomo_size'] #pomo의 크기
        self.initial_battery = 100

        self.FLAG__use_saved_problems = False #저장된 문제 데이터를 사용할지 여부(기본값은 false)
        self.saved_depot_xy = None # 얻어진 depot의 좌표를 저장할 변수
        self.saved_node_xy = None # 얻어진 고객 노드들의 좌표를 저장할 변수
        self.saved_node_demand = None #얻어진 고객 노드들의 수요를 저장할 변수
        self.saved_index = None # 얻어진 문제 데이터의 인덱스를 추적하는 변수

        # Const @Load_Problem
        ####################################
        self.batch_size = None # 배치 크기를 저장하는 변수
        self.BATCH_IDX = None # 배치 인덱스를 저장하는 텐서
        self.POMO_IDX = None # POMO 인덱스를 저장하는 텐서(에이전트의 인덱스)
        # IDX.shape: (batch, pomo) 
        self.depot_node_xy = None #depot과 node의 좌표를 포함하는 텐서
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None #depot과 node의 수요를 포함하는 텐서
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################``
        self.selected_count = None #현재까지 선택된 노드의 수를 저장하는 변수
        self.current_node = None #현재 선택된 노드를 저장하는 변수
        # shape: (batch, pomo)
        self.selected_node_list = None # 현재까지 선택된 노드들이 순서대로 저장되는 노드 리스트 변수
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None #각 에이전트가 현재 depot에 있는지 나타내는 변수
        # shape: (batch, pomo)
        self.load = None #각 에이전트가 현재 드론에 남아있는 용량을 저장하는 변수
        # shape: (batch, pomo)
        self.soc = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None #방문한 노드들에 대해 마스크를 적용하는 플래그 변수
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None #선택 불가능한 노드에 대해 마스크를 적용하는 변수
        # shape: (batch, pomo, problem+1)
        self.finished = None #각 에이전트의 여행이 완료되었는지를 나타내는 변수
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State() #환경을 초기화한 상태를 저장하기 위한 객체
        self.step_state = Step_State() # 각 단계에서의 상태를 저장하기 위한 객체

    def use_saved_problems(self, filename, device): # 이전에 생성된 문제 데이터를 로드하여 사용할 수 있도록 설정하는 역할
        self.FLAG__use_saved_problems = True # 저장된 문제 데이터를 사용할지 여부를 나타내는 변수로 기본 값은 True

        loaded_dict = torch.load(filename, map_location=device) #저장된 데이터 불러옴
        self.saved_depot_xy = loaded_dict['depot_xy'] #저장된 데이터의 디폿 값
        self.saved_node_xy = loaded_dict['node_xy'] # 저장된 데이터의 노드값
        self.saved_node_demand = loaded_dict['node_demand'] #저장된 데이터의 수요값
        self.saved_index = 0 #로드된 데이터를 사용하기 위한 인덱스 초기화(이건 depot 고정일때 다시 한번 생각해 봐야함)

    def load_problems(self, batch_size, aug_factor=1): #문제 데이터를 로드하거나 필요에 따라 데이터를 증강하는 역할을 함
        self.batch_size = batch_size #배치사이즈를 클래스 인스턴스의 self.batch_size 변수에 저장

        if not self.FLAG__use_saved_problems: # 저장된 문제 데이터를 사용할지 여부를 결정
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size) #false일 경우 새 문제 데이터 생성
        else: # true일 경우 저장된 문제 데이터 로드
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            self.saved_index = self.saved_index + batch_size # 저장된 인덱스 이후의 데이터를 사용할 수 있도록 함

        if aug_factor > 1: #데이터 증강 배율이 1보다 크면 데이터 증강 수행
            if aug_factor == 8: #8배인경우 배치 크기 8배로 설정
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else: # 1과 8 둘다 아니면 예외를 발생
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1) #depot과 node의 좌표를 결합하여 저장
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1)) #차량 기지에 대해서는 수요를 0으로 설정
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1) #차량 기지의 수요와 고객 노드들의 수요를 결합하여 총 수요를 저장
        # shape: (batch, problem+1)


        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size) #각 배치의 인덱스를 반복하여 pomo 크기만큼 확장한 인덱스 텐서 생성
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size) #pomo의 인덱스를 배치 크기만큼 확장한 인덱스 텐서를 생성

        self.reset_state.depot_xy = depot_xy #depot.xy를 reset_state 객체에 저장
        self.reset_state.node_xy = node_xy #node.xy를 reset_state 객체에 저장
        self.reset_state.node_demand = node_demand #node_demand를 reset_state 객체에 저장

        self.step_state.BATCH_IDX = self.BATCH_IDX #이전에 생성한 batch_idx를 step_state개체의 batch_idx 속성에 저장
        self.step_state.POMO_IDX = self.POMO_IDX #이전에 생성한 pomo_idx를 step_state개체의 pomo_idx 속성에 저장
        
        # batch : 여러 문제 인스턴스를 동시에 처리하여 학습 속도를 높임
        # POMO :각 문제 인스턴스에 대해 다양한 경로를 생성하여 더 나은 솔루션을 찾도록 도와줌

    def reset(self): #환경 상태를 초기화하여 새로운 에피소드를 시작할 준비를 함
        self.selected_count = 0 # 선택된 노드의 개수를 0으로 초기화
        self.current_node = None # 현재 선택된 노드를 none으로 설정하여 아직 어떤 노드도 설정되지 않았음을 나타냄
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long) #선택된 노드들의 리스트를 저장하는 역할을 하며 초기화
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool) #모든 pomo의 위치가 depot에 있음을 나타냄
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size)) # 모든 pomo가 가지고 있는 용량을 나타냄
        # shape: (batch, pomo)
        self.soc = torch.ones(size=(self.batch_size, self.pomo_size)) * (self.initial_battery-5)  # 초기 배터리 잔량을 95로 설정 (예시)
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)) #각 노드가 방문되었는지 여부를 나타내며 모든 요소가 0으로 초기화
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)) #선택할 수 없는 노드에 대해 마스킴하며 모든 요소가 0으로 초기화
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool) # 각 POMO가 작업을 완료했는지 여부를 나타내며 모든 요소가 false로 초기화
        # shape: (batch, pomo)

        reward = None #초기에는 보상이 없으므로 none
        done = False #에피소드가 완료되지 않았으므로 false
        return self.reset_state, reward, done # 초기화된 상태의 보상과 완료 여부를 반환

    def pre_step(self): #reset_state를 step_state에 저장하고 다음 단계를 수행하기 전 상태를 업데이트하도록 준비하는 역할
        self.step_state.selected_count = self.selected_count # 현재까지 선택된 노드의 개수를 나타냄
        self.step_state.load = self.load # 각 POMO가 현재 가지고 있는 용량을 나타냄
        self.step_state.soc = self.soc # 각 드론의 배터리 잔량 업데이트
        self.step_state.current_node = self.current_node # 현재 선택된 노드를 나타냄
        self.step_state.ninf_mask = self.ninf_mask # 선택할 수 없는 노드를 마스킹하기 위한 텐서
        self.step_state.finished = self.finished # 각 POMO가 작업을 완료했는지 여부를 나타냄

        reward = None #보상을 none으로 초기화
        done = False #에피소드가 완료되지 않았음을 표현
        return self.step_state, reward, done #현재 상태 보상 완료상태를 반환

    def step(self, selected): 
        # selected.shape: (batch, pomo)

        # Dynamic-1: 노드 선택과 업데이트
        self.selected_count += 1  # 현재까지 선택된 노드의 개수를 증가
        self.current_node = selected  # 현재 POMO들이 위치한 노드 저장
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)  # 선택한 노드 기록

        # Dynamic-2: 적재량과 배터리 업데이트
        self.at_the_depot = (selected == 0)  # 각 POMO가 현재 depot에 있는지 확인
        # 수요 리스트 가져오기 및 선택된 노드의 수요 추출
        
        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)  # 각 노드의 수요 저장
        gathering_index = selected[:, :, None]  # 선택된 노드의 인덱스 가져오기
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)  # 선택된 노드의 수요 가져오기

        batch_size, pomo_size = self.current_node.shape  # (batch, pomo)


        
        # 이전 노드를 가져올 수 있는지 확인
        if self.selected_node_list.shape[2] > 1:
            # 이전에 선택된 노드 인덱스를 가져옴
            previous_node = self.selected_node_list[:, :, -2].unsqueeze(-1).expand(batch_size, pomo_size, 2)  # (batch, pomo, 2)
        else:
            # 첫 스텝에서는 이전 노드가 없으므로 depot 좌표로 설정
            previous_node = torch.zeros(batch_size, pomo_size, 2, device=self.depot_node_xy.device, dtype=torch.long)

        # 이전에 선택된 노드의 좌표 가져오기
        previous_node_xy = self.depot_node_xy.gather(1, previous_node)  # (batch, pomo, 2)

        # 현재 노드 좌표 가져오기
        current_node = self.current_node.unsqueeze(-1).expand(batch_size, pomo_size, 2)  # (batch, pomo, 2)
        current_node_xy = self.depot_node_xy.gather(1, current_node)  # (batch, pomo, 2)

        # 거리 계산
        distance = torch.sqrt(((current_node_xy - previous_node_xy) ** 2).sum(dim=-1))  # (batch, pomo)


        # 배터리 소모 계산 (이전 노드 -> 현재 노드 이동)
        soc_consumption = self.calculate_soc(self.load, distance)  # POMO 차원 포함
        self.soc = self.soc - soc_consumption

        
    
        
        
        # Load 업데이트 (방문한 노드의 수요만큼 감소)
        self.load = self.load - selected_demand
        self.load[self.at_the_depot] = 1  # depot에 있는 드론은 적재량을 다시 채움
        # 바로 이전 노드와 현재 노드 간의 좌표를 가져올 때 gather 사용
    
        # 방문한 노드 마스킹 처리
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot은 항상 선택 가능하도록 유지




        # 선택 불가능한 노드 마스킹
        self.load_ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        self.load_ninf_mask[demand_too_large] = float('-inf')
        self.load_ninf_mask[:, :, 0][~self.at_the_depot] = 0  # depot은 항상 선택 가능하도록 유지



        
        self.ninf_mask = self.load_ninf_mask.clone()
        


        for pomo_idx in range(self.pomo_size):  # 각 POMO의 드론에 대해
            # DEPOT에 있는 드론들은 배터리를 95로 충전

            self.soc[self.at_the_depot] = 95

            for node in range(1, self.problem_size + 1):  # depot(0)을 제외한 각 노드에 대해
                # 현재 위치에서 특정 노드를 거쳐 depot까지 가는 배터리 소모량 계산
                total_battery_needed = self.node_to_depot(pomo_idx, node)
                

                

                # 배터리 잔량이 15 이상 남는지 확인
                soc_too_large = self.soc[:, pomo_idx].unsqueeze(-1) < total_battery_needed + 15

                
                
                # soc_too_large가 True인 경우에만 마스킹 적용
                for batch_idx in range(self.batch_size):# soc_too_large가 True인 경우에만 마스킹 적용
                    if soc_too_large[batch_idx, pomo_idx].any():
                        self.ninf_mask[batch_idx, pomo_idx, node] = float('-inf')

    
        self.ninf_mask[:, :, 0][~self.at_the_depot] = 0  # depot은 항상 선택 가능하도록 유지# 배터리 부족으로 선택 불가로 마스킹                 
        # 완료 여부 체크
        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished = self.finished + newly_finished

        # depot은 완료된 이후에도 선택 가능하도록 설정 
        self.ninf_mask[:, :, 0][self.finished] = 0

        # 상태 업데이트
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.soc = self.soc  # SOC 업데이트
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # 완료 여부 반환
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # 이동 거리 계산 후 보상 부여 (음수 값)
        else:
            reward = None

        return self.step_state, reward, done


    
    def calculate_soc(self, payload, distances):
        alpha = 2.29705
        beta = 3.87886

        # BCR 계산 후 소수점 오류를 줄이기 위해 반올림
        bcr = torch.round((alpha * payload + beta) * 1e6) / 1e6  # 소수점 6자리까지 반올림


        soc_consumption = bcr * distances
        
        return soc_consumption
        
    #def calculate_travel_time(self, current_node):
        # 노드 간의 거리와 속도에 따라 시간을 계산합니다.
    #   distances = self._get_travel_distance()
    #  time_per_node = distances / 22.0  # 속도 22km/h로 고정
    #    return time_per_node
    
    def node_to_depot(self, pomo_idx, node_idx):
        # 현재 노드 인덱스 추출 (batch 크기와 맞음)
        current_node_idx = self.current_node[:, pomo_idx].unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
        
        # 현재 노드의 좌표를 얻기 위한 gather 사용 (batch, pomo, 2)
        current_node_xy = self.depot_node_xy.gather(1, current_node_idx.expand(-1, 1, 2)).squeeze(1)  # (batch, 2)

        # 특정 노드 좌표 가져오기 (node_idx는 단일 노드를 의미)
        target_node_xy = self.depot_node_xy[:, node_idx, :]  # (batch, 2)

        # 현재 노드에서 특정 노드까지 거리 계산 (유클리드 거리)
        distance_to_node = torch.sqrt(((current_node_xy - target_node_xy) ** 2).sum(dim=-1))  # (batch,)

        # 특정 노드에서 depot으로 돌아가는 거리 계산
        depot_xy = self.depot_node_xy[:, 0, :]  # depot의 좌표 (batch, 2)
        distance_to_depot = torch.sqrt(((target_node_xy - depot_xy) ** 2).sum(dim=-1))  # (batch,)

        # 노드까지 소모되는 배터리 양 계산
        battery_consumption_to_node = self.calculate_soc(self.load[:, pomo_idx], distance_to_node)  # (batch,)

        # 특정 노드에서 수요 처리 후 남은 적재량 계산
        remaining_load = self.load[:, pomo_idx] - self.depot_node_demand[:, node_idx]  # (batch,)
        
        # 남은 적재량으로 depot까지 이동할 때 소모되는 배터리 양 계산
        battery_consumption_to_depot = self.calculate_soc(remaining_load, distance_to_depot)  # (batch,)

        # 총 배터리 소모량 계산
        total_battery_needed = battery_consumption_to_node + battery_consumption_to_depot  # (batch,)
        
        return total_battery_needed




    
    def calculate_distance_to_depot(self, selected):
        # 선택된 노드에서 디팟까지의 거리를 계산하는 함수
        depot_xy = self.depot_node_xy[:, 0, :].unsqueeze(1).expand(self.batch_size, selected.size(1), -1)  # (batch, pomo, 2)로 확장
        node_xy = self.depot_node_xy[self.BATCH_IDX, selected]  # 선택된 노드의 좌표

        distance_to_depot = torch.sqrt(((node_xy - depot_xy) ** 2).sum(dim=-1))  # 유클리드 거리 계산
        return distance_to_depot

    def _get_segment_distances(self):
            gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
            # shape: (batch, pomo, selected_list_length, 2)
    
            all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            # shape: (batch, pomo, problem+1, 2)

            ordered_seq = all_xy.gather(dim=2, index=gathering_index)
            # 선택된 노드들의 좌표를 순서대로 나열
            # shape: (batch, pomo, selected_list_length, 2)

            rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
            # ordered_seq를 한 칸씩 이동하여 각 구간의 거리를 계산하기 위한 준비
            # shape: (batch, pomo, selected_list_length, 2)

            segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
            # 각 구간의 유클리드 거리 계산
            # shape: (batch, pomo, selected_list_length)

            return segment_lengths
        
    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2) 
        # 마지막에 2인 이유는 (x,y)의 노드 인덱스를 가져오기 위함임
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1) 
        #각 배치와 pomo인덱스에 대해 모든 노드의 좌표를 반복하여 확장한 텐서
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index) 
        #gathering_index를 사용하여 all xy 텐서에서 각 노드의 좌표를 추출하며 이 코드는 선택된 노드들의 좌표를 순서대로 나열하겠다는 의미
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1) 
        #ordered_seq을 한 칸씩 왼쪽으로 이동시킨 텐서로 마지막 노드는 처음으로 이동
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt() 
        #ordered_seq과 rolled_seq사이의 차이를 구하고 이를 제곱한 후 마지막 차원에서 제곱하여 마지막 차원에서 합산하여 각 구간의 거리의 제곱을 계산
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2) #각 구간의 거리를 합산하여 총 이동 거리를 계산
        # shape: (batch, pomo)
        
        return travel_distances

