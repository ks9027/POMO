
import torch

import sys
import os
import logging
import logging
from tqdm import tqdm


# 상위 폴더 경로를 추가
# 현재 파일이 위치한 디렉토리에서 두 단계 상위 폴더로 이동
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# sys.path에 두 단계 상위 폴더를 추가
sys.path.append(grandparent_dir)

from CVRPEnv_battery import CVRPEnv as Env
from CVRPModel import CVRPModel as Model




from utils.utils import *


class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params): #trainer와 비교했을떄 optimizer가 없음

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params #trainer와 비교했을떄 optimizer가 없음

        # result folder, logger
        self.logger = logging.getLogger(name='trainer')
        self.result_folder = get_result_folder() #self.result_log = LogData()가 없음


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params) #trainer와 비교했을때 optimizer와 scheduler가 없음

        # Restore
        model_load = tester_params['model_load'] 
        checkpoint_fullname = '{path}/checkpoint-{epoch}_modified.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        
        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset() #시간 측정기를 초기화

        score_AM = AverageMeter() #score_am 초기화
        aug_score_AM = AverageMeter() #aug_score_am 초기화

        if self.tester_params['test_data_load']['enable']: #저장된 문제 불러오기
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes'] #테스트 에피소드수를 설정하고 에피소를 0으로 초기화
        episode = 0

        while episode < test_num_episode: #에피소드가 지정된 에피소드 수만큼 도달할때까지 테스트를 반복

            remaining = test_num_episode - episode #남은 에피소드 수를 계산하고 테스트 배치 크기를 설정
            batch_size = min(self.tester_params['test_batch_size'], remaining) # 남은 에피소드 수가 설정된 배치 크기보다 적으면 남은 에피소드만큼만 배치 크기로 설정

            score, aug_score = self._test_one_batch(batch_size) #한 배치를 호출하여 점수 반환

            score_AM.update(score, batch_size) #반환된 점수를 업데이트
            aug_score_AM.update(aug_score, batch_size) #반환된 증강 점수를 업데이트

            episode += batch_size #현재 처리한 배치 크기만큼 episode 증가

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode) #걍 로그 남기기임
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']: #데이터 증강이 활성화 되어있는지 확인
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1 #그렇지 않다면 aug_factor를 1로 설정

        # Ready
        ###############################################
        self.model.eval() #모델 평가 모드로 설정
        with torch.no_grad(): #경사하강 하지 않도록 만듬
            self.env.load_problems(batch_size, aug_factor) #주어진 배치 크기와 증강팩터를 사용해 문제 데이터를 환경에 로드
            reset_state, _, _ = self.env.reset() #환경을 초기 상태로 리셋 후 반환
            self.model.pre_forward(reset_state) #모델의 인코더를 통해 초기상태 전처리(경로에 대한 확률 설정)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step() #초기 상태 설정하고 초기 상태의 보상 완료 여부 반환
        step_count = 0
        while not done: #완료가 되지 않으면
            selected, _ = self.model(state) #현재 상태에 대해 선택된 행동을 반환(node 선택)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected) #state reward done 반환(현재 노드 인덱스, 다음 노드 인덱스, 완료 여부)
            
            step_count += 1
            if reward is not None:
                self.logger.info(f"Step {step_count} complete. Current reward: {reward.mean().item()}")
            else:
                self.logger.info(f"Step {step_count} complete. Reward is None")

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size) #보상 텐서의 크기를 재조정하여 증강 팩터, 배치 크기, POMO 크기에 따라 재정렬
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo #데이터 증강 결과 계산 후 가장 좋은 보상 선택
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value #첫 번째 증강요소(원본)에 대한 평균 보상 계산

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation #증강된 모든 데이터에서 가장 좋은 결과 선택
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value #전체 증강 데이터를 고려한 평균 보상 계산(보상이 낮을수록 좋기때문에 부호 반전)

        self.logger.info(f"Batch completed. No-AUG Score: {no_aug_score.item()}, AUG Score: {aug_score.item()}")
        return no_aug_score.item(), aug_score.item() #증강 x일떄와 증강일때의 score 값(스칼라) 반환
