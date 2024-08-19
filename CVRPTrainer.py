
import torch
from logging import getLogger #코드 실행 과정에서 발생하는 정보를 기록하는데 사용

from CVRPEnv import CVRPEnv as Env #CVRP 환경 클래스 임포트
from CVRPModel import CVRPModel as Model #CVRP 모델 클래스 임포트

from torch.optim import Adam as Optimizer #adam 임포트
from torch.optim.lr_scheduler import MultiStepLR as Scheduler #학습률 조정 스케쥴러 임포트

from utils.utils import * #유틸 임포트


class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params #생성자로부터 받은 매개변수들을 클래스 내부 변수로 저장
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer') #로깅도구를 초기화하며 초거를 trainer로 설정
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params) #cvrp 모델 인스턴스 생성
        self.env = Env(**self.env_params) #cvrp 환경 인스턴스 생성
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer']) #모델 가중치를 업데이트하는 최적화기를 초기화
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler']) #학습률 관리하는 스케줄러 초기화

        # Restore
        self.start_epoch = 1 #학습의 시작 에포크를 1로 초기화
        model_load = trainer_params['model_load'] #학습을 재개할 때 사용했던 파라미터를 가져옴
        if model_load['enable']: #모델 로드가 활성화되어있으면 저장된 모델을 불러옴
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load) #불러올 체크포인트 파일의 전체 경로를 설정
            checkpoint = torch.load(checkpoint_fullname, map_location=device) #지정된 경로에 있는 체크포인트 파일을 불러옴
            self.model.load_state_dict(checkpoint['model_state_dict']) #모델의 가중치를 로드하여 적용
            self.start_epoch = 1 + model_load['epoch'] #학습 중단 지점부터 에포크값 업데이트
            self.result_log.set_raw_data(checkpoint['result_log']) #로그를 로드하여 현재 로그에 적용
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #옵티마이저(adam)의 상태를 불러옴
            self.scheduler.last_epoch = model_load['epoch']-1 #학습률 스케쥴러의 마지막 에포크를 설정
            self.logger.info('Saved Model Loaded !!') #로드가 완료되었음을 로그에 기록

        # utility
        self.time_estimator = TimeEstimator() #학습 시간추정을 위한 도구를 초기화

    def run(self):
        self.time_estimator.reset(self.start_epoch) #학습 시간 추정을 위해 객체 상태를 초기화
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1): #학습이 시작되는 루프
            self.logger.info('=================================================================') #에포크의 시작을 위해 구분선 추가

            # LR Decay
            self.scheduler.step() #학습률 스케줄러를 한 단계 진행 #스케줄러는 자동으로 관리하며 에포크에 맞게 학습률 조정

            # Train
            train_score, train_loss = self._train_one_epoch(epoch) #한 에포크에 대한 학습을 수행
            self.result_log.append('train_score', epoch, train_score) #train score에 대한 로그 기록
            self.result_log.append('train_loss', epoch, train_loss) #train loss에 대한 로그 기록

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            #현재 에포크까지 소요된 시간과 남은 시간의 추정치를 문자열 형식으로 반환
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))
            # 학습 로그를 남기는 코드

            all_done = (epoch == self.trainer_params['epochs']) #학습이 완료되었는지 표시
            model_save_interval = self.trainer_params['logging']['model_save_interval'] # 몇 번째 에포크마다 모델을 저장할지에 대한 간격 정의
            img_save_interval = self.trainer_params['logging']['img_save_interval'] # 몇 번째 에포크마다 이미지 로그를 저장할지에 대한 간격 정의

            # Save latest images, every epoch
            if epoch > 1: #에포크가 1보다 큰 경우에만 로그 이미지 저장
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0: #모든 에포크가 완료되거나 model save interval에 따라 학습 중간 결과 모델 저장
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # Save Image #에포크의 시각화한 이미지를 저장
            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # All-done announcement #학습 완료 후 로그 출력
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter() #에포크 동안의 평균 점수
        loss_AM = AverageMeter() #에포크 동안의 평균 손실

        train_num_episode = self.trainer_params['train_episodes'] #현재 에포크에서 학습할 에피소드의 수를 나타냄
        episode = 0 #현재까지 처리한 에피소드 수
        loop_cnt = 0 #반복 횟수 추적
        while episode < train_num_episode: #에피소드 수가 train_num_episode에 도달할 때까지 계속됨

            remaining = train_num_episode - episode #현재 남아있는 에피소드 수 
            batch_size = min(self.trainer_params['train_batch_size'], remaining) 
            # self.trainer_params['train_batch_size']가 설정된 최대 배치 크기이고, 남아있는 에피소드 수가 그보다 작다면 남은 에피소드 수만큼 처리

            avg_score, avg_loss = self._train_one_batch(batch_size) #주어진 배치 크기만큼의 데이터를 사용해 모델을 학습하며 이 배치에서 얻어진 점수를 score_AM에 추가
            score_AM.update(avg_score, batch_size) #현재 배치에서 얻은 평균 점수를 추가하여 에포크 전체 평균 점수 업데이트
            loss_AM.update(avg_loss, batch_size) #현재 배치에서 얻은 평균 손실를 추가하여 에포크 전체 평균 손실 업데이트

            episode += batch_size #현재 에피소드 수를 업데이트 함

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch: #첫 에포크인지 확인
                loop_cnt += 1                # 첫번째 에포크 동안 몇 개의 배치가 처리되었는지 카운트
                if loop_cnt <= 10: # 첫 10개의 배치만 기록
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}' #각 에포크가 끝날 때마다 한 번씩 로그를 기록
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size): #배치 단위로 모델을 학습시키는 과정을 구현

        # Prep
        ###############################################
        self.model.train() #모델을 학습 모드로 전환
        self.env.load_problems(batch_size) #현재 배치 크기만큼의 문제를 환경에 로드
        reset_state, _, _ = self.env.reset() #환경 초기화
        self.model.pre_forward(reset_state) #초기 상태를 모델의 pre forward 메서드에 전달

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0)) #배치와 pomo의 크기에 대한 텐서를 초기화(확률 리스트를 초기화)
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step() #pre step을 호출하여 초기 상태를 가져오고 state reward done 변수에 저장

        while not done:
            selected, prob = self.model(state) #현재 상태에 대해 모델의 예측을 수행하고, 선택된 노드와 해당 확률을 반환
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected) #환경 상태를 업데이트하고 새로운 상태 보상 종료 여부를 반환
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2) # 각 단계에서의 확률을 추가

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True) #advantage는 현재 보상에서 평균 보상을 뺀 값을 의미(특정 상태에서의 행동이 얼마나 좋은지 또는 나쁜지 표현)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2) #각 확률에대해 로그를 취한뒤 합산하여 log_prob계산
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD #advantage와 log prob의 곱에 마이너스 기호를 붙여 손실 계산
        # shape: (batch, pomo)
        loss_mean = loss.mean() #평균 손실 계산

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo #pono의 여러 선택중 최대 보상을 취한다. 최상의 경로 선택을 의미
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value #최대 보상의 평균을 구해 score_mean 계산

        # Step & Return
        ###############################################
        self.model.zero_grad() #모델의 모든 경사도를 0으로 초기화
        loss_mean.backward() #손실에 대한 역전파 수행해서 경사도 계산
        self.optimizer.step() #옵티마이저를 이용하여 모델의 가중치 업데이트
        return score_mean.item(), loss_mean.item() #score_mean과 loss_mean 반환