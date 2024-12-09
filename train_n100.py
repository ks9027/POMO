##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = False
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # CVRProblemDef 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))  # utils 경로 추가


os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def  
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CVRPTrainer import CVRPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'problem_size':10,
    'pomo_size': 10,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-5,
        'weight_decay': 1e-4
    },
    'scheduler': {
        'milestones': [4900, 5000],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 50,
    'train_episodes': 128,
    'train_batch_size': 128,
    'prev_model_path': None,
    'logging': {
        'model_save_interval': 500,
        'img_save_interval': 500,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_cvrp_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_CVRP20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 2000,  # epoch version of pre-trained model to laod.

    }
}

logger_params = {
    'log_file': {
        'desc': 'train_cvrp_n100_with_instNorm',
        'filename': 'run_log'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 10
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 5


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
