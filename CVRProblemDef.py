
import subprocess  # subprocess 모듈을 가져옴
import sys

# torch 모듈을 임시로 설치하는 코드
try:
    import torch
except ImportError:
    print("Torch is not installed. Installing torch...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
    import torch




def get_random_problems(batch_size, problem_size, scale=5): # size를 입력으로 받아 무작위로 생성된 경로 문제를 반환
    #batch_size = 동시에 생성할 문제의 개수
    #problem_size = 문제 내의 고객 노드의 수(방문해야 할 위치의 수)

    depot_xy = torch.full((batch_size, 1, 2), 0.5) * scale # 나중에 depot 고정해야하니까 잊지 말기!!!
    # shape: (batch, 1, 2) # (문제의 개수, depot의 개수, 2차원)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))*scale 
    # shape: (batch, problem, 2)  # (문제의 개수, node의 개수, 2차원)

    node_demand = torch.rand(size=(batch_size, problem_size)) * 0.3
    
    return depot_xy, node_xy, node_demand



def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2) # 입력된 좌표 데이터를 8배로 증강하여 반환
    # N은 각 문제에 포함된 노드의 수를 의미

    x = xy_data[:, :, [0]] #모든 (x,y) 좌표중 x값만 뽑아냄
    y = xy_data[:, :, [1]] #모든 (x,y) 좌표중 y값만 뽑아냄
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2) #원본 데이터
    dat2 = torch.cat((1 - x, y), dim=2) # x축 기준 좌표 반전
    dat3 = torch.cat((x, 1 - y), dim=2) # y축 기준 좌표 반전
    dat4 = torch.cat((1 - x, 1 - y), dim=2) # 원점 기준 좌표 반전
    dat5 = torch.cat((y, x), dim=2) # y=x 직선에 대한 반전
    dat6 = torch.cat((1 - y, x), dim=2) # y=-x 직선에 대한 반전
    dat7 = torch.cat((y, 1 - x), dim=2) # x 축 반전 후 y= x에 대해 반전
    dat8 = torch.cat((1 - y, 1 - x), dim=2) # 원점 기준 좌표 반전 후 y = x 직선에 대해 반전

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)
    # 한 데이터로 8개의 데이터를 만듦으로 총 데이터의 shape 은 (8*B, N, 2)
 
    return aug_xy_data
