
import torch
import torch.nn as nn
import torch.nn.functional as F


class CVRPModel(nn.Module): #nn.module을 상속받아 정의

    def __init__(self, **model_params): #초기화 함수로 객체가 생성될 때 자동으로 호출(모델의 하이퍼파라미터를 딕셔너리 형태로 받아옴)
        super().__init__() #nn.module의 초기화 메서드를 호출하여 기본 기능을 사용할 수 있게 함
        self.model_params = model_params #전달받은 하이퍼파라미터를 저장

        self.encoder = CVRP_Encoder(**model_params) #인코더 모듈을 초기화 **는 dict로 받아온다는 뜻이며 model params가 1개가 아니라는 의미
        self.decoder = CVRP_Decoder(**model_params) #디코더 모듈을 초기화
        self.encoded_nodes = None #인코딩된 노드 정보를 저장하기 위한 변수를 초기화 초기값은 none이며 pre_forward에서 값이 채워짐
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state): #reset_state를 기반으로 모델의 인코더와 디코더를 초기화하는 역할을 함
        depot_xy = reset_state.depot_xy #reset_state에서 depot의 좌표를 가져옴
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy #reset_state에서 node_xy의 좌표를 가져옴
        # shape: (batch, problem, 2) 
        node_demand = reset_state.node_demand #reset_state에서 node_demand의 값을 가져옴
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2) # # 노드의 좌표와 수요를 결합하여 하나의 텐서로 만듬
        # shape: (batch, problem, 3)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand) #인코더를 사용하여 depot 과 node 정보를 인코딩된 벡터로 변환
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes) #인코딩된 노드 정보를 디코더에 전달하여 디코더가 이후에 예측 작업을 준비할 수 있도록 함
     

    def forward(self, state): #모델이 주어진 상태에서 다음 움직임을 결정하는 역할을 하는 클래스
        batch_size = state.BATCH_IDX.size(0) #현재 상태에서 배치의 크기를 계산
        pomo_size = state.BATCH_IDX.size(1) #현재 상태에서 POMO의 크기를 계산


        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long) #모든 차량이 출발지에서 출발함을 나타내며 selected는 0으로 표현
            prob = torch.ones(size=(batch_size, pomo_size)) # 첫번째 선택이므로 선택확률은 모두 1로 설정. 이 의미는 모든 차량이 동일한 출발지에서 출발하는 것을 나타냄

            # # Use Averaged encoded nodes for decoder input_1
            # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q1(encoded_nodes_mean)

            # # Use encoded_depot for decoder input_2
            # encoded_first_node = self.encoded_nodes[:, [0], :]
            # # shape: (batch,   1, embedding)
            # self.decoder.set_q2(encoded_first_node)

        elif state.selected_count == 1:  # Second Move, POMO #depot에서 노드로의 1번째 움직임을 의미
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size) #각 POMO의 에이전트가 처음에 다른 노드를 선택하게 함
            prob = torch.ones(size=(batch_size, pomo_size)) #prob은 동일한 확률을 부여하여 각 선택이 동일한 가중치를 가지도록 함

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node) #마지막으로 방문한 노드의 임베딩을 가져옴
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, state.load, state.soc, ninf_mask=state.ninf_mask) #디코더를 통해 각 가능한 노드에 대한 확률을 계산
            # shape: (batch, pomo, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax': #모델의 학습은 softmax 기반으로 평가
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements #확률이 0인 요소를 선택하는 버그가 있어 while 루프를 사용하여 이를 방지
                    with torch.no_grad(): ##경사 계산 비활성화 (파라미터를 변경하지 않기 떄문에 메모리 사용을 줄이기 위함)
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size) #각 드론이 각 노드로 이동할 확률을 나타내며 차원을 변형하고 복구
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size) # 선택된 노드에 대한 실제 확률 값을 담음
                    # shape: (batch, pomo)


                    # 배터리가 부족한 경우, 드론을 depot으로 복귀시킴
                    if (prob != 0).all() and (state.soc[state.BATCH_IDX, state.POMO_IDX] > 15).all():
                        break

            else:
                selected = probs.argmax(dim=2) #확률이 가장 높은 노드를 선택 / prob은 반환할 필요가 없기때문에 none을 반환
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick): #텐서에서 특정 노드를 선택하여 그 노드에 대한 임베딩 값을 가져오는 역할(선택된 노드들의 임베딩 반환)
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0) #batch size 추출
    pomo_size = node_index_to_pick.size(1) #pomo size 추출
    embedding_dim = encoded_nodes.size(2) # 임베딩 노드 추출

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim) # 각 노드 인덱스가 어떤 차원에서 선택되어야 하는지를 명시적으로 표현
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index) #각 batch와 pomo의 인덱스에 대해 선택된 노드의 임베딩값을 담음
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class CVRP_Encoder(nn.Module): #nn.module을 상속받아 정의
    def __init__(self, **model_params): #여러개의 파라미터를 딕셔너리 형태로 받아옴
        super().__init__() # 인코더의 파라미터들을 딕셔너리 형태로 받아옴 
        self.model_params = model_params 
        embedding_dim = self.model_params['embedding_dim'] #각 노드를 임베딩할 때 사용할 임베딩 벡터의 크기
        encoder_layer_num = self.model_params['encoder_layer_num'] # 인코더 내부에 쌓을 레이어의 수

        self.embedding_depot = nn.Linear(2, embedding_dim) # depot의 2차원 임베딩 벡터를 의미
        self.embedding_node = nn.Linear(3, embedding_dim) # node의 3차원 임베딩 벡터를 의미 (좌표 및 수요)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)]) #encoder layer의 수만큼 생성하여 리스트에 저장

    def forward(self, depot_xy, node_xy_demand):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy) #embedding depot은 2차원의 임베딩 벡터
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand) #embedding node는 3차원의 임베딩 벡터
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out = layer(out) #여러 레이어 겹겹히 쌓아놓은것을 통해 깊은 인코딩을 수행

        return out #인코딩 완료 결과를 반환
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim'] #임베딩 차원을 의미
        head_num = self.model_params['head_num']  #mha의 headd의 수를 의미
        qkv_dim = self.model_params['qkv_dim'] # 쿼리, 키 , value 벡터의 차원을 의미

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False) #qkv 각각 선형 변환 레이어이며 mga  에서 여러개의 병렬 벡터를 처리하기 위해 입력을 변환
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim) #여러 헤드에서 나온 결과를 결합하여 최종 임베딩 차원으로 변환

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params) #mha 이후에 정규화
        self.feed_forward = FeedForward(**model_params) #피드포워드 네트워크
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params) # 피드포워드 후 정규화
 
    def forward(self, input1):#인코더 한 층을 처리하는 과정을 구현
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num'] #multi head의 갯수

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num) #쿼리, 키, value를 mha에 맞게 재구성함
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v) #qkv의 계산을 수행
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat) #멀티헤드 어텐션의 출력으 다시 원래의 임베딩 차원 크기로 변환
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out) #mha 정규화
        out2 = self.feed_forward(out1) #피드포워드
        out3 = self.add_n_normalization_2(out1, out2) #피드포워드 정규화

        return out3 #인코더의 최종값을 도출
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module): #디코더 구현
    def __init__(self, **model_params): #디코더 파라미터를 담고 있는 딕셔너리
        super().__init__() #nn.module 상속
        self.model_params = model_params #트렌스포머 디코더에서 사용될 임베딩 크기 ,mha의 수 qkv의 차원을 담음
        embedding_dim = self.model_params['embedding_dim'] 
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim+2, head_num * qkv_dim, bias=False) #얘만 왜 embedding_dim+1이지?
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim) #멀티헤드 어텐션 출력 결합

        self.k = None  # saved key, for multi-head attention 멀티헤드 어텐션에서 다용될 key와 value를 저장하기 위한 변수
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention #단일 헤드 어텐션을 위한 key를 저장
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes): #인코더에서 나온 출력으로 각 노드의 인코딩을 포함
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num'] #mha에서 head수를 가져옴

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num) #encodes node에 대해 key 계산
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num) #encodes node에 대해 value 계산
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2) #단일헤드어텐션을 위한 key로 encodes node를 사용하여 계산
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1): #첫번째 query를 설정하기 위한 입력
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2): #두번쨰 query를 설정하기 위한 입력
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, load, soc, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num'] #headnum 수

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None], soc[:,:,None]), dim=2) #last node와 수요를 결합하여 하나의 입력으로 만듦 #decoder의 input 값
        # shape = (batch, group, EMBEDDING_DIM+2)


        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num) #input cat을 Wq선형 변환을 통해 q_last로 변환
        # shape: (batch, head_num, pomo, qkv_dim)
        
        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last # q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask) #multihead attention 수행 #불가능한 선택지에 대해 마스킹 처리
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat) #최종 출력 벡터 생성
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key) #최종 출력 벡터와 single head key의 행렬 곱셈을 수행하여 score 계산
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim'] #score을 스케일링
        logit_clipping = self.model_params['logit_clipping'] #점수를 클리핑하는데 사용

        score_scaled = score / sqrt_embedding_dim #점수가 커지는 것으 방지
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled) #점수가 커지는 것으 방지

        score_masked = score_clipped + ninf_mask #softmax 할 시 확률이 0에 가까워짐

        probs = F.softmax(score_masked, dim=2) # softmax 적용하여 probs 계산
        # shape: (batch, pomo, problem)

        return probs
    
    


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num): #mha에서 사용하는 입력 텐서qkv를 적절한 형태로 변환
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0) 
    n = qkv.size(1) #qkv.size.shape (batch_s, n(시퀀스 길이))

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed #(batch, head_num,n,key_dim)


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None): #마스크의 차원이 다른 이유는 모든 그룹에 대해 동일한 노드를 무시하느냐 그룹별로 다른 노드를 마스크하느냐에 따라 다름
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3)) #q와 k의 행렬곱을 통해 어텐션 스코어를 구함 
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float)) #QK^T/루트(k_dim)
    if rank2_ninf_mask is not None: #마스크가 주어진 경우 특정 위치의 값을 매우 작게 만듬
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled) #score에 대해 softmax 이후
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v) #v값을 내적하여 완벽한 attention score를 구함
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat #각 노드와 다른 노드와의 관계를 나타내는 확률 값이 최종적으로 decoder를 통해 도출


class AddAndInstanceNormalization(nn.Module): #두 입력 텐서를 더한 후 그 합에 대해 인스턴스 정규화
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False) #1D 인스턴스 정규화를 수행하는 파이토치 모듈을 초기화

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2 #두 입력 텐서를 더함
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2) #텐서의 차원을 바꿔줌 (batch, problem, embedding) 그 이유는 인스턴스 정규화가 마지막 차원을 기준으로 정규화를 수행하기 때문
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed) #채널별로 평균 0 표준편차를 1로 맞추는 역할을 함
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2) #다시 원래의 차원으로 돌림
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):   #두 입력 텐서를 더한 후 그 합에 대해 배치 정규화
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim'] 
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True) #임베딩 차원에 대해 배치 정규화 수행
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0) #input1.shape = (batch_s,problem_s,embedding_dim)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1))) # w1에서 linear을 통해 차원 확대 한 다음 relu 적용 후 w2로 linear를 통해 차원 축소