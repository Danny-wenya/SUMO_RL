import numpy as np
import warnings
import matplotlib.pyplot as plt
from collections import namedtuple
from random import randint
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import traci
import random
import pickle
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


import os
import sys
import traci
from sumolib import checkBinary
import numpy as np

warnings.filterwarnings("ignore")

# 哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈
class env:
    def __init__(self):
        # 环境变量
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        # 输出文件
        self.trip_filename= './Simulation/output.xml'
        self.tl_cycle =180
        self.time_step=1


    # 提取交叉路口+道路+检测器对应关系
    def get_tl_control_lane(self,tl_list):
        road_detector = {}
        tl_road = {}

        for detector in traci.lanearea.getIDList():
            lane = traci.lanearea.getLaneID(detector)
            road = lane.split('_')[0]

            if road not in road_detector:
                road_detector[road]=[]
            road_detector[road].append(detector)
        
        for tl in tl_list:
            tl_road[tl] = {}
            control_lane = traci.trafficlight.getControlledLanes(tl)
            control_road = list (x.split('_')[0] for x in control_lane)

            for road in control_road:
                if road not in tl_road:
                    tl_road[tl][road]=road_detector[road]

        del road_detector,tl_list

        return tl_road


    # step一次完成一次信号设置和执行，执行时间为180秒。
    def run_cycle(self):
        # 设置信号灯
        # 初始化
        step = 0 
        num_vhs = {} #临时数据
        wait_time = {} #车辆的等待时间
        veh_speed = {}
        WT=[]
        SP=[]
        VH=[]
        tl_list=traci.trafficlight.getIDList()
        tl_data = self.get_tl_control_lane(tl_list) 

        for tl in tl_data:
            num_vhs[tl]={}
            wait_time[tl] = {}
            veh_speed[tl] = {}

            for road in tl_data[tl]:
                num_vhs[tl].update({road:{}})
                wait_time[tl].update({road:{}})
                veh_speed[tl].update({road:{}})

                for detector in tl_data[tl][road]:
                    num_vhs[tl][road].update({detector:[]})
                    wait_time[tl][road].update({detector:[]})
                    veh_speed[tl][road].update({detector:[]})


        while traci.simulation.getMinExpectedNumber()>0:
            traci.simulationStep()
            step+=1
            for tl in tl_data:
                for road in tl_data[tl]:
                    for detector in tl_data[tl][road]:
                        veh_ids = traci.lanearea.getLastStepVehicleIDs(detector) 
                        for veh in veh_ids:
                            WT.append(traci.vehicle.getAccumulatedWaitingTime(veh))
                            SP.append(traci.vehicle.getSpeed(veh))

                        if step%180==0 and step!=0:
                            if WT==[]:
                                wait_time[tl][road][detector].append(0)
                            else:
                                wait_time[tl][road][detector].append(np.mean(WT))

                            if SP==[]:
                                veh_speed[tl][road][detector].append(0)
                            else:
                                veh_speed[tl][road][detector].append(np.mean(SP))

                            num_vhs[tl][road][detector].append(len(veh_ids))
                            WT=[]
                            SP=[]

            if step==self.tl_cycle:
                break

        reward=[]
        for tl in tl_list:
            for road in wait_time[tl]:
                for it in wait_time[tl][road].values():
                    reward.extend(it)
        reward=-np.mean(np.delete(reward,np.where(np.isnan(reward)))) #性能越好等待时间越小
        
        # 每条路的平均等待时间,每条路的平均速度，每条路的总车流量，每条路的通行时间
        cycle_data={}
        for tl in tl_data:
            cycle_data[tl]=[]
            for road in tl_data[tl]:
                for detector in tl_data[tl][road]:
                    WT=wait_time[tl][road][detector]
                    SP=veh_speed[tl][road][detector]
                    VH=num_vhs[tl][road][detector]
                    cycle_data[tl].append(WT+SP+VH)

        del num_vhs,wait_time,veh_speed,tl_list,tl_data,WT,SP,VH
            
        return list(cycle_data.values())[0],np.array([reward])
    

    # 启动SUMO
    def init_simu(self):
        sumoBinary  = checkBinary('sumo')  
        sumoCmd = [sumoBinary,'-c','./Simulation/shenzhen12.sumocfg',"--step-length="+str(self.time_step),
                "--collision.action=warn",'--time-to-teleport=60',"--no-step-log", "true"]
        traci.start(sumoCmd)

        action_index=0
        dua_list=[70,14,70,14] 
        a,b,c,d=dua_list
        time_line=[a for _ in range(5)]+[a+b,c,c,c+d]+[a for _ in range(4)]+[a+b,a+b,c,c,c+d]
        cycle_data,reward=self.run_cycle()

        if len(np.unique(cycle_data))==0:
            cycle_data=None
        else:
            cycle_data=np.concatenate([np.array([time_line]),np.array(cycle_data).T],axis=0).T 

        del a,b,c,d,time_line

        return cycle_data,reward,action_index,dua_list


    def step(self,dua_list):
        cycle_data,reward=self.run_cycle()
        a,b,c,d=dua_list
        time_line=[a for _ in range(5)]+[a+b,c,c,c+d]+[a for _ in range(4)]+[a+b,a+b,c,c,c+d]
        
        if len(np.unique(cycle_data))==0:
            cycle_data=None
        else:
            cycle_data=np.concatenate([np.array([time_line]),np.array(cycle_data).T],axis=0).T 

        del a,b,c,d,dua_list,time_line
            
        return cycle_data,reward


# TD误差存储类
class TDErrorMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, td_error):

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def get_prioritized_indexes(self, batch_size):
        '''根据TD误差以概率获得index'''

        # 计算TD误差总和
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)  # 添加一个微小值

        # 为batch_size 生成随机数并且按照升序排列
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (
                        abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1

            # 使用微小值进行计算超出内存
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)

        del sum_absolute_td_error,rand_list,idx,tmp_sum_absolute_td_error

        return indexes

    def update_td_error(self, updated_td_errors):
        self.memory = updated_td_errors

    def save(self,episode):
        with open(f'./td_error_memory_{episode}','wb') as f:
            pickle.dump(self.memory,f)


# 用于存储经验的内存类
class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # memory的最大长度
        self.memory = []  # 存储过往经验
        self.index = 0  # 表示要保存的数据

    def push(self, state, action, state_next, reward,dua_list):
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 内存未满时添加

        self.memory[self.index] = Transition(state, action, state_next, reward,dua_list)

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        # 随机检索batch_size大小的样本返回
        return random.sample(self.memory, batch_size)
    
    def save(self,episode):
        with open(f'./simulation_data_{episode}.pkl','wb') as f:
            pickle.dump(self.memory,f)

    def __len__(self):
        # 返回当前memory的长度
        return len(self.memory)
    
    

class Brain:
    def __init__(self, net_type,params_dict):
        # 获取CartPole的两个动作（向左或向右）
        self.num_actions = params_dict["num_actions"]

        # 创建存储经验的对象
        self.memory = ReplayMemory(CAPACITY)

        # 主Q网络  目标Q网络
        if net_type==0:
            self.main_q_network = self.SelfAttentionNet(params_dict)
            self.target_q_network = self.SelfAttentionNet(params_dict)
        elif net_type==1:
            self.main_q_network = self.MultiheadAttentionNet(params_dict)
            self.target_q_network = self.MultiheadAttentionNet(params_dict)
        elif net_type==2:
            self.main_q_network = self.TransformerEncoderNet(params_dict)
            self.target_q_network = self.TransformerEncoderNet(params_dict)

        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=LEARNING_RATE)
        self.td_error_memory = TDErrorMemory(CAPACITY)

        # 动作空间
        self.action_space=[[0, 0, 0, 0],[-5, 0, 0, 5],[-5, 0, 5, 0],[-5, 5, 0, 0],[0, -5, 0, 5],
                      [0, -5, 5, 0],[0, 0, -5, 5],[0, 0, 5, -5],[0, 5, -5, 0],[0, 5, 0, -5],
                      [5, -5, 0, 0],[5, 0, -5, 0],[5, 0, 0, -5],[-5, -5, 5, 5],[-5, 5, -5, 5],
                      [-5, 5, 5, -5],[5, -5, -5, 5],[5, -5, 5, -5],[5, 5, -5, -5]]

        


    def replay(self, episode,step):
        '''通过经验回放学习参数'''

        # 若经验池大小小于小批量数据时，不执行任何操作
        if len(self.memory) < BATCH_SIZE:
            return

        #  创建小批量数据
        self.batch, self.state_batch, self.action_batch, self.reward_batch, \
        self.non_final_next_states = self.make_mini_batch(episode)

        # 求Q(s_t,a_t)作为监督信息
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 更新参数
        self.update_main_q_network(episode,step)

    def save_model(self,episode):
        torch.save(self.main_q_network.state_dict(), f'./models/model_{episode}.pt')

    def load_model(self,load_model_path):
        self.main_q_network.load_state_dict(torch.load(load_model_path))


    class MultiheadAttentionNet(nn.Module):
        def __init__(self, params_dict):
            super().__init__()
            self.query = torch.nn.Linear(params_dict["input_dim"], params_dict["hidden_dim"]).to(device)
            self.key = torch.nn.Linear(params_dict["input_dim"], params_dict["hidden_dim"]).to(device)
            self.value = torch.nn.Linear(params_dict["input_dim"], params_dict["hidden_dim"]).to(device)
            # Define the multi-head attention module
            self.multihead_attention = nn.MultiheadAttention(embed_dim=params_dict["hidden_dim"], num_heads=params_dict["n_heads"],batch_first=True).to(device)

            self.fc3_adv = nn.Linear(int(params_dict["seq_length"]*params_dict["hidden_dim"]),params_dict["num_actions"]).to(device)  
            self.fc3_v = nn.Linear(int(params_dict["seq_length"]*params_dict["hidden_dim"]), 1).to(device)  


        def forward(self, x):
            x=x.to(device)
            query = self.query(x)  # [batch_size,18,hidden_size]
            key = self.key(x)      # [batch_size,18,hidden_size]
            value = self.value(x)  # [batch_size,18,hidden_size]
            y, attention_scores = self.multihead_attention(query, key, value)
            y_=y.contiguous().view(y.shape[0],-1) # [batch_size,18*hidden_size]

            adv = self.fc3_adv(y_) # [batch_size,action_nums]
            val = self.fc3_v(y_)   #.expand(-1, adv.size(1)) # [batch_size,1]  
            output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

            del x,query,key,value,y,y_,adv,val

            return output



    class SelfAttentionNet(nn.Module):
        def __init__(self, params_dict):
            super().__init__()
            self.query = torch.nn.Linear(params_dict["input_dim"], params_dict["hidden_dim"]).to(device)
            self.key = torch.nn.Linear(params_dict["input_dim"], params_dict["hidden_dim"]).to(device)
            self.value = torch.nn.Linear(params_dict["input_dim"], params_dict["hidden_dim"]).to(device)

            self.fc3_adv = nn.Linear(int(params_dict["seq_length"]*params_dict["hidden_dim"]),params_dict["num_actions"]).to(device)  
            self.fc3_v = nn.Linear(int(params_dict["seq_length"]*params_dict["hidden_dim"]), 1).to(device)  

        def forward(self, x):

            x=x.to(device) 
            query = self.query(x)  # [batch_size,18,hidden_size]
            key = self.key(x)      # [batch_size,18,hidden_size]
            value = self.value(x)  # [batch_size,18,hidden_size]
            
            scores = torch.matmul(query, key.transpose(-2, -1)) # [batch_size,18,18]
            attention_weights = F.softmax(scores, dim=-1)
            y = torch.matmul(attention_weights, value)  # [batch_size,18,hidden_size]
            y_=y.contiguous().view(y.shape[0],-1) # [batch_size,18*hidden_size]

            adv = self.fc3_adv(y_) # [batch_size,action_nums]
            val = self.fc3_v(y_)   #.expand(-1, adv.size(1)) # [batch_size,1]  动作优势的维度？？？
            output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

            del x,query,key,value,scores,attention_weights,y,y_,adv,val

            return output
        

    class TransformerEncoderNet(nn.Module):
        def __init__(self, params_dict):
            super().__init__()
            self.linear = nn.Linear(params_dict["input_dim"], params_dict["d_model"]).to(device)
            self.encoder_layers = nn.TransformerEncoderLayer(d_model=params_dict["d_model"], nhead=params_dict["n_head"]).to(device)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=params_dict["num_layers"]).to(device)

            self.fc3_adv = nn.Linear(int(params_dict["seq_length"]*params_dict["d_model"]),params_dict["num_actions"]).to(device)  
            self.fc3_v = nn.Linear(int(params_dict["seq_length"]*params_dict["d_model"]), 1).to(device)  
    

        def forward(self, x):
            # x: [batch_size, seq_length, input_dim]
            x=x.to(device)
            y=self.linear(x.transpose(0,1))
            y = self.transformer_encoder(y).transpose(0,1)
            y_=y.contiguous().view(y.shape[0],-1) # [batch_size,18*d_model]

            adv = self.fc3_adv(y_) # [batch_size,action_nums]
            val = self.fc3_v(y_)   #.expand(-1, adv.size(1)) # [batch_size,1]  动作优势的维度？？？
            output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

            del x,y,y_,adv,val

            return output



    def make_mini_batch(self, episode):
        # 从经验池中获取小批量数据
        if episode < 30:
            transitions = self.memory.sample(BATCH_SIZE)
        else:
            indexes = self.td_error_memory.get_prioritized_indexes(BATCH_SIZE)
            transitions = [self.memory.memory[n] for n in indexes]

        # 将(state, action, state_next, reward)×BATCH_SIZE
        # 转换为(state×BATCH_SIZE, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # 将每个变量的元素转为与小批量数据对应的形式
        state_batch = torch.vstack(batch.state)
        action_batch = torch.vstack(batch.action) 
        reward_batch = torch.vstack(batch.reward)
        non_final_next_states = torch.vstack([s for s in batch.next_state
                                           if s is not None])
       
        return batch, state_batch, action_batch, reward_batch, non_final_next_states


    def get_expected_state_action_values(self):
        # 找到Q(s_t,a_t）作为监督信息

        self.main_q_network.eval()
        self.target_q_network.eval()

        # 网络输出的Q(s_t,a_t)
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)

        # 求max{Q(s_t+1, a)}的值   
        # 创建索引掩码以检查CartPole是否完成且具有next_state
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, self.batch.next_state))).to(device)
        # 首先全部设置为0
        next_state_values = torch.zeros(BATCH_SIZE).to(device)

        a_m = torch.zeros(BATCH_SIZE).long().to(device)
        # 从主Q网络求下一个状态中最大Q值的动作a_m，并返回该动作对应的索引
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]

        # 仅过滤有下一状态的，并将size32转换成size 32*1
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 从目标Q网络求出具有下一状态的index的动作a_m的最大Q值
        # 使用squeeze（）将size[minibatch*1]压缩为[minibatch]
        with torch.no_grad():
            next_state_values[non_final_mask] = \
                self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states)\
                    .detach().squeeze()

        # 从Q公式中求Q(s_t, a_t)作为监督信息
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        del non_final_mask,next_state_values,a_m,a_m_non_final_next_states


        return expected_state_action_values



    def update_main_q_network(self,episode,step):

        self.optimizer.zero_grad()

        # 使用autocast开启半精度上下文
        with autocast():
            self.main_q_network.train()
            loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
            
        # 调用backward时，使用scaler.scale来缩放梯度
        scaler.scale(loss).backward()

        # scaler.step()会反向缩放梯度更新模型参数
        scaler.step(self.optimizer)

        # 更新缩放器的比例，用于下一次迭代
        scaler.update()

        # tensor board
        writer.add_scalar("Loss/train", loss,int(episode*CYCLE_COUNT+step))

        # 释放内存
        del loss


    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def update_td_error_memory(self):

        self.main_q_network.eval()
        self.target_q_network.eval()

        transitions = self.memory.memory
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_action_values = self.main_q_network(
            state_batch).gather(1, action_batch)

        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, batch.next_state))).to(device)

        next_state_values = torch.zeros(len(self.memory)).to(device)
        a_m = torch.zeros(len(self.memory)).long().to(device)

        a_m[non_final_mask] = self.main_q_network(non_final_next_states).detach().max(1)[1]

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_q_network(
                non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 找到TD误差
        td_errors = (reward_batch + GAMMA * next_state_values) - \
                    state_action_values.squeeze()

        self.td_error_memory.memory = td_errors.detach().cpu().data.numpy().tolist()


        # 释放内存
        del transitions,batch,state_batch,action_batch,reward_batch,non_final_next_states,\
            state_action_values,a_m,a_m_non_final_next_states,td_errors





    def decide_action(self, state, episode,dua_list):
        # 获取动作
        model_flag=True
        epsilon = 0.5 * (1 / (episode + 1))
        
        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action_index = self.main_q_network(state).max(1)[1].view(1, 1)
                action=self.action_space[action_index.cpu().data.numpy()[0][0]]
        else:
            model_flag=False
            action_index=randint(0,17)
            action=self.action_space[action_index]
            action_index=torch.LongTensor([[action_index]]).to(device)
        
        # 计算相位
        phase=[]
        programID=0
        punish=False
        a,b,c,d=list(np.array(dua_list)+np.array(action))

        # 惩罚项
        if np.sum(np.array([a,b,c,d])<5)!=0:
            punish=True
            a,b,c,d=[70,14,70,14]

        # 设置信号灯
        phase.append(traci.trafficlight.Phase(a,'GGGGGgrrrrrrGGGGGggrrrrrr',next=()))
        phase.append(traci.trafficlight.Phase(3,'yyyyygrrrrrryyyyyggrrrrrr',next=()))
        phase.append(traci.trafficlight.Phase(b,'rrrrrGrrrrrrrrrrrGGrrrrrr',next=()))
        phase.append(traci.trafficlight.Phase(3,'rrrrryrrrrrrrrrrryyrrrrrr',next=()))
        phase.append(traci.trafficlight.Phase(c,'rrrrrrGGGGGgrrrrrrrGGGGgg',next=()))
        phase.append(traci.trafficlight.Phase(3,'rrrrrryyyyygrrrrrrryyyygg',next=()))
        phase.append(traci.trafficlight.Phase(d,'rrrrrrrrrrrGrrrrrrrrrrrGG',next=()))
        phase.append(traci.trafficlight.Phase(3,'rrrrrrrrrrryrrrrrrrrrrryy',next=()))
        tl_list=traci.trafficlight.getIDList()
        logic = traci.trafficlight.Logic(programID,0,0,phases=phase)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(tl_list[0], logic)

        del epsilon,episode,action,phase,programID,tl_list,logic

        return action_index,[a,b,c,d],punish,model_flag



class Agent:
    # 智能体，带杆的小车

    def __init__(self,net_type,params_dict):
        '''设置任务状态和动作的数量'''
        self.brain = Brain(net_type,params_dict)

    def update_q_function(self, episode,step):
        '''更新Q函数'''
        self.brain.replay(episode,step)

    def get_action(self, state, episode,dua_list):
        '''确定动作'''
        action,dua_list,punish,model_flag= self.brain.decide_action(state, episode,dua_list)
        return action,dua_list,punish,model_flag

    def memorize(self, state, action, state_next, reward,dua_list):
        '''将state, action, state_next, reward保存在经验池中'''
        self.brain.memory.push(state, action, state_next, reward,dua_list)

    def update_target_q_function(self):
        # 将目标Q网络更新到与主网络相同
        self.brain.update_target_q_network()

    def memorize_td_error(self, td_error):
        # 存储TD误差
        self.brain.td_error_memory.push(td_error)

    def update_td_error_memory(self):
        # 更新存储在TD误差存储器中的误差
        self.brain.update_td_error_memory()

    def save_expirence(self,episode):
        self.brain.memory.save(episode)

    def save_td_error_memory(self,episode):
        self.brain.td_error_memory.save(episode)

    def save_model(self,episode):
        self.brain.save_model(episode)

    def load_model(self,load_model_path):
        self.brain.load_model(load_model_path)




class Environment:

    def __init__(self,net_type,params_dict,load_model_path):
        self.env=env()
        if net_type==0:
            self.agent = Agent(net_type,params_dict)  #selfAettention
        elif net_type==1:
            self.agent = Agent(net_type,params_dict)   #MultiheadAttentionNet
        elif net_type==2:
            self.agent = Agent(net_type,params_dict)   #TransformerEncoderNet

        # 加载模型
        if load_model_path:
            self.agent.load_model(load_model_path)


    def run(self):

        model_action_cnt=0

        for episode in range(NUM_EPISODES):

            state,reward,_,dua_list=self.env.init_simu()
            state = torch.from_numpy(state).unsqueeze(0).float().to(device)
            
            for step in range(CYCLE_COUNT):   # 24h 的step数

                # 通过动作a_t得到s_{t+1}和done标志   
                action_index,dua_list,punish,model_flag=self.agent.get_action(state, episode,dua_list)
                observation_next, reward= self.env.step(dua_list) 

                # 将动作反馈数据添加到tensorboard
                writer.add_scalar("reward /train", reward,int(episode*CYCLE_COUNT+step))
                writer.add_histogram("action /train", action_index[0][0],int(episode*CYCLE_COUNT+step))

                if model_flag:
                    model_action_cnt+=1
                    writer.add_histogram("model action /train", action_index[0][0],model_action_cnt)

                if observation_next is not None:
                    for i in range(18):
                        writer.add_scalar(f"wait time {i} /train",torch.tensor(observation_next[i,1]),episode*CYCLE_COUNT+step)
                        writer.add_scalar(f"speed {i} /train",torch.tensor(observation_next[i,2]),episode*CYCLE_COUNT+step)
                        writer.add_scalar(f"vehicle nums {i} /train",torch.tensor(observation_next[i,3]),episode*CYCLE_COUNT+step)


                # 将模型权重添加到TensorBoard
                for name, param in self.agent.brain.main_q_network.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step=int(episode*CYCLE_COUNT+step))

                # 将计算图添加到tensorboard
                dummy_input = torch.randn(32,params_dict["seq_length"], params_dict["input_dim"])  
                writer.add_graph(self.agent.brain.main_q_network, dummy_input)


                # 加入惩罚项
                if punish:
                    reward=torch.from_numpy(np.array([10000])).float().to(device) 
                else:
                    reward = torch.from_numpy(reward).float().to(device)  

                # state_next的处理
                if observation_next is None:
                    state_next = None 
                elif state is not None:
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).unsqueeze(0).float().to(device)

                    # 向经验池中添加经验
                    self.agent.memorize(state, action_index, state_next, reward,dua_list)

                    # 将TD误差添加到误差变量中
                    self.agent.memorize_td_error(0)

                    # 优先经验回放中更新Q函数
                    self.agent.update_q_function(episode,step)


                # 更新state
                state = state_next

                # done处理
                if step==CYCLE_COUNT-1:
                    print('%d Episode: Finished after %d' % (episode, step + 1))

                    # 更新TD误差存储变量中的内容
                    self.agent.update_td_error_memory()

                    # 更新目标网络+保存模型
                    if episode % 2 == 0:
                        self.agent.update_target_q_function()

                    break


                # 删除缓存
                del action_index,punish,model_flag,observation_next, reward
                    

            # 关闭仿真
            traci.close()

            # 保存模型
            self.agent.save_model(episode)

            # 保存经验池到本地
            print('Memory pool size is:',len(self.agent.brain.memory))
            if episode%1==0:
                self.agent.save_expirence(episode)    
                self.agent.save_td_error_memory(episode)       

            



if __name__ == '__main__':

    # 常量
    GAMMA = 0.99  # 时间折扣率
    BATCH_SIZE = 32
    CAPACITY = 10000
    NUM_EPISODES = 1000
    TD_ERROR_EPSILON = 0.0001
    LEARNING_RATE=0.001
    Transition = namedtuple(
        "Transition", ("state", "action", "next_state", "reward","dua_list"))
    
    #仿真时间
    total_time = 24 * 60 * 60
    tl_cycle = 180
    CYCLE_COUNT =int(total_time/tl_cycle) #一天的周期数

    # 检查是否有可用的 GPU 设备
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 选择第一个可用的 GPU
    else:
        device = torch.device("cpu")   # 如果没有 GPU，则使用 CPU

    # 初始化半精度训练的混合精度缩放器
    scaler = GradScaler()

    # tensorboard
    writer = SummaryWriter()

    # 配置神经网络参数
    net_type=0
    load_model_path="./models/model_3.pt"
    if net_type==0:
        params_dict={"seq_length":18,"input_dim":4,"hidden_dim":64,"num_actions":19}
    elif net_type==1:
        params_dict={"seq_length":18,"input_dim":4,"hidden_dim":64,"num_actions":19,"nhead":8}
    elif net_type==2:
        params_dict={"seq_length":18,"input_dim":4,"d_model":64,"n_head":8,"num_layers":6,"num_actions":19}

    # 启动环境
    cartpole_env = Environment(net_type=net_type,params_dict=params_dict,load_model_path=load_model_path)
    cartpole_env.run()

    # 更新tensorboard
    writer.flush()
    writer.close()
