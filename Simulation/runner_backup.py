import os
import sys
import traci
from sumolib import checkBinary
import numpy as np
#import car_env
import json
import traci.constants as tc
from Simulation.tools import signal
# from Simulation.tools import deal_data
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
trip_filename= 'output.xml'
time_step = 1

#设置仿真时间
time = 24 *60*60
tl_cycle = 120

cycle_count = time/tl_cycle #一天的周期数
tl_list=['point1','point2','point3','point4','point5','point6','point7''point8','point9','point10','point11','point12']
# detector = traci.lanearea.getIDList()
def init_simu():
    sumoBinary  = checkBinary('sumo')
    sumoCmd = [sumoBinary,'-c','shenzhen12.sumocfg',"--step-length="+str(time_step),"--collision.action=warn","--fcd-output=./output/output.xml"]
    traci.start(sumoCmd)

def get_tl_control_lane(tl_list):
    road_detector = {}
    for detector in traci.lanearea.getIDList():
        lane = traci.lanearea.getLaneID(detector)

        road = lane.split('_')[0]
        if road not in road_detector:
            road_detector[road]=[]
        road_detector[road].append(detector)
    tl_road = {}
    for tl in tl_list:
        tl_road[tl] = {}
        control_lane = traci.trafficlight.getControlledLanes(tl)

        control_road = list (x.split('_')[0] for x in control_lane)
        for road in control_road:
            if road not in tl_road:
                tl_road[tl][road]=road_detector[road]
    return tl_road

def step(tl_list):
    step = 0 #仿真的初始步长
    tl_data = get_tl_control_lane(tl_list) # 获取信号灯控制的道路以及检测器

    output_data = np.zeros((12,4,3))   #output_data 是每个周期输出的数据，12个交叉口，4个方向
    temp_data = {} #临时数据
    wait_time = {} #车辆的等待时间
    while traci.simulation.getMinExpectedNumber()>0:
        #初步获取检测器数据
        for tl in tl_data:
            if tl not in temp_data:
                temp_data[tl]={}
            if tl not in wait_time:
                wait_time[tl] = {}
            for road in tl_data[tl]:
                if road not in temp_data[tl]:
                    temp_data[tl][road]=[[],[]]
                for detector in tl_data[tl][road]:
                    veh_ids = traci.lanearea.getLastStepVehicleIDs(detector) #每辆车的延误时间
                    for veh in veh_ids:
                        wait_time[tl][veh] = traci.vehicle.getAccumulatedWaitingTime(veh)
                    temp_data[tl][road][0].extend(list(veh_ids))
                    queue = traci.lanearea.getJamLengthMeters(detector) #每个检测器每个步长的排队长度
                    if queue >1:
                        temp_data[tl][road][1].append(queue)
                    # halting = halting+traci.lanearea.getLastStepHaltingNumber()
        # print(wait_time)
        #处理初步获取的数据
        for tl in temp_data:
            length = len(temp_data[tl])
            road_list = list(temp_data[tl].keys())
            # print(road_list)
            init_index=0
            if length == 3:
                init_index = 1
            for road in temp_data[tl]:
                output_data[tl_list.index(tl)][road_list.index(road)+init_index][0] =  len(set(temp_data[tl][road][0])) #流量数据
                for veh in set(temp_data[tl][road][0]):
                    output_data[tl_list.index(tl)][road_list.index(road)+init_index][2] +=wait_time[tl][veh] #延误数据
                if (temp_data[tl][road][1]):
                    output_data[tl_list.index(tl)][road_list.index(road)+init_index][1] = max(temp_data[tl][road][1]) #排队长度


        #下面应该是进行强化学习的code
        tl_list = traci.trafficlight.getIDList()

        for tl in tl_list:
            programID = traci.trafficlight.getProgram(tl)
            # signal.set_singal(tl,programID,dua_list) #根据返回的绿灯时间进行信号配时
        #     print(tl,traci.trafficlight.getControlledLanes(tl))
        if  step == tl_cycle:
            break

        traci.simulationStep()
        step+=1


def main():
    count= 0
    init_simu() #初始化仿真
    tl_list = traci.trafficlight.getIDList()
    tl_list = sorted(tl_list,key=lambda x:x[:len(x)-5]) #对信号灯list按照交叉口排序排序
    # print(tl_list)
    #这里可以设置强化学习的次数 加一个循环即可，代码中未写出
    while count < cycle_count:
        count+=1
        step(tl_list)
    traci.close()

if __name__ == '__main__':
    main()