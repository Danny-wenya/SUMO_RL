import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci


def set_singal(tl,programID,dua_list):
    phase = []
    play=[]
    time_len =0
    if tl == 'point1':
        phase,time_len,play =point1(dua_list)
    if tl == 'point2':
        phase,time_len,play = point2(dua_list)
    if tl == 'point3':
        phase,time_len,play = point3(dua_list)
    if tl == 'point4':
        phase,time_len,play = point4(dua_list)

    logic = traci.trafficlight.Logic(programID,0,0,phases=phase)
    traci.trafficlight.setCompleteRedYellowGreenDefinition(tl, logic)
    return time_len,play


def point1(dua_list):    #左转优先级低可能会导致左转车道一直不能放行，建议绿灯通行不区分优先级。
    '''实验证明不区分优先级的路网一旦拥堵，汽车会彻底纠缠堵死，若区分了优先级，会优先放行一个方向的，但若另一个方向无法放行，只能等到直行车辆放完，左转车辆才有放行机会，为避免拥堵，可以加大某些路口放行时间。'''
    NS = dua_list[0]
    SL = dua_list[1]
    ES = dua_list[2]
    WL = dua_list[3]
    SS = dua_list[4]
    NL = dua_list[5]
    WS = dua_list[6]
    EL = dua_list[7]
    phase = []
    time_len = 0

    if NS>=SS:
        a=WS-5
        b=ES-5
        c=SL-5
        d=NS-SS if NS-SS>25 else 25
        e=SS-10
        play = [a, b, c, d, e]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr',next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrrrrGrrrGrrrrrrGyyy', next = ()))
        phase.append(traci.trafficlight.Phase(a,'GrrrrrrGrrrGrrrrrrGGGG', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrrrrGyyyGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(b,'GrrrrrrGGGGGrrrrrrGrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrryyGrrrGrrrryyGrrr', next=()))
        phase.append(traci.trafficlight.Phase(c,'GrrrrGGGrrrGrrrrggGrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GyyyyrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(d,'GGGGGrrGrrrGrrrrrrGrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GGGGGrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GGGGGrrGrrrGyyyyrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(e,'GGGGGrrGrrrGGGGGrrGrrr', next=()))

    else:
        a=WS - 5
        b=ES - 5
        c=NL - 5
        d=SS-NS if SS-NS>25 else 25
        e=NS - 10
        play = [a, b, c, d, e]
        time_len=sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrrrrGrrrGrrrrrrGyyy', next=()))
        phase.append(traci.trafficlight.Phase(a,'GrrrrrrGrrrGrrrrrrGGGG', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrrrrGyyyGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(b,'GrrrrrrGGGGGrrrrrrGrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrryyGrrrGrrrryyGrrr', next=()))
        phase.append(traci.trafficlight.Phase(c,'GrrrrGGGrrrGrrrrGGGrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrGGGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrrrrGrrrGyyyyGGGrrr', next=()))
        phase.append(traci.trafficlight.Phase(d,'GrrrrrrGrrrGGGGGGGGrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGGGGGrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GyyyyrrGrrrGGGGGrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(e,'GGGGGrrGrrrGGGGGrrGrrr', next=()))

    return phase,time_len,play

def point2(dua_list):
    NS = dua_list[0]    #0
    SL = dua_list[1]    #60
    ES = dua_list[2]    #120
    WL = dua_list[3]    #0
    SS = dua_list[4]    #0
    NL = dua_list[5]    #0
    WS = dua_list[6]    #60
    EL = dua_list[7]    #60
    phase=[]


    # phase2 = [0, 60, 120, 0, 0, 0, 60, 60]
    a=EL-5
    b=WS-5
    c=SL-5
    play = [a, b, c]
    time_len = sum(play)+5*len(play)

    phase.append(traci.trafficlight.Phase(2,'rrrGrrGrr', next=()))
    phase.append(traci.trafficlight.Phase(3,'yyyGrrGrr', next=()))
    phase.append(traci.trafficlight.Phase(a,'GGGGrrGrr', next=()))

    phase.append(traci.trafficlight.Phase(2,'rrrGrrGrr', next=()))
    phase.append(traci.trafficlight.Phase(3,'rrrGrrGyy', next=()))
    phase.append(traci.trafficlight.Phase(b,'rrrGrrGGG', next=()))

    phase.append(traci.trafficlight.Phase(2,'rrrGrrGrr', next=()))
    phase.append(traci.trafficlight.Phase(3,'rrrGyyGrr', next=()))
    phase.append(traci.trafficlight.Phase(c,'rrrGGGGrr', next=()))


    return phase,time_len,play



def point3(dua_list):
    NS = dua_list[0]    #0
    SL = dua_list[1]    #60
    ES = dua_list[2]    #120
    WL = dua_list[3]    #0
    SS = dua_list[4]    #0
    NL = dua_list[5]    #0
    WS = dua_list[6]    #60
    EL = dua_list[7]    #60
    phase = []
    time_len=0
    play=[]
    # phase3 = [0, 60, 120, 0, 0, 0, 60, 60]
    if ES>=WS:
        a=WL-5
        b=ES - WS if ES-WS>25 else 25
        c=WS - 10
        d=SS - 5
        e=NS-5
        play = [a, b, c, d, e]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2, 'GrrrGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrGrrryGrrrGrrry', next=()))
        phase.append(traci.trafficlight.Phase(a, 'GrrrGrrrGGrrrGrrrG', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrGrrrGGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrGyyyGGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(b, 'GrrrGGGGGGrrrGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrGGGGrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrGGGGrGrrrGyyyr', next=()))
        phase.append(traci.trafficlight.Phase(c, 'GrrrGGGGrGrrrGGGGr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrGrrrrGyyyGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(d, 'GrrrGrrrrGGGGGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GyyyGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(e, 'GGGGGrrrrGrrrGrrrr', next=()))

    elif ES<WS :
        a=EL-5
        b=WS-ES if WS-ES>25 else 25
        c=ES-10
        d=SS-5
        e=NS-5
        play = [a, b, c, d, e]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2,'GrrrGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrGrrryGrrrGrrry', next=()))
        phase.append(traci.trafficlight.Phase(a,'GrrrGrrrGGrrrGrrrG', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrGrrrrGrrrGrrrG', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrGrrrrGrrrGyyyG', next=()))
        phase.append(traci.trafficlight.Phase(b,'GrrrGrrrrGrrrGGGGG', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrGrrrrGrrrGGGGr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrGyyyrGrrrGGGGr', next=()))
        phase.append(traci.trafficlight.Phase(c,'GrrrGGGGrGrrrGGGGr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrGrrrrGyyyGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(d,'GrrrGrrrrGGGGGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GyyyGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(e,'GGGGGrrrrGrrrGrrrr', next=()))


    return phase,time_len,play


def point4(dua_list):
    # dua_list = [45 for _ in range(8)]
    NS = dua_list[0]
    SL = dua_list[1]
    ES = dua_list[2]
    WL = dua_list[3]
    SS = dua_list[4]
    NL = dua_list[5]
    WS = dua_list[6]
    EL = dua_list[7]
    phase = []
    time_len=0
    play=[]

    if NS>=SS and ES>=WS :
        a=WL - 5
        b=ES - WS if ES-WS>25 else 25
        c=WS - 10
        d=SL - 5
        e=NS - SS if NS-SS>25 else 25
        f=SS - 10
        play = [a, b, c, d, e, f]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrryGrrrrrGrrry', next=()))
        phase.append(traci.trafficlight.Phase(a, 'GrrrrGrrrGGrrrrrGrrrG', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrGGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGyyyGGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(b, 'GrrrrGGGGGGrrrrrGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGGGGGGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGGGGGGrrrrrGyyrr', next=()))
        phase.append(traci.trafficlight.Phase(c, 'GrrrrGGGGGGrrrrrGGGrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrryGrrrrGrrryyGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(d, 'GrrrGGrrrrGrrrGGGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrGGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GyyyGGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(e, 'GGGGGGrrrrGrrrrrGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GGGGGGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GGGGGGrrrrGyyyrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(f, 'GGGGGGrrrrGGGGrrGrrrr', next=()))

    elif NS>=SS and ES<WS :
        a=EL - 5
        b=WS - ES if WS-ES>25 else 25
        c=ES - 10
        d=SL - 5
        e=NS - SS if NS-SS>25 else 25
        f=SS - 10
        play = [a, b, c, d, e, f]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrryGrrrrrGrrry', next=()))
        phase.append(traci.trafficlight.Phase(a, 'GrrrrGrrrGGrrrrrGrrrG', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrG', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrrrGrrrrrGyyyG', next=()))
        phase.append(traci.trafficlight.Phase(b, 'GrrrrGrrrrGrrrrrGGGGG', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGGGGr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGyyyrGrrrrrGGGGr', next=()))
        phase.append(traci.trafficlight.Phase(c, 'GrrrrGGGGrGrrrrrGGGGr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrryGrrrrGrrryyGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(d, 'GrrrGGrrrrGrrrGGGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrGGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GyyyGGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(e, 'GGGGGGrrrrGrrrrrGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GGGGrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GGGGrGrrrrGyyyrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(f, 'GGGGrGrrrrGGGGrrGrrrr', next=()))

    elif NS <SS and ES >=WS :
        a=WL - 5
        b=ES - WS if ES-WS>25 else 25
        c=WS - 10
        d=NL - 5
        e=SS - NS if SS-NS>25 else 25
        f=NS - 10
        play = [a, b, c, d, e, f]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrryGrrrrrGrrry', next=()))
        phase.append(traci.trafficlight.Phase(a, 'GrrrrGrrrGGrrrrrGrrrG', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGyyyrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(b, 'GrrrrGGGGrGrrrrrGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGGGGrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGGGGrGrrrrrGyyyr', next=()))
        phase.append(traci.trafficlight.Phase(c, 'GrrrrGGGGrGrrrrrGGGGr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrryGrrrrGrrryyGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(d, 'GrrrGGrrrrGrrrGGGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrGGGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrrrGyyyGGGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(e, 'GrrrrGrrrrGGGGGGGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGGGGrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GyyyrGrrrrGGGGrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(f, 'GGGGrGrrrrGGGGrrGrrrr', next=()))

    elif NS <SS and ES <WS  :
        a=EL - 5
        b=WS - ES if WS-ES>25 else 25
        c=ES - 10
        d=NL - 5
        e=SS - NS if SS-NS>25 else 25
        f=NS - 10
        play = [a, b, c, d, e, f]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrryGrrrrrGrrry', next=()))
        phase.append(traci.trafficlight.Phase(a, 'GrrrrGrrrGGrrrrrGrrrG', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrG', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrrrGrrrrrGrrrG', next=()))
        phase.append(traci.trafficlight.Phase(b, 'GrrrrGrrrrGrrrrrGrrrG', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGGGGr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGyyyrGrrrrrGGGGr', next=()))
        phase.append(traci.trafficlight.Phase(c, 'GrrrrGGGGrGrrrrrGGGGr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrryGrrrrGrrryyGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(d, 'GrrrGGrrrrGrrrGGGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrGGGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrrrGyyyGGGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(e, 'GrrrrGrrrrGGGGGGGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGGGGrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GyyyrGrrrrGGGGrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(f, 'GGGGrGrrrrGGGGrrGrrrr', next=()))

    return phase,time_len,play