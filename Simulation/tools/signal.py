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
        c=SL-15
        d=NS-SS+10
        e=SS-10
        play = [a, b, c, d, e]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr',next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrrrrGrrrGrrrrrrGyyy', next = ()))
        phase.append(traci.trafficlight.Phase(a,'GrrrrrrGrrrGrrrrrrGGGg', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrrrrGyyyGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(b,'GrrrrrrGGGgGrrrrrrGrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrryyGrrrGrrrryyGrrr', next=()))
        phase.append(traci.trafficlight.Phase(c,'GrrrrggGrrrGrrrrggGrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GyyyyrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(d,'GGGGGrrGrrrGrrrrrrGrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GGGGGrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GGGGGrrGrrrGyyyyrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(e,'GGGGGrrGrrrGGGGGrrGrrr', next=()))

    else:
        a=WS - 5
        b=ES - 5
        c=NL - 15
        d=SS-NS+10
        e=NS - 10
        play = [a, b, c, d, e]
        time_len=sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrrrrGrrrGrrrrrrGyyy', next=()))
        phase.append(traci.trafficlight.Phase(a,'GrrrrrrGrrrGrrrrrrGGGg', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrrrrGyyyGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(b,'GrrrrrrGGGgGrrrrrrGrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrrrGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrryyGrrrGrrrryyGrrr', next=()))
        phase.append(traci.trafficlight.Phase(c,'GrrrrggGrrrGrrrrggGrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrrrrGrrrGrrrrggGrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrrrrGrrrGyyyyggGrrr', next=()))
        phase.append(traci.trafficlight.Phase(d,'GrrrrrrGrrrGGGGGggGrrr', next=()))

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
    phase.append(traci.trafficlight.Phase(a,'GGgGrrGrr', next=()))

    phase.append(traci.trafficlight.Phase(2,'rrrGrrGrr', next=()))
    phase.append(traci.trafficlight.Phase(3,'rrrGrrGyy', next=()))
    phase.append(traci.trafficlight.Phase(b,'rrrGrrGGG', next=()))

    phase.append(traci.trafficlight.Phase(2,'rrrGrrGrr', next=()))
    phase.append(traci.trafficlight.Phase(3,'rrrGyyGrr', next=()))
    phase.append(traci.trafficlight.Phase(c,'rrrGggGrr', next=()))


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
        a=WL-15
        b=ES - WS+10
        c=WS - 10
        d=SS - 5
        e=NS-5
        play = [a, b, c, d, e]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2, 'GrrrGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrGrrryGrrrGrrry', next=()))
        phase.append(traci.trafficlight.Phase(a, 'GrrrGrrrgGrrrGrrrg', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrGrrrgGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrGyyygGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(b, 'GrrrGGGGgGrrrGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrGGGGrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrGGGGrGrrrGyyyr', next=()))
        phase.append(traci.trafficlight.Phase(c, 'GrrrGGGGrGrrrGGGGr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrGrrrrGyyyGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(d, 'GrrrGrrrrGGGgGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GyyyGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(e, 'GGGgGrrrrGrrrGrrrr', next=()))

    elif ES<WS :
        a=EL-15
        b=WS-ES+10
        c=ES-10
        d=SS-5
        e=NS-5
        play = [a, b, c, d, e]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2,'GrrrGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrGrrryGrrrGrrry', next=()))
        phase.append(traci.trafficlight.Phase(a,'GrrrGrrrgGrrrGrrrg', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrGrrrrGrrrGrrrg', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrGrrrrGrrrGyyyg', next=()))
        phase.append(traci.trafficlight.Phase(b,'GrrrGrrrrGrrrGGGGg', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrGrrrrGrrrGGGGr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrGyyyrGrrrGGGGr', next=()))
        phase.append(traci.trafficlight.Phase(c,'GrrrGGGGrGrrrGGGGr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GrrrGrrrrGyyyGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(d,'GrrrGrrrrGGGgGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2,'GrrrGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3,'GyyyGrrrrGrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(e,'GGGgGrrrrGrrrGrrrr', next=()))


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
        a=WL - 15
        b=ES - WS+10
        c=WS - 10
        d=SL - 15
        e=NS - SS+10
        f=SS - 10
        play = [a, b, c, d, e, f]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrryGrrrrrGrrry', next=()))
        phase.append(traci.trafficlight.Phase(a, 'GrrrrGrrrgGrrrrrGrrrg', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrgGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGyyygGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(b, 'GrrrrGGGGgGrrrrrGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGGGGgGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGGGGgGrrrrrGyyrr', next=()))
        phase.append(traci.trafficlight.Phase(c, 'GrrrrGGGGgGrrrrrGGGrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrryGrrrrGrrryyGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(d, 'GrrrgGrrrrGrrrggGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrgGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GyyygGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(e, 'GGGGgGrrrrGrrrrrGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GGGGgGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GGGGgGrrrrGyyyrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(f, 'GGGGgGrrrrGGGGrrGrrrr', next=()))

    elif NS>=SS and ES<WS :
        a=EL - 15
        b=WS - ES+10
        c=ES - 10
        d=SL - 15
        e=NS - SS+10
        f=SS - 10
        play = [a, b, c, d, e, f]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrryGrrrrrGrrry', next=()))
        phase.append(traci.trafficlight.Phase(a, 'GrrrrGrrrgGrrrrrGrrrg', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrg', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrrrGrrrrrGyyyg', next=()))
        phase.append(traci.trafficlight.Phase(b, 'GrrrrGrrrrGrrrrrGGGGg', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGGGGr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGyyyrGrrrrrGGGGr', next=()))
        phase.append(traci.trafficlight.Phase(c, 'GrrrrGGGGrGrrrrrGGGGr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrryGrrrrGrrryyGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(d, 'GrrrgGrrrrGrrrggGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrgGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GyyygGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(e, 'GGGGgGrrrrGrrrrrGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GGGGrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GGGGrGrrrrGyyyrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(f, 'GGGGrGrrrrGGGGrrGrrrr', next=()))

    elif NS <SS and ES >=WS :
        a=WL - 15
        b=ES - WS+10
        c=WS - 10
        d=NL - 15
        e=SS - NS+10
        f=NS - 10
        play = [a, b, c, d, e, f]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrryGrrrrrGrrry', next=()))
        phase.append(traci.trafficlight.Phase(a, 'GrrrrGrrrgGrrrrrGrrrg', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGyyyrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(b, 'GrrrrGGGGrGrrrrrGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGGGGrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGGGGrGrrrrrGyyyr', next=()))
        phase.append(traci.trafficlight.Phase(c, 'GrrrrGGGGrGrrrrrGGGGr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrryGrrrrGrrryyGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(d, 'GrrrgGrrrrGrrrggGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrggGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrrrGyyyggGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(e, 'GrrrrGrrrrGGGGggGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGGGGrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GyyyrGrrrrGGGGrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(f, 'GGGGrGrrrrGGGGrrGrrrr', next=()))

    elif NS <SS and ES <WS  :
        a=EL - 15
        b=WS - ES +10
        c=ES - 10
        d=NL - 15
        e=SS - NS +10
        f=NS - 10
        play = [a, b, c, d, e, f]
        time_len = sum(play)+5*len(play)

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrryGrrrrrGrrry', next=()))
        phase.append(traci.trafficlight.Phase(a, 'GrrrrGrrrgGrrrrrGrrrg', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrg', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrrrGrrrrrGrrrg', next=()))
        phase.append(traci.trafficlight.Phase(b, 'GrrrrGrrrrGrrrrrGrrrg', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGGGGr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGyyyrGrrrrrGGGGr', next=()))
        phase.append(traci.trafficlight.Phase(c, 'GrrrrGGGGrGrrrrrGGGGr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrryGrrrrGrrryyGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(d, 'GrrrgGrrrrGrrrggGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGrrrggGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GrrrrGrrrrGyyyggGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(e, 'GrrrrGrrrrGGGGggGrrrr', next=()))

        phase.append(traci.trafficlight.Phase(2, 'GrrrrGrrrrGGGGrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(3, 'GyyyrGrrrrGGGGrrGrrrr', next=()))
        phase.append(traci.trafficlight.Phase(f, 'GGGGrGrrrrGGGGrrGrrrr', next=()))

    return phase,time_len,play