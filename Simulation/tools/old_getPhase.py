
# 固定周期
Cycle = 180

######### 定义phase的NEMA相位对应的索引########
#phase[WL]就对应WL相位的绿灯时间
NS = 0
SL = 1 
ES = 2
WL = 3
SS = 4 
NL = 5 
WS = 6 
EL = 7
#############################################
def getPhaseFromAction1(phase, act):
    """
    1交叉口的动作选择方案 23个
    定周期配时：根据当前的相位和选择的动作调整得到新的相位配时方案
    输入：当前的配时方案phase 和 需要执行的动作操作 act
    输出：新的配时方案
    """
    act = int(act)
    if act == 0:
        phase[0] += 5
        phase[1] -= 5
    elif act == 1:
        phase[0] -= 5
        phase[1] += 5
    elif act == 2:
        phase[4] += 5
        phase[5] -= 5
    elif act == 3:
        phase[4] -= 5
        phase[5] += 5
    elif act == 5:
        phase[2] += 5
        phase[7] += 5
        phase[0] -= 5
        phase[4] -= 5
    elif act == 6:
        phase[2] += 5
        phase[7] += 5
        phase[0] -= 5
        phase[5] -= 5
    elif act == 7:
        phase[2] += 5
        phase[7] += 5
        phase[1] -= 5
        phase[4] -= 5
    elif act == 8:
        phase[2] += 5
        phase[7] += 5
        phase[1] -= 5
        phase[5] -= 5
    elif act == 9:
        phase[6] += 5
        phase[3] += 5
        phase[0] -= 5
        phase[4] -= 5
    elif act == 10:
        phase[6] += 5
        phase[3] += 5
        phase[0] -= 5
        phase[5] -= 5
    elif act == 11:
        phase[6] += 5
        phase[3] += 5
        phase[1] -= 5
        phase[4] -= 5
    elif act == 12:
        phase[6] += 5
        phase[3] += 5
        phase[1] -= 5
        phase[5] -= 5
    elif act == 13:
        phase[2] -= 5
        phase[7] -= 5
        phase[0] += 5
        phase[4] += 5
    elif act == 14:
        phase[2] -= 5
        phase[7] -= 5
        phase[0] += 5
        phase[5] += 5
    elif act == 15:
        phase[2] -= 5
        phase[7] -= 5
        phase[1] += 5
        phase[4] += 5
    elif act == 16:
        phase[2] -= 5
        phase[7] -= 5
        phase[1] += 5
        phase[5] += 5
    elif act == 17:
        phase[6] -= 5
        phase[3] -= 5
        phase[0] += 5
        phase[4] += 5
    elif act == 18:
        phase[6] -= 5
        phase[3] -= 5
        phase[0] += 5
        phase[5] += 5
    elif act == 19:
        phase[6] -= 5
        phase[3] -= 5
        phase[1] += 5
        phase[4] += 5
    elif act == 20:
        phase[6] -= 5
        phase[3] -= 5
        phase[1] += 5
        phase[5] += 5
    elif act == 21:
        phase[2] -= 5
        phase[7] -= 5
        phase[6] += 5
        phase[3] += 5
    elif act == 22:
        phase[2] += 5
        phase[7] += 5
        phase[6] -= 5
        phase[3] -= 5
    return phase

def getPhaseFromAction2(phase, act):
    """
    2交叉口的动作选择方案 7个
    定周期配时：根据当前的相位和选择的动作调整得到新的相位配时方案
    输入：当前的配时方案phase 和 需要执行的动作操作 act
    输出：新的配时方案

    phase2 = [0, 60, 120, 0, 0, 0, 60, 60]
    """
    act = int(act)
    if act == 0:
        phase[6] += 5
        phase[7] -= 5
    elif act == 1:
        phase[6] -= 5
        phase[7] += 5
    elif act == 2:
        phase[1] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 3:
        phase[1] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 5:
        phase[1] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 6:
        phase[1] += 5
        phase[2] -= 5
        phase[7] -= 5
    return phase



def getPhaseFromAction3(phase, act):
    """
    6交叉口的动作选择方案 23个
    定周期配时：根据当前的相位和选择的动作调整得到新的相位配时方案
    输入：当前的配时方案phase 和 需要执行的动作操作 act
    输出：新的配时方案
    phase3 = [0, 60, 120, 0, 0, 0, 60, 60]
    """
    act = int(act)
    if act == 0:
        phase[2] += 5
        phase[3] -= 5
    elif act == 1:
        phase[2] -= 5
        phase[3] += 5
    elif act == 2:
        phase[6] += 5
        phase[7] -= 5
    elif act == 3:
        phase[6] -= 5
        phase[7] += 5
    elif act == 5:
        phase[0] += 5
        phase[5] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 6:
        phase[0] += 5
        phase[5] += 5
        phase[2] -= 5
        phase[7] -= 5
    elif act == 7:
        phase[0] += 5
        phase[5] += 5
        phase[3] -= 5
        phase[6] -= 5
    elif act == 8:
        phase[0] += 5
        phase[5] += 5
        phase[3] -= 5
        phase[7] -= 5
    elif act == 9:
        phase[4] += 5
        phase[1] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 10:
        phase[4] += 5
        phase[1] += 5
        phase[2] -= 5
        phase[7] -= 5
    elif act == 11:
        phase[4] += 5
        phase[1] += 5
        phase[3] -= 5
        phase[6] -= 5
    elif act == 12:
        phase[4] += 5
        phase[1] += 5
        phase[3] -= 5
        phase[7] -= 5
    elif act == 13:
        phase[0] -= 5
        phase[5] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 14:
        phase[0] -= 5
        phase[5] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 15:
        phase[0] -= 5
        phase[5] -= 5
        phase[3] += 5
        phase[6] += 5
    elif act == 16:
        phase[0] -= 5
        phase[5] -= 5
        phase[3] += 5
        phase[7] += 5
    elif act == 17:
        phase[4] -= 5
        phase[1] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 18:
        phase[4] -= 5
        phase[1] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 19:
        phase[4] -= 5
        phase[1] -= 5
        phase[3] += 5
        phase[6] += 5
    elif act == 20:
        phase[4] -= 5
        phase[1] -= 5
        phase[3] += 5
        phase[7] += 5
    elif act == 21:
        phase[0] -= 5
        phase[5] -= 5
        phase[4] += 5
        phase[1] += 5
    elif act == 22:
        phase[0] += 5
        phase[5] += 5
        phase[4] -= 5
        phase[1] -= 5
    return phase



def getPhaseFromAction4(phase, act):
    """
    5交叉口的动作选择方案 41个
    定周期配时：根据当前的相位和选择的动作调整得到新的相位配时方案
    输入：当前的配时方案phase 和 需要执行的动作操作 act
    输出：新的配时方案
    """
    act = int(act)
    # 将phase（0-8）两个一组共划分为4组([1,2][3,4][5,6][7,8])，act值域为0-8 
    if act > 0 and act < 9: 
        if act % 2 == 0 and act != 8: # act为偶数且不为8时，将4个小组的第一个数加上周期，第二个数减去周期。当act为8时，phase不变.
            phase[act] += 5
            phase[act+1] -= 5
        elif act % 2 == 1: # act为奇数时，则与偶数相反，第一个数减去周期，第二个数加上周期
            phase[act-1] -= 5
            phase[act] += 5
    # 表示barrier右移
    elif act == 9:
        phase[0] += 5
        phase[4] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 10:
        phase[0] += 5
        phase[4] += 5
        phase[2] -= 5
        phase[7] -= 5
    elif act == 11:
        phase[0] += 5
        phase[4] += 5
        phase[3] -= 5
        phase[6] -= 5
    elif act == 12:
        phase[0] += 5
        phase[4] += 5
        phase[3] -= 5
        phase[7] -= 5
    elif act == 13:
        phase[0] += 5
        phase[5] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 14:
        phase[0] += 5
        phase[5] += 5
        phase[2] -= 5
        phase[7] -= 5
    elif act == 15:
        phase[0] += 5
        phase[5] += 5
        phase[3] -= 5
        phase[6] -= 5
    elif act == 16:
        phase[0] += 5
        phase[5] += 5
        phase[3] -= 5
        phase[7] -= 5
    elif act == 17:
        phase[1] += 5
        phase[4] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 18:
        phase[1] += 5
        phase[4] += 5
        phase[2] -= 5
        phase[7] -= 5
    elif act == 19:
        phase[1] += 5
        phase[4] += 5
        phase[3] -= 5
        phase[6] -= 5
    elif act == 20:
        phase[1] += 5
        phase[4] += 5
        phase[3] -= 5
        phase[7] -= 5
    elif act == 21:
        phase[1] += 5
        phase[5] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 22:
        phase[1] += 5
        phase[5] += 5
        phase[2] -= 5
        phase[7] -= 5
    elif act == 23:
        phase[1] += 5
        phase[5] += 5
        phase[3] -= 5
        phase[6] -= 5
    elif act == 24:
        phase[1] += 5
        phase[5] += 5
        phase[3] -= 5
        phase[7] -= 5
    elif act == 25:
        phase[0] -= 5
        phase[4] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 26:
        phase[0] -= 5
        phase[4] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 27:
        phase[0] -= 5
        phase[4] -= 5
        phase[3] += 5
        phase[6] += 5
    elif act == 28:
        phase[0] -= 5
        phase[4] -= 5
        phase[3] += 5
        phase[7] += 5
    elif act == 29:
        phase[0] -= 5
        phase[5] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 30:
        phase[0] -= 5
        phase[5] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 31:
        phase[0] -= 5
        phase[5] -= 5
        phase[3] += 5
        phase[6] += 5
    elif act == 32:
        phase[0] -= 5
        phase[5] -= 5
        phase[3] += 5
        phase[7] += 5
    elif act == 33:
        phase[1] -= 5
        phase[4] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 34:
        phase[1] -= 5
        phase[4] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 35:
        phase[1] -= 5
        phase[4] -= 5
        phase[3] += 5
        phase[6] += 5
    elif act == 36:
        phase[1] -= 5
        phase[4] -= 5
        phase[3] += 5
        phase[7] += 5
    elif act == 37:
        phase[1] -= 5
        phase[5] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 38:
        phase[1] -= 5
        phase[5] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 39:
        phase[1] -= 5
        phase[5] -= 5
        phase[3] += 5
        phase[6] += 5
    elif act == 40:
        phase[1] -= 5
        phase[5] -= 5
        phase[3] += 5
        phase[7] += 5
    return phase



