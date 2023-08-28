def not_change(phase,tmin=30,tmax=70):
    not_change=[]
    for i,value in enumerate(phase):
        if value<=tmin or value>=tmax:
            not_change.append(i)
    return not_change


def reset_phase_check(phase,init_phase,tmin=30,tmax=70):
    cunt=0
    for i, value in enumerate(phase):
        if value<=tmin or value>=tmax:
            cunt+=1
    if cunt/len(phase)>=0.5:
        return init_phase
    return phase


def Compute(i, op, j):
    if op == '+':
        return i + j
    elif op == '-':
        return i - j
    elif op == '*':
        return i * j
    elif op == '/':
        return i / j
    else:
        return None


def ActPhase(phase, indx, ops, tmin, tmax):
    new_phase = []
    for i in indx:
        if Compute(phase[i], ops[i], 5) < tmin or Compute(phase[i], ops[i], 5) > tmax:
            return phase
        else:
            new_phase.append(Compute(phase[i], ops[i], 5))

    assert len(new_phase) == len(phase)
    return new_phase


def getPhaseFromAction1(phase, act,init_phase,tmin,tmax):

    phase=reset_phase_check(phase,init_phase, tmin=tmin, tmax=tmax)
    not_allow_change=not_change(phase, tmin=tmin, tmax=tmax)
    print('phase',phase)
    print('not',not_allow_change)
    act = int(act)

    if act == 0:
        indx=[0,1,4,5]
        ops=['+','-','+','-']
        phase=ActPhase(phase, indx, ops, tmin, tmax)
    elif act == 1:
        indx = [0, 1, 4, 5]
        ops = ['-', '+', '-', '+']
        phase = ActPhase(phase, indx, ops, tmin, tmax)
    elif act == 2:
        indx = [2,0,4]
        ops = ['+', '-','-']
        phase = ActPhase(phase, indx, ops, tmin, tmax)
        phase[2]+=5
        phase[0]-=5
        phase[4]-=5
    elif act == 3:
        phase[2] += 5
        phase[0] -= 5
        phase[5] -= 5
    elif act == 4:
        phase[2] += 5
        phase[1] -= 5
        phase[4] -= 5
    elif act == 5:
        phase[2] += 5
        phase[1] -= 5
        phase[5] -= 5
    elif act == 6:
        phase[6] += 5
        phase[0] -= 5
        phase[4] -= 5
    elif act == 7:
        phase[6] += 5
        phase[0] -= 5
        phase[5] -= 5
    elif act == 8:
        phase[6] += 5
        phase[1] -= 5
        phase[4] -= 5
    elif act == 9:
        phase[6] += 5
        phase[1] -= 5
        phase[5] -= 5
    elif act == 10:
        phase[2] -= 5
        phase[0] += 5
        phase[4] += 5
    elif act == 11:
        phase[2] -= 5
        phase[0] += 5
        phase[5] += 5
    elif act == 12:
        phase[2] -= 5
        phase[1] += 5
        phase[4] += 5
    elif act == 13:
        phase[2] -= 5
        phase[1] += 5
        phase[5] += 5
    elif act == 14:
        phase[6] -= 5
        phase[0] += 5
        phase[4] += 5
    elif act == 15:
        phase[6] -= 5
        phase[0] += 5
        phase[5] += 5
    elif act == 16:
        phase[6] -= 5
        phase[1] += 5
        phase[4] += 5
    elif act == 17:
        phase[6] -= 5
        phase[1] += 5
        phase[5] += 5
    elif act == 18:
        phase[2] -= 5
        phase[6] += 5
    elif act == 19:
        phase[2] += 5
        phase[6] -= 5
    return phase

def getPhaseFromAction2(phase, act,init_phase,tmin,tmax):
    """
    2交叉口的动作选择方案 7个
    定周期配时：根据当前的相位和选择的动作调整得到新的相位配时方案
    输入：当前的配时方案phase 和 需要执行的动作操作 act
    输出：新的配时方案

    phase2 = [0, 60, 120, 0, 0, 0, 60, 60]
    """
    #[1,6,7]
    phase = reset_phase_check(phase, init_phase, tmin=tmin, tmax=tmax)
    not_allow_change = not_change(phase, tmin=tmin, tmax=tmax)
    act = int(act)
    if act == 0 and not any([6,7]) in not_allow_change:
        phase[6] += 5
        phase[7] -= 5
    elif act == 1 and not any([6,7]) in not_allow_change:
        phase[6] -= 5
        phase[7] += 5
    elif act == 2 and not any([1,6]) in not_allow_change:
        phase[1] -= 5
        phase[6] += 5
    elif act == 3 and not any([1,7]) in not_allow_change:
        phase[1] -= 5
        phase[7] += 5
    elif act == 4 and not any([1,6]) in not_allow_change:
        phase[1] += 5
        phase[6] -= 5
    elif act == 5 and not any([1,7]) in not_allow_change:
        phase[1] += 5
        phase[7] -= 5
    return phase



def getPhaseFromAction3(phase, act,init_phase,tmin,tmax):
    """
    6交叉口的动作选择方案 23个
    定周期配时：根据当前的相位和选择的动作调整得到新的相位配时方案
    输入：当前的配时方案phase 和 需要执行的动作操作 act
    输出：新的配时方案
    phase3 = [0, 60, 120, 0, 0, 0, 60, 60]
    """
    #[0,2,3,4,6]
    phase = reset_phase_check(phase, init_phase, tmin=tmin, tmax=tmax)
    not_allow_change = not_change(phase, tmin=tmin, tmax=tmax)
    act = int(act)
    if act == 0:
        if not any([2,3]) in not_allow_change:
            phase[2] += 5
            phase[3] -= 5
        if not any([6,7]) in not_allow_change:
            phase[6] += 5
            phase[7] -= 5
    elif act == 1:
        if not any([2, 3]) in not_allow_change:
            phase[2] -= 5
            phase[3] += 5
        if not any([6,7]) in not_allow_change:
            phase[6] -= 5
            phase[7] += 5
    elif act == 2 and not any([0,2,6]) in not_allow_change:
        phase[0] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 3 and not any([0,2,7]) in not_allow_change:
        phase[0] += 5
        phase[2] -= 5
        phase[7] -= 5
    elif act == 4 and not any([0,3,6]) in not_allow_change:
        phase[0] += 5
        phase[3] -= 5
        phase[6] -= 5
    elif act == 5 and not any([0,3,7]) in not_allow_change:
        phase[0] += 5
        phase[3] -= 5
        phase[7] -= 5
    elif act == 6 and not any([4,2,6]) in not_allow_change:
        phase[4] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 7 and not any([4,2,7]) in not_allow_change:
        phase[4] += 5
        phase[2] -= 5
        phase[7] -= 5
    elif act == 8 and not any([4,3,6]) in not_allow_change:
        phase[4] += 5
        phase[3] -= 5
        phase[6] -= 5
    elif act == 9 and not any([4,3,7]) in not_allow_change:
        phase[4] += 5
        phase[3] -= 5
        phase[7] -= 5
    elif act == 10 and not any([0,2,6]) in not_allow_change:
        phase[0] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 11 and not any([0,2,7]) in not_allow_change:
        phase[0] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 12 and not any([0,3,6]) in not_allow_change:
        phase[0] -= 5
        phase[3] += 5
        phase[6] += 5
    elif act == 13 and not any([0,3,7]) in not_allow_change:
        phase[0] -= 5
        phase[3] += 5
        phase[7] += 5
    elif act == 14 and not any([4,2,6]) in not_allow_change:
        phase[4] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 15 and not any([4,2,7]) in not_allow_change:
        phase[4] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 16 and not any([4,3,6]) in not_allow_change:
        phase[4] -= 5
        phase[3] += 5
        phase[6] += 5
    elif act == 17 and not any([4,3,7]) in not_allow_change:
        phase[4] -= 5
        phase[3] += 5
        phase[7] += 5
    elif act == 18 and not any([0,4]) in not_allow_change:
        phase[0] -= 5
        phase[4] += 5
    elif act == 19 and not any([0,4]) in not_allow_change:
        phase[0] += 5
        phase[4] -= 5
    return phase



def getPhaseFromAction4(phase, act,init_phase,tmin,tmax):
    """
    5交叉口的动作选择方案 41个
    定周期配时：根据当前的相位和选择的动作调整得到新的相位配时方案
    输入：当前的配时方案phase 和 需要执行的动作操作 act
    输出：新的配时方案
    """

    phase = reset_phase_check(phase, init_phase, tmin=tmin, tmax=tmax)
    not_allow_change = not_change(phase, tmin=tmin, tmax=tmax)
    act = int(act)
    if act >= 0 and act < 9 :
        if act % 2 == 0 and act != 8 and not any([act,act+1]) in not_allow_change:
            phase[act] += 5
            phase[act+1] -= 5
        elif act % 2 == 1 and not any([act-1,act]) in not_allow_change:
            phase[act-1] -= 5
            phase[act] += 5

    elif act == 9 and not any([0,4,2,6]) in not_allow_change:
        phase[0] += 5
        phase[4] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 10 and not any([0,4,2,7]) in not_allow_change:
        phase[0] += 5
        phase[4] += 5
        phase[2] -= 5
        phase[7] -= 5
    elif act == 11 and not any([0,4,3,6]) in not_allow_change:
        phase[0] += 5
        phase[4] += 5
        phase[3] -= 5
        phase[6] -= 5
    elif act == 12 and not any([0,4,3,7]) in not_allow_change:
        phase[0] += 5
        phase[4] += 5
        phase[3] -= 5
        phase[7] -= 5
    elif act == 13 and not any([0,5,2,6]) in not_allow_change:
        phase[0] += 5
        phase[5] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 14 and not any([0,5,2,7]) in not_allow_change:
        phase[0] += 5
        phase[5] += 5
        phase[2] -= 5
        phase[7] -= 5
    elif act == 15 and not any([0,5,3,6]) in not_allow_change:
        phase[0] += 5
        phase[5] += 5
        phase[3] -= 5
        phase[6] -= 5
    elif act == 16 and not any([0,5,3,7]) in not_allow_change:
        phase[0] += 5
        phase[5] += 5
        phase[3] -= 5
        phase[7] -= 5
    elif act == 17 and not any([1,4,2,6]) in not_allow_change:
        phase[1] += 5
        phase[4] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 18 and not any([1,4,2,7]) in not_allow_change:
        phase[1] += 5
        phase[4] += 5
        phase[2] -= 5
        phase[7] -= 5
    elif act == 19 and not any([1,4,3,6]) in not_allow_change:
        phase[1] += 5
        phase[4] += 5
        phase[3] -= 5
        phase[6] -= 5
    elif act == 20 and not any([1,4,3,7]) in not_allow_change:
        phase[1] += 5
        phase[4] += 5
        phase[3] -= 5
        phase[7] -= 5
    elif act == 21 and not any([1,5,2,6]) in not_allow_change:
        phase[1] += 5
        phase[5] += 5
        phase[2] -= 5
        phase[6] -= 5
    elif act == 22 and not any([1,5,2,7]) in not_allow_change:
        phase[1] += 5
        phase[5] += 5
        phase[2] -= 5
        phase[7] -= 5
    elif act == 23 and not any([1,5,3,6]) in not_allow_change:
        phase[1] += 5
        phase[5] += 5
        phase[3] -= 5
        phase[6] -= 5
    elif act == 24 and not any([1,5,3,7]) in not_allow_change:
        phase[1] += 5
        phase[5] += 5
        phase[3] -= 5
        phase[7] -= 5
    elif act == 25 and not any([0,4,2,6]) in not_allow_change:
        phase[0] -= 5
        phase[4] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 26 and not any([0,4,2,7]) in not_allow_change:
        phase[0] -= 5
        phase[4] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 27 and not any([0,4,3,6]) in not_allow_change:
        phase[0] -= 5
        phase[4] -= 5
        phase[3] += 5
        phase[6] += 5
    elif act == 28 and not any([0,4,3,7]) in not_allow_change:
        phase[0] -= 5
        phase[4] -= 5
        phase[3] += 5
        phase[7] += 5
    elif act == 29 and not any([0,5,2,6]) in not_allow_change:
        phase[0] -= 5
        phase[5] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 30 and not any([0,5,2,7]) in not_allow_change:
        phase[0] -= 5
        phase[5] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 31 and not any([0,5,3,6]) in not_allow_change:
        phase[0] -= 5
        phase[5] -= 5
        phase[3] += 5
        phase[6] += 5
    elif act == 32 and not any([0,5,3,7]) in not_allow_change:
        phase[0] -= 5
        phase[5] -= 5
        phase[3] += 5
        phase[7] += 5
    elif act == 33 and not any([1,4,2,6]) in not_allow_change:
        phase[1] -= 5
        phase[4] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 34 and not any([1,4,2,7]) in not_allow_change:
        phase[1] -= 5
        phase[4] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 35 and not any([1,4,3,6]) in not_allow_change:
        phase[1] -= 5
        phase[4] -= 5
        phase[3] += 5
        phase[6] += 5
    elif act == 36 and not any([1,4,3,7]) in not_allow_change:
        phase[1] -= 5
        phase[4] -= 5
        phase[3] += 5
        phase[7] += 5
    elif act == 37 and not any([1,5,2,6]) in not_allow_change:
        phase[1] -= 5
        phase[5] -= 5
        phase[2] += 5
        phase[6] += 5
    elif act == 38 and not any([1,5,2,7]) in not_allow_change:
        phase[1] -= 5
        phase[5] -= 5
        phase[2] += 5
        phase[7] += 5
    elif act == 39 and not any([1,5,3,6]) in not_allow_change:
        phase[1] -= 5
        phase[5] -= 5
        phase[3] += 5
        phase[6] += 5
    elif act == 40 and not any([1,5,3,7]) in not_allow_change:
        phase[1] -= 5
        phase[5] -= 5
        phase[3] += 5
        phase[7] += 5
    return phase



