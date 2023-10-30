import copy


def reset_phase_check(phase,used,init_phase,tmin=30,tmax=70):
    cunt=0
    for i, value in enumerate(phase):
        if value<=tmin or value>=tmax:
            cunt+=1
    if cunt/used>=0.6:
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
    new_phase=copy.deepcopy(phase)
    for r,i in enumerate(indx):
        min=tmin
        # if i%2==0: #直行车道的下限多5
        #     min=tmin+5

        if Compute(phase[i], ops[r], 5) < min or Compute(phase[i], ops[r], 5) > tmax:
            return phase,False
        else:
            new_phase[i]=Compute(phase[i], ops[r], 5)
    flag=True
    return new_phase,flag


def getPhaseFromAction1(phase,act,init_phase,tmin,tmax):
    # used=6
    # phase=reset_phase_check(phase,used,init_phase, tmin=tmin, tmax=tmax)
    act = int(act)
    indx=list(range(8))
    ops=['+']*8

    if act == 0:
        indx=[0,1]
        ops=['+','-']
    elif act == 1:
        indx = [4,5]
        ops = ['+', '-']
    elif act == 2:
        indx = [0, 1]
        ops = ['-', '+']
    elif act == 3:
        indx = [4,5]
        ops = ['-', '+']
    elif act == 4:
        indx = [2,0,4]
        ops = ['+', '-','-']
    elif act == 5:
        indx = [2, 0, 5]
        ops = ['+', '-', '-']
    elif act == 6:
        indx = [2, 1, 4]
        ops = ['+', '-', '-']
    elif act == 7:
        indx = [2, 0, 5]
        ops = ['+', '-', '-']
    elif act == 8:
        indx = [6, 0, 4]
        ops = ['+', '-', '-']
    elif act == 9:
        indx = [6, 0, 5]
        ops = ['+', '-', '-']
    elif act == 10:
        indx = [6,1,4]
        ops = ['+', '-', '-']
    elif act == 11:
        indx = [6,1,5]
        ops = ['+', '-', '-']
    elif act == 12:
        indx = [2, 0, 4]
        ops = ['-', '+', '+']
    elif act == 13:
        indx = [2, 0, 5]
        ops = ['-', '+', '+']
    elif act == 14:
        indx = [2, 1, 4]
        ops = ['-', '+', '+']
    elif act == 15:
        indx = [2, 1, 4]
        ops = ['-', '+', '+']
    elif act == 16:
        indx = [6, 0, 4]
        ops = ['-', '+', '+']
    elif act == 17:
        indx = [6, 0, 5]
        ops = ['-', '+', '+']
    elif act == 18:
        indx = [6, 1, 4]
        ops = ['-', '+', '+']
    elif act == 19:
        indx = [6, 1, 5]
        ops = ['-', '+', '+']
    elif act == 20:
        indx = [2,6]
        ops = ['-', '+']
    elif act == 21:
        indx = [2, 6]
        ops = ['+', '-']
    phase,flag = ActPhase(phase, indx, ops, tmin, tmax)
    # print(flag)
    return phase

def getPhaseFromAction2(phase, act,init_phase,tmin,tmax):
    # used=3
    # phase = reset_phase_check(phase, used, init_phase, tmin=tmin, tmax=tmax)
    act = int(act)
    indx=list(range(8))
    ops=['+']*8
    act = int(act)
    # print(act)
    if act == 0:
        indx = [6,7]
        ops = ['+', '-']
    elif act == 1:
        indx = [6, 7]
        ops = ['-', '+']
    elif act == 2:
        indx = [1,6]
        ops = ['-', '+']
    elif act == 3:
        indx = [1, 7]
        ops = ['-', '+']
    elif act == 4:
        indx = [1, 6]
        ops = ['+', '-']
    elif act == 5:
        indx = [1, 7]
        ops = ['+', '-']
    phase,flag = ActPhase(phase, indx, ops, tmin, tmax)
    # print(flag)
    return phase



def getPhaseFromAction3(phase, act,init_phase,tmin,tmax):
    # used=5
    # phase=reset_phase_check(phase,used,init_phase, tmin=tmin, tmax=tmax)
    act = int(act)
    indx=list(range(8))
    ops=['+']*8
    act = int(act)

    if act == 0:
        indx = [2,3]
        ops = ['+', '-']
    elif act == 1:
        indx = [6,7]
        ops = ['+', '-']
    elif act == 2:
        indx = [2, 3]
        ops = ['-', '+']
    elif act == 3:
        indx = [6,7]
        ops = ['-', '+']
    elif act == 4:
        indx = [0,2,6]
        ops = ['+', '-','-']
    elif act == 5:
        indx = [0, 2, 7]
        ops = ['+', '-', '-']
    elif act == 6:
        indx = [0, 3, 6]
        ops = ['+', '-', '-']
    elif act == 7:
        indx = [0,3,7]
        ops = ['+', '-', '-']
    elif act == 8:
        indx = [4, 2, 6]
        ops = ['+', '-', '-']
    elif act == 9:
        indx = [4, 2, 7]
        ops = ['+', '-', '-']
    elif act == 10:
        indx = [4,3, 6]
        ops = ['+', '-', '-']
    elif act == 11:
        indx = [4, 3, 7]
        ops = ['+', '-', '-']
    elif act == 12:
        indx = [0,2, 6]
        ops = ['-', '+','+']
    elif act == 13:
        indx = [0, 2, 7]
        ops = ['-', '+', '+']
    elif act == 14:
        indx = [0, 3, 6]
        ops = ['-', '+', '+']
    elif act == 15:
        indx = [0, 3,7]
        ops = ['-', '+', '+']
    elif act == 16:
        indx = [4, 2, 6]
        ops = ['-', '+', '+']
    elif act == 17:
        indx = [4,2,7]
        ops = ['-', '+', '+']
    elif act == 18:
        indx = [4,3,6]
        ops = ['-', '+', '+']
    elif act == 19:
        indx = [4,3,7]
        ops = ['-', '+', '+']
    elif act ==20:
        indx = [0, 4]
        ops = ['-', '+']
    elif act ==21:
        indx = [0,4]
        ops = ['+', '-']
    phase, flag = ActPhase(phase, indx, ops, tmin, tmax)
    # print(flag)
    return phase



def getPhaseFromAction4(phase, act,init_phase,tmin,tmax):
    # used=8
    # phase = reset_phase_check(phase,used,init_phase, tmin=tmin, tmax=tmax)
    indx = list(range(8))
    ops = ['+'] * 8
    act = int(act)
    if act >= 0 and act < 8 :
        if act % 2 == 0:
            indx = [act,act+1]
            ops = ['+', '-']
        elif act % 2 == 1:
            indx = [act-1, act]
            ops = ['-', '+']

    elif act == 8:
        indx = [0,4,2,6]
        ops = ['+', '+','-','-']
    elif act == 9:
        indx = [0, 4, 2, 7]
        ops = ['+', '+', '-', '-']
    elif act == 10:
        indx = [0, 4, 3, 6]
        ops = ['+', '+', '-', '-']
    elif act == 11:
        indx = [0, 4, 3,7]
        ops = ['+', '+', '-', '-']
    elif act == 12:
        indx = [0, 5, 2, 6]
        ops = ['+', '+', '-', '-']
    elif act == 13:
        indx = [0, 5, 2, 7]
        ops = ['+', '+', '-', '-']
    elif act == 14:
        indx = [0, 5,3,6]
        ops = ['+', '+', '-', '-']
    elif act == 15:
        indx = [0,5,3,7]
        ops = ['+', '+', '-', '-']
    elif act == 16:
        indx = [1,4,2,6]
        ops = ['+', '+', '-', '-']
    elif act == 17:
        indx = [1,4,2,7]
        ops = ['+', '+', '-', '-']
    elif act == 18:
        indx = [1,4,3,6]
        ops = ['+', '+', '-', '-']
    elif act == 19:
        indx = [1,4, 3, 7]
        ops = ['+', '+', '-', '-']
    elif act == 20:
        indx = [1,5,2,6]
        ops = ['+', '+', '-', '-']
    elif act == 21:
        indx = [1,5,2,7]
        ops = ['+', '+', '-', '-']
    elif act == 22:
        indx = [1,5,3,6]
        ops = ['+', '+', '-', '-']
    elif act == 23:
        indx = [1, 5, 3,7]
        ops = ['+', '+', '-', '-']
    elif act == 24:
        indx = [0,4,2,6]
        ops = ['-', '-', '+', '+']
    elif act == 25:
        indx = [0, 4, 2, 7]
        ops = ['-', '-', '+', '+']
    elif act == 26:
        indx = [0, 4, 3, 6]
        ops = ['-', '-', '+', '+']
    elif act == 27:
        indx = [0, 4, 3,7]
        ops = ['-', '-', '+', '+']
    elif act == 28:
        indx = [0, 5, 2, 6]
        ops = ['-', '-', '+', '+']
    elif act == 29:
        indx = [0, 5, 2, 6]
        ops = ['-', '-', '+', '+']
    elif act == 30:
        indx = [0, 5,3, 6]
        ops = ['-', '-', '+', '+']
    elif act == 31:
        indx = [0, 5,3,7]
        ops = ['-', '-', '+', '+']
    elif act == 32:
        indx = [1, 4, 2, 6]
        ops = ['-', '-', '+', '+']
    elif act == 33:
        indx = [1, 4, 2, 7]
        ops = ['-', '-', '+', '+']
    elif act == 34:
        indx = [1,4,3,6]
        ops = ['-', '-', '+', '+']
    elif act == 35:
        indx = [1,4,3,7]
        ops = ['-', '-', '+', '+']
    elif act == 36:
        indx = [1,5,2,6]
        ops = ['-', '-', '+', '+']
    elif act == 37:
        indx = [1,5,2,7]
        ops = ['-', '-', '+', '+']
    elif act == 38:
        indx = [1,5,3,6]
        ops = ['-', '-', '+', '+']
    elif act == 39:
        indx = [1,5,3,7]
        ops = ['-', '-', '+', '+']
    phase, flag = ActPhase(phase, indx, ops, tmin, tmax)
    # print(flag)
    return phase



