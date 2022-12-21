import numpy as np


def sum_rewards(states, terminated, get_reward): # states表示所有状态，terminated表示是否终止，get_reward表示奖励函数
    rewards = [] # 存储奖励
    for i in range(len(states)): # 遍历所有状态
        rewards.append(get_reward(states[i], terminated[i])) # 计算奖励
    return rewards


def get_reward_1(state, terminated): 
    "Calculates the environments reward for the next state"

    if terminated: # 如果终止了，就返回-10
        return -10 # 为什么要返回-10呢？因为如果终止了，就说明没有达到目标，所以奖励应该是负的
    if state[0] > 0.25 and state[0] < 0.75: # 如果状态在0.25和0.75之间，就返回1
        return 1 
    return 0 # 如果不在0.25和0.75之间，就返回0


def get_reward_2(state, terminated):
    "Calculates the environments reward for the next state"
    if terminated:
        return -10 
    if state[0] > 0.4 and state[0] < 0.6:
        return 1
    return 0


def get_reward_3(state, terminated):
    "Calculates the environments reward for the next state"
    if terminated:
        return -10
    if state[0] > 0.45 and state[0] < 0.55:
        return 1
    return 0


def get_reward_ABS(state, terminated):
    "Calculates the environments reward for the next state"

    if terminated:
        return -10
    return np.absolute(0.5 - state[0]) # 0.5是目标位置，这里是绝对误差


def get_reward_SSE(state, terminated):
    "Calculates the environments reward for the next state"

    if terminated:
        return -10
    return (0.5 - state[0]) ** 2 # 0.5是目标位置，这里是平方误差
