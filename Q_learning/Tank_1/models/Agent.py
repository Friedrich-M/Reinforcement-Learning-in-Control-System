from collections import deque
import torch
from .Network import Net
import numpy as np
import random


class Agent:
    def __init__(self, AGENT_PARAMS):
        "Parameters are set in the params.py file"
        self.memory_size = AGENT_PARAMS["MEMORY_LENGTH"]  # 记忆的长度，用于存储之前的状态
        # 用于存储之前的状态，deque是一个双向队列，可以从两端添加和删除元素
        self.memory = deque(maxlen=self.memory_size)
        self.load_model = AGENT_PARAMS["LOAD_MODEL"]  # 是否加载模型
        self.model_name = AGENT_PARAMS["LOAD_MODEL_NAME"]  # 加载模型的名称
        self.save_model = AGENT_PARAMS["SAVE_MODEL"]  # 是否保存模型
        self.train_model = AGENT_PARAMS["TRAIN_MODEL"]  # 是否训练模型

        self.load_model_path = AGENT_PARAMS["LOAD_MODEL_PATH"]  # 加载模型的路径
        self.save_model_path = AGENT_PARAMS["SAVE_MODEL_PATH"]  # 保存模型的路径

        self.n_tanks = AGENT_PARAMS["N_TANKS"]  # 水箱的数量
        # 每个状态包含的观测值的数量，分别为水位高度，水位的变化率，水位是否大于水箱高度的一半，以及上一个时间步的阀门的位置
        self.state_size = AGENT_PARAMS["OBSERVATIONS"]
        self.action_state = None  # 当前阀门的位置
        self.action_size = AGENT_PARAMS["VALVE_POSITIONS"]  # 阀门可以选择的位置的数量
        self.action_choices = self._build_action_choices(
            self.action_size)  # 阀门所有可以选择的位置，即所有可供选择阀门开度值
        self.actions = None  # 当前状态下所有水箱的阀门的位置（即动作）
        self.action_delay_cnt = [9] * self.n_tanks  # 阀门的延迟时间，用于防止阀门的频繁开关
        self.action_delay = AGENT_PARAMS["ACTION_DELAY"]

        self.epsilon = AGENT_PARAMS["EPSILON"]  # 随机选择动作的概率，用于探索，初始值为 1
        self.epsilon_min = AGENT_PARAMS["EPSILON_MIN"]  # 随机选择动作的概率的最小值
        # 随机选择动作的概率的衰减率，每次训练后都会衰减
        self.epsilon_decay = AGENT_PARAMS["EPSILON_DECAY"]
        self.gamma = AGENT_PARAMS["GAMMA"]  # 折扣因子

        self.learning_rate = AGENT_PARAMS["LEARNING_RATE"]  # 学习率
        self.hl_size = AGENT_PARAMS["HIDDEN_LAYER_SIZE"]  # 隐藏层的神经元的数量
        self.batch_size = AGENT_PARAMS["BATCH_SIZE"]  # 每次训练的样本数量

        self.Q_eval, self.Q_next = [], []  # Q_eval 用于评估当前的状态，Q_next 用于评估之后的状态
        for i in range(self.n_tanks):  # 为每个水箱创建一个预测当前状态的Q值，预测下一个状态的Q值的神经网络
            Q_eval_, Q_next_ = self._build_ANN(
                self.state_size,
                self.hl_size,
                self.action_size,
                learning_rate=self.learning_rate,
                i=i,
            )  # 预测当前状态的Q值，预测下一个状态的Q值的神经网络
            self.Q_eval.append(Q_eval_)
            self.Q_next.append(Q_next_)

    def _build_action_choices(self, action_size):
        "Create a list of the valve positions ranging from 0-1"
        valve_positions = []  # 阀门的位置
        for i in range(action_size):  # 阀门的位置的数量
            # 可以选择的阀门的位置，分别为从 0 到 1，步长为 1/(action_size - 1)
            valve_positions.append((i) / (action_size - 1))
        return np.array(list(reversed(valve_positions)))  # 反转阀门的位置，从大到小

    def _build_ANN(  # 构建神经网络
        self, input_size, hidden_size, action_size, learning_rate, i
    ):  # input_size: 状态的数量，hidden_size: 隐藏层的数量，action_size: 阀门可以选择的位置的数量，learning_rate: 学习率，i: 第几个水箱
        if self.load_model[i]:  # 是否加载模型
            path = '/Users/mlz/Documents/智能控制/大作业/Reinforcement-Learning-in-Process-Control/Q_learning/Tank_1/Network_[5, 5]HL0.pt'
            pytorch_path = torch.load(path)  # 加载模型
            n_hl = (len(pytorch_path)-3)  # 隐藏层的数量
            if n_hl == 0:  # zero hidden later
                h_size = []
            elif n_hl == 1:  # 1 hidden layer
                h_size = [len(pytorch_path['input.weight'])]
            elif n_hl == 3:  # 2 hidden layers
                h_size = [len(pytorch_path['input.weight']),
                          len(pytorch_path['hl1.bias'])]
            else:
                raise ValueError
            Q_net = Net(input_size, h_size, action_size, learning_rate[i])
            Q_net.load_state_dict(pytorch_path)
            Q_net.eval()
            return Q_net, Q_net
        "Creates or loads a ANN valve function approximator"

        # 创建神经网络，用于评估当前的状态
        # 该预测网络为一个MLP，输入为状态，输出为不同动作的Q值，即Q(s,a)
        Q_eval = Net(input_size, hidden_size[i], action_size, learning_rate[i])
        # 创建神经网络，用于评估下一个状态
        Q_next = Net(input_size, hidden_size[i], action_size, learning_rate[i])
        return Q_eval, Q_next

    def get_z(self, action):  # 获取阀门的位置
        z = []
        for action in self.actions:  # 遍历每个动作
            z.append(self.action_choices[action])  # 阀门的开度
        return z

    def remember(self, states, reward, terminated, t):
        "Stores instances of each time step"

        replay = []  # 记录
        for i in range(self.n_tanks):  # 遍历每个水箱

            if terminated[i]:  # 如果水箱已经到达终止状态，即高于最高水位或低于最低水位
                # + 2 是因为 states 中包含了当前状态和下一个状态
                if len(states) <= self.action_delay[i] + 2:
                    action_state = states[i][0]
                else:
                    action_state_index = -self.action_delay_cnt[i] - 2
                    action_state = states[action_state_index][i]
                replay.append(
                    np.array(
                        [
                            action_state,  # 当前的状态
                            self.actions[i],  # 当前的动作
                            reward[i],  # 当前的奖励
                            states[-1][i],  # 下一个状态
                            terminated[i],  # 是否终止
                            False,  # 是否是最后一个状态
                            str(i) + "model",  # 第几个水箱
                        ]
                    )
                )

            elif (
                self.action_delay_cnt[i] >= self.action_delay[i]
                and t >= self.action_delay[i]
            ):
                action_state = states[-self.action_delay[i] - 2][i]
                replay.append(
                    np.array(
                        [
                            action_state,
                            self.actions[i],
                            reward[i],
                            states[-1][i],
                            terminated[i],
                            False,
                            str(i) + "model",
                        ]
                    )
                )
            elif True in terminated:

                action_state_index = -self.action_delay_cnt[i] - 2
                try:
                    action_state = states[action_state_index][i]
                except IndexError:
                    action_state = states[0][i]
                replay.append(
                    np.array(
                        [
                            action_state,
                            self.actions[i],
                            reward[i],
                            states[-1][i],
                            terminated[i],
                            False,
                            str(i) + "model",
                        ]
                    )
                )
        if True in terminated:
            self.memory.append(replay)
        elif not len(replay) == self.n_tanks:
            return
        else:
            self.memory.append(replay)

    def act_greedy(self, state, i):  # 采用贪婪策略，选择最优的动作
        "Predict the optimal action to take given the current state"

        choice = self.Q_eval[i].forward(state[i])  # 这里通过mlp网络预测当前状态下所有动作的效用值
        action = torch.argmax(choice).item()  # 选择最优的动作
        return action

    def act(self, state):  # 采用ε-greedy策略，选择动作
        """
        Agent uses the state and gives either an
        action of exploration or explotation
        """
        actions = []
        for i in range(self.n_tanks):  # 遍历每个水箱
            # 这里的 action_delay_cnt 是一个计数器，用于计算是否到达 action_delay
            if self.action_delay_cnt[i] >= self.action_delay[i]:
                self.action_delay_cnt[i] = 0  # 重置 action_delay_cnt 计数器为 0

                # ε-greedy方法：每个状态以ε的概率进行探索，此时将随机选取动作，而剩下的1-ε的概率则进行开发，此时将选择最优的动作
                # Exploration，随机选择动作
                if np.random.rand() <= float(self.epsilon[i]):
                    random_action = random.randint(
                        0, self.action_size - 1)
                    action = random_action
                    actions.append(action)
                else:
                    action = self.act_greedy(state, i)  # Exploitation，选择最优的动作
                    actions.append(action)
            else:
                # 如果还没有到达 action_delay，选择上一次的动作，即延迟 action_delay_cnt 个时间步再选择动作
                actions.append(self.actions[i])
                self.action_delay_cnt[i] += 1
        self.actions = actions
        return self.actions

    def is_ready(self):
        "Check if enough data has been collected"
        # 检查是否收集了足够的数据
        if len(self.memory) < self.batch_size:
            return False
        return True

    def Qreplay(self, e):
        """"
        Train the model to improve the predicted value of consecutive
        recurring states, Off policy Q-learning with batch training
        """
        minibatch = np.array(random.sample(self.memory, self.batch_size))
        for j in range(self.n_tanks):  # 遍历每个水箱
            if self.train_model[j]:
                agent_batch = minibatch[:, j]  # 从 batch 中取出该水箱的样本
                dummy_data = np.stack(agent_batch[:, 5])  
                dummy_data_index = np.where(dummy_data)[0] 
                agent_batch_comp = np.delete(
                    agent_batch, dummy_data_index, axis=0) 

                states = np.stack(agent_batch_comp[:, 0])  # 从样本中取出状态
                actions = np.stack(agent_batch_comp[:, 1])  # 从样本中取出动作
                rewards = np.stack(agent_batch_comp[:, 2])  # 从样本中取出奖励
                next_states = np.stack(agent_batch_comp[:, 3])  # 从样本中取出下一个状态
                terminated = np.stack(agent_batch_comp[:, 4])  # 从样本中取出是否终止的标志

                self.Q_eval[j].zero_grad()  # 清空梯度
                Qpred = self.Q_eval[j].forward(states).to(
                    self.Q_eval[j].device)  # 根据当前的状态，预测所有动作的效用值
                Qnext = (
                    self.Q_next[j].forward(next_states).to(
                        self.Q_next[j].device)  # 根据下一个状态，预测所有动作的效用值
                )

                maxA = Qnext.max(1)[1]  # 找出每个样本的最大效用值对应的动作
                rewards = torch.tensor(rewards, dtype=torch.float32).to(
                    self.Q_eval[j].device
                )  

                Q_target = Qpred.clone()  # Q_target 为目标值，即 Q-learning 的目标
                for i, Qnext_a in enumerate(maxA):  # 遍历每个样本
                    if not terminated[i]:  # 如果没有终止
                        Q_target[i, actions[i]] = rewards[
                            i
                        ] + self.gamma * torch.max(Qnext[i, Qnext_a])  # 新状态的效用值 = 奖励 + 折扣因子 * 下一个状态的最大效用值，更新 Q_target
                    else:
                        Q_target[i, actions[i]] = rewards[i]  # 如果该样本终止了，那么目标值就是奖励
                loss = (
                    self.Q_eval[j].loss(Qpred, Q_target).to(
                        self.Q_eval[j].device)  # 将当前的效用值与目标值进行比较，计算损失
                )
                loss.backward()  # 反向传播

                self.Q_eval[j].optimizer.step()  # 更新参数
                self.decay_exploration(j)  # 降低 epsilon 的值，以便于选择贪婪的动作，而不是随机的动作

    def decay_exploration(self, j):
        "Lower the epsilon value to favour greedy actions"
        if self.epsilon[j] > self.epsilon_min[j]:
            self.epsilon[j] = self.epsilon[j] * \
                self.epsilon_decay[j]

    def reset(self, init_state):
        self.action_state = init_state[0]  # 重置 action_state
        self.action = None  # 重置 action
        self.action_delay_cnt = self.action_delay  # 重置 action_delay_cnt

    def save_trained_model(self):
        "Save the model given a better model has been fitted"
        for i in range(self.n_tanks):
            if self.save_model[i]:
                model_name = "Network_" + \
                    str(self.hl_size[i]) + "HL" + str(i)  # 生成模型名称

                path = self.save_model_path + model_name + ".pt"  # 生成模型路径
                torch.save(self.Q_eval[i].state_dict(), path)  # 保存模型
        print("ANN_Model was saved")  # 打印保存模型的信息
