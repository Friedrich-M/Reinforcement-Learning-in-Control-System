from models.tank_model.tank import Tank
from visualize.window import Window
import numpy as np


class Environment: # 表示环境
    "Parameters are set in the params.py file"

    def __init__(self, TANK_PARAMS, TANK_DIST, MAIN_PARAMS):
        self.tanks = [] # 水箱列表
        for i, PARAMS in enumerate(TANK_PARAMS):
            tank = Tank( # 初始化水箱
                height=PARAMS["height"],  # 水箱高度
                radius=PARAMS["width"], # 水箱半径
                max_level=PARAMS["max_level"], # 水箱最大水位
                min_level=PARAMS["min_level"], # 水箱最小水位
                pipe_radius=PARAMS["pipe_radius"], # 水箱管道半径
                init_level=PARAMS["init_level"], # 水箱初始水位
                dist=TANK_DIST[i], # 水箱的干扰输入流量
            )
            self.tanks.append(tank) # 将水箱添加到水箱列表中
        self.n_tanks = len(self.tanks) # 表示水箱的数量
        self.running = True # 表示是否运行
        self.terminated = [False] * self.n_tanks # 表示每个水箱是否已经到达最大或最小水位
        self.q_inn = [0] * (self.n_tanks + 1) # 表示每个水箱的输入流量，q_inn[0]表示第一个水箱的输入流量，q_inn[1]表示第二个水箱的输入流量，以此类推，最后一个元素表示最后一个水箱的输出流量
        self.show_rendering = MAIN_PARAMS["RENDER"] # 是否显示渲染，即是否显示水箱的水位

        if self.show_rendering: # 如果显示渲染，则初始化窗口
            self.window = Window(self.tanks) # 初始化窗口

    def get_next_state(self, z, state, t): # z表示上一个状态空间中每个水箱的动作，即阀门的开度，state表示上一个状态空间，t表示当前时间
        """
        Calculates the dynamics of the agents action
        and gives back the next state
        """
        next_state = [] # 表示下一个状态
        prev_q_out = 0 # 表示上一个水箱的输出流量
        for i in range(self.n_tanks):
            dldt, q_out = self.tanks[i].get_dhdt(z[i], t, prev_q_out) # 根据当前水箱采取的动作和前一个水箱的输出流量，计算当前水箱的水位变化率和当前水箱的输出流量
            self.q_inn[i + 1] = q_out # 更新下一个水箱的输入流量，即当前水箱的输出流量
            self.tanks[i].change_level(dldt) # 更新当前水箱的水位
            z_ = 0 if i == 0 else z[i - 1] # 表示上一个水箱的动作，即阀门的开度，如果当前水箱是第一个水箱，则上一个水箱的动作为0，即阀门全开，如果当前水箱不是第一个水箱，则上一个水箱的动作为z[i-1]
            # Check terminate state 检查是否到达最大或最小水位
            if self.tanks[i].level < self.tanks[i].min: # 如果水箱的水位小于最小水位，则表示水箱已经到达最小水位
                self.terminated[i] = True # 表示水箱已经到达最小水位
                self.tanks[i].level = self.tanks[i].min # 更新水箱的水位
            elif self.tanks[i].level > self.tanks[i].max: # 如果水箱的水位大于最大水位，则表示水箱已经到达最大水位
                self.terminated[i] = True # 表示水箱已经到达最大水位
                self.tanks[i].level = self.tanks[i].max # 更新水箱的水位

            grad = (dldt + 0.1) / 0.2 # 表示水位变化率的归一化值，水位变化
            if self.tanks[i].level > 0.5 * self.tanks[i].h: # 如果水箱的水位大于水箱的一半
                above = 1 # 表示水箱的水位大于水箱的一半
            else:
                above = 0 # 表示水箱的水位小于水箱的一半

            next_state.append(
                np.array(
                    [self.tanks[i].level / self.tanks[i].h, grad, above, z_] # 表示当前水箱水位的归一化值，水位变化率的归一化值，水箱的水位是否大于水箱的一半，上一个水箱的动作
                )
            )
        return self.terminated, next_state # 返回是否到达终止状态，以及下一个状态

    def reset(self):
        "Reset the environment to the initial tank level and disturbance"
        # 将水箱的水位重置为初始化水位，将输入的分布重置为初始化分布
        init_state = [] # 表示初始化状态
        self.terminated = [False] * self.n_tanks # 表示是否到达终止状态,初始化为False
        for i in range(self.n_tanks): # 遍历每一个水箱
            self.tanks[i].reset()  # reset to initial tank level 
            if self.tanks[i].add_dist: # 如果水箱添加了输入
                self.tanks[i].dist.reset()  # reset to nominal disturbance # 重置干扰为初始化干扰
            init_state.append(
                np.array([self.tanks[i].init_l / self.tanks[i].h, 0, 1, 0]) # 表示水箱的水位的归一化值，水位变化率的归一化值，水箱的水位大于水箱的一半，上一个水箱的动作
            )  # Level plus gradient
        return [init_state], [] # 返回初始化状态

    def render(self, action):
        "Draw the water level of the tank in pygame"

        if self.show_rendering: # 如果显示渲染
            running = self.window.Draw(action) # 绘制窗口
            if not running: # 如果窗口关闭
                self.running = False # 表示环境关闭
