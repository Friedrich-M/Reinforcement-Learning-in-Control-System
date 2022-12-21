import numpy as np
from models.tank_model.disturbance import InflowDist


class Tank:  # 水箱类
    "Cylindric tank"
    g = 9.81  # 重力加速度
    rho = 1000  # 水的密度

    def __init__(
        self,
        height,  # 水箱的高度
        radius,  # 水箱的半径
        pipe_radius,  # 水管的半径
        max_level,  # 水箱的最大水位
        min_level,  # 水箱的最小水位
        init_level,  # 水箱的初始水位
        dist,  # 水箱的干扰输入
        prev_tank=None,  # 上一个水箱
    ):
        self.h = height  # 水箱的高度
        self.r = radius  # 水箱的半径
        self.A = radius ** 2 * np.pi  # 水箱的底面积

        self.init_l = height * init_level  # 水箱的初始水位
        self.level = self.init_l  # 水箱的当前水位，初始值为初始水位

        self.max = max_level * height  # 水箱的最大水位
        self.min = min_level * height  # 水箱的最小水位
        self.prev_q_out = 0  # 上一个水箱的输出流量

        self.A_pipe = pipe_radius ** 2 * np.pi  # 水管的横截面积
        self.add_dist = dist["add"]  # 是否添加干扰输入流量
        if dist["add"]:  # 如果添加输入流量
            self.dist = InflowDist(  # 输入流量的分布
                nom_flow=dist["nom_flow"],  # 输入流量的均值
                var_flow=dist["var_flow"],  # 输入流量的方差
                max_flow=dist["max_flow"],  # 输入流量的最大值
                min_flow=dist["min_flow"],  # 输入流量的最小值
                add_step=dist["add_step"],  # 是否添加阶梯输入流量
                step_flow=dist["step_flow"],  # 阶梯输入流量的大小
                step_time=dist["step_time"],  # 阶梯输入流量的时间
                pre_def_dist=dist["pre_def_dist"],  # 是否使用预置的干扰输入流量分布
                max_time=dist["max_time"],  # 输入流量的最大时间
            )

    def change_level(self, dldt):
        self.level += dldt * self.h  # 水箱的当前水位 = 水箱的当前水位 + 水箱的水位变化率 * 水箱的高度

    def get_dhdt(self, action, t, prev_q_out):  # action:阀门开度，t:时间，prev_q_out:上一个水箱的输出流量
        "Calculates the change in water level"
        if self.add_dist:  # 如果添加干扰输入流量
            # 输入流量 = 干扰输入流量 + 上一个水箱的输出流量
            q_inn = self.dist.get_flow(t) + prev_q_out
        else:  # 如果不添加干扰输入流量
            q_inn = prev_q_out  # 输入流量 = 上一个水箱的输出流量

        # 获取水箱的参数，f:阀门开度，A_pipe:水管横截面积，g:重力加速度，l:水箱的当前水位，delta_p:压力差，rho:水的密度，r:水箱的半径
        f, A_pipe, g, l, delta_p, rho, r = self.get_params(action)
        # 输出流量 = 阀门开度 * 水管横截面积 * 根号(1 * 重力加速度 * 水箱的当前水位 + 压力差 / 水的密度)
        q_out = f * A_pipe * np.sqrt(1 * g * l + delta_p / rho)

        term1 = q_inn / (np.pi * r ** 2)  # 流入流量 / 底部圆面积 = 输入流量使水位上升的level
        term2 = (q_out) / (np.pi * r ** 2)  # 流出流量 / 底部圆面积 = 输出流量使水位下降的level
        new_level = term1 - term2  # 水箱的水位变化 = 输入流量使水位上升的level - 输出流量使水位下降的level
        return new_level, q_out  # 返回当前水箱的水位变化率和当前水箱的输出流量

    def reset(self):
        "reset tank to initial liquid level"
        self.level = self.init_l  # 水箱的当前水位 = 水箱的初始水位

    def get_valve(self, action):
        "linear valve equation"
        return action  # 动作 = 当前阀门开度

    def get_params(self, action):
        "collects the tanks parameters"
        f = self.get_valve(action)  # 动作，即阀门开度
        # 返回水箱的参数，f:阀门开度，A_pipe:水管横截面积，g:重力加速度，l:水箱的当前水位，delta_p:压力差，rho:水的密度，r:水箱的半径
        return f, self.A_pipe, Tank.g, self.level, 0, Tank.rho, self.r
