import numpy as np


class InflowDist:
    "Inlet disturbance flow"  # 输入流量的扰动

    def __init__(
        self,
        nom_flow,  # 输入扰动流量的均值
        var_flow,  # 输入扰动流量的方差
        max_flow,  # 输入扰动流量的最大值
        min_flow,  # 输入扰动流量的最小值
        add_step,  # 是否添加阶梯扰动
        step_flow,  # 阶梯扰动的流量
        step_time,  # 阶梯扰动的时间
        pre_def_dist,  # 是否使用预定义的扰动流量
        max_time,  # 最大时间
    ):
        self.var_flow = var_flow
        self.nom_flow = nom_flow
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.max_time = max_time
        self.flow = [nom_flow]  # 初始化输入扰动流量为流量的均值
        self.add_step = add_step
        if self.add_step:
            self.step_flow = step_flow
            self.step_time = step_time
        self.pre_def_dist = pre_def_dist
        if self.pre_def_dist:  # 如果是预定义的扰动流量
            csv_name = "disturbance_" + str(self.max_time) + ".csv"
            self.flow = np.genfromtxt(csv_name, delimiter=",")  # 读取csv文件

    def get_flow(self, t):
        "Gausian distribution of flow rate"
        if self.pre_def_dist:  # 如果是预定义的扰动流量
            return self.flow[t]  # 返回预定义的扰动流量
        else:
            if self.add_step:  # 如果添加了阶梯扰动
                if t > self.step_time:  # 如果时间大于阶梯扰动的时间，添加阶梯扰动，否则不添加
                    self.flow.append(self.step_flow)  # 添加阶梯扰动的流量
                    self.max_flow = self.step_flow  # 最大流量为阶梯扰动的流量
                    self.add_step = False  # 添加阶梯扰动的标志位为False
            new_flow = np.random.normal(
                self.flow[-1], self.var_flow)  # 生成一个服从正态分布的扰动流量
            if new_flow > self.max_flow:  # 如果扰动流量大于最大流量，则返回最大流量，这是为了防止流量过大，导致设备损坏
                self.flow.append(self.max_flow)  
                return self.flow[-1]  
            elif new_flow < self.min_flow:  # 如果扰动流量小于最小流量，则返回最小流量，这是为了防止流量过小，导致设备损坏
                self.flow.append(self.min_flow) 
                return self.flow[-1]  
            else:
                self.flow.append(new_flow)  # 添加扰动流量
                return self.flow[-1]  # 返回扰动流量

    def reset(self):
        "Sets dstubance flow to nominal value"
        if not self.pre_def_dist:  # 如果不是预定义的扰动流量
            self.flow = [self.nom_flow]  # 扰动流量初始化为均值
