MAIN_PARAMS = {
    "EPISODES": 20000, # 迭代次数
    "MEAN_EPISODE": 50, # 每50次迭代打印一次平均值
    "MAX_TIME": 200, # 每次迭代的最大时间
    "RENDER": True, # 是否显示动画
    "MAX_MEAN_REWARD": 150,  # minimum reward before saving model 表示最大的平均奖励
}

AGENT_PARAMS = {
    "N_TANKS": 1, # 水箱的数量
    "SS_POSITION": 0.5, # 阀门的稳态位置，即阀门的位置在这个位置的时候，水箱的水位是稳定的
    "VALVE_START_POSITION": 0.2, # 阀门的初始位置
    "ACTION_DELAY": [5], # 阀门的动作延迟，即每次动作之后，阀门的位置会在这个延迟之后才会改变
    "INIT_ACTION": 0, # 水箱的初始动作
    "VALVE_POSITIONS": 10, # 阀门的位置数量，即阀门的位置是一个离散的值，这个值表示阀门的位置有多少个
    "EPSILON": [1], # epsilon表示随机选择动作的概率，即每次选择动作的时候，会有epsilon的概率随机选择动作，1-epsilon的概率选择最优动作
    "EPSILON_MIN": [0.05], # epsilon的最小值，即epsilon会随着迭代次数的增加而减小，但是减小到这个值之后就不再减小
    "EPSILON_DECAY": [0.995], # epsilon的衰减率，即每次迭代epsilon都会乘以这个衰减率
    # Q(s,a) = (1 - learning_rate) * Q(s,a) +  learning_rate * (reward + gamma * max(Q(s',a')))
    "LEARNING_RATE": [0.001], # 水箱的学习，即每次更新Q表的时候，Q表的值会乘以这个学习率，学习率越大，表示越看重当前的样本，越不看重记忆中的样本
    "HIDDEN_LAYER_SIZE": [[5, 5]], # 水箱的神经网络的隐藏层的大小，即每个隐藏层有多少个神经元，这里有两个隐藏层，每个隐藏层有5个神经元
    "BATCH_SIZE": 5, # 水箱的批量大小，即每次更新Q表的时候，会从记忆中随机抽取这么多个样本，然后更新Q表
    "MEMORY_LENGTH": 10000, # 水箱的记忆长度，即记忆中最多可以存储多少个样本，当记忆中的样本数量超过这个值之后，会从记忆中随机删除一些样本
    "OBSERVATIONS": 4,  # level, gradient, is_above 0.5, prevous valve position 水箱的观测值，即每个时间步水箱的观测值分别为水位，水位的梯度，水位是否大于0.5，以及上一个时间步的阀门的位置
    "GAMMA": 0.9, # 水箱的折扣因子，折扣因子越大，越注重以往经验，越小，越重视眼前利益
    "SAVE_MODEL": [True], # 是否保存模型
    "LOAD_MODEL": [False], # 是否加载模型
    "TRAIN_MODEL": [True], # 是否训练模型
    "LOAD_MODEL_NAME": ["tank1_ql"], # 加载模型的名称
    "LOAD_MODEL_PATH": "Q_learning/Tank_1/", # 加载模型的路径
    "SAVE_MODEL_PATH": "Q_learning/Tank_1/", # 保存模型的路径
}

# Model parameters Tank 1
TANK1_PARAMS = {
    "height": 10, # 水箱的高度
    "init_level": 0.5, # 水箱的初始水位，这里表示水箱的初始水位为高度的一半
    "width": 10, # 水箱的宽度
    "pipe_radius": 0.5, # 水箱的管道半径
    "max_level": 0.75, # 水箱的最大水位
    "min_level": 0.25, # 水箱的最小水位
}

TANK1_DIST = { # 水箱的干扰输入
    "add": True, # 是否添加干扰输入
    "pre_def_dist": False, # 是否使用预定义的输入分布
    "nom_flow": 1,  # 2.7503 # 水箱的流量的均值
    "var_flow": 0.1, # 水箱的流量的方差
    "max_flow": 2, # 水箱的流量的最大值
    "min_flow": 0.7, # 水箱的流量的最小值
    "add_step": False, # 是否添加步长的干扰，即每次干扰的流量不是固定的，而是在一个范围内随机选择
    "step_time": int(MAIN_PARAMS["MAX_TIME"] / 2), # 水箱的分布的步长的时间，即在这个时间之前，流量是一个值，之后是另一个值
    "step_flow": 2, # 水箱的分布的步长的流量，即在step_time之前，流量是一个值，之后是另一个值
    "max_time": MAIN_PARAMS["MAX_TIME"], # 水箱的分布的最大时间
}

TANK_PARAMS = [TANK1_PARAMS] # 水箱的参数
TANK_DIST = [TANK1_DIST] # 水箱的输入，即流量的分布
