# Add the ptdraft folder path to the sys.path list
from rewards import sum_rewards
from rewards import get_reward_2 as get_reward
import numpy as np
import matplotlib.pyplot as plt
from params import MAIN_PARAMS, AGENT_PARAMS, TANK_PARAMS, TANK_DIST
from models.Agent import Agent
from models.environment import Environment
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

plt.style.use("ggplot")  # 使用ggplot风格


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 只显示 warning 和 Error


def main():
    # ============= Initialize variables and objects ===========#
    max_mean_reward = MAIN_PARAMS["MAX_MEAN_REWARD"]  # 表示最大的平均奖励，用于判断是否停止训练
    environment = Environment(TANK_PARAMS, TANK_DIST, MAIN_PARAMS)  # 初始化环境
    agent = Agent(AGENT_PARAMS)  # 初始化智能体
    mean_episode = MAIN_PARAMS["MEAN_EPISODE"]  # 表示平均奖励的次数，用于计算平均奖励
    episodes = MAIN_PARAMS["EPISODES"]  # 表示训练的次数

    all_rewards = []  # 存储所有的奖励
    all_mean_rewards = []  # 存储所有的平均奖励
    t_mean = []  # 存储所有的平均时间

    # ================= Running episodes =================#
    try:
        for e in range(episodes):
            states, episode_reward = environment.reset()  # Reset level in tank
            for t in range(MAIN_PARAMS["MAX_TIME"]):  # 表示每次训练的最大时间，即每次训练的最大步数
                actions = agent.act(states[-1])  # 根据前一个状态空间，选择当前每个水箱对应阀门的动作
                # 根据当前动作选择对应阀门的状态（开度），z是一个列表，表示每个水箱对应阀门的状态，state[-1]是上一个状态空间，t是当前时间
                z = agent.get_z(actions)

                terminated, next_state = environment.get_next_state(
                    z, states[-1], t
                )  # 根据前一个状态空间中每个水箱的动作列表，上一个状态空间，当前时间t，得到下一个状态空间，以及是否终止的标志
                rewards = sum_rewards(
                    next_state, terminated, get_reward  # 当前状态空间下，每个水箱采取动作得到的reward
                )

                # Store data
                rewards = sum_rewards(
                    next_state, terminated, get_reward)  # 得到奖励
                # 将所有奖励相加，并添加到rewards中，此时rewards表示当前状态空间下，每个水箱的奖励以及所有奖励的和
                rewards.append(np.sum(rewards))
                episode_reward.append(rewards)  # 添加到当前迭代次数下记录rewards的列表中

                states.append(next_state)  # 将当前状态空间存储到states中
                agent.remember(states, rewards, terminated, t)  # 将数据存储到agent中

                if environment.show_rendering:
                    environment.render(z)  # 显示渲染
                if True in terminated:
                    break  # 如果终止了，就跳出循环

            episode_reward = np.array(episode_reward)
            episode_total_reward = []
            t_mean.append(t)
            for i in range(environment.n_tanks + 1):
                episode_total_reward.append(sum(episode_reward[:, i]))
            all_rewards.append(episode_total_reward)

            # Print mean reward and save better models
            if e % mean_episode == 0 and e != 0:
                mean_reward = np.array(all_rewards[-mean_episode:])
                mean_r = []
                t_mean = int(np.mean(t_mean))
                for i in range(environment.n_tanks + 1):
                    mean_r.append(np.mean(mean_reward[:, i]))
                all_mean_rewards.append(mean_r)
                print(
                    f"Mean {mean_episode} of {e}/{episodes} episodes ### timestep {t_mean+1} ### tot reward: {mean_r[-1]} ### ex1 {agent.epsilon[0]}"
                )
                t_mean = []
                if mean_r[-1] >= max_mean_reward:
                    agent.save_trained_model()
                    max_mean_reward = mean_r[-1]
                # Train model
            if agent.is_ready():
                agent.Qreplay(e)

            if not environment.running:
                break
            # if agent.epsilon <= agent.epsilon_min:
            #     break
    except KeyboardInterrupt:
        pass
    print("Memory length: {}".format(len(agent.memory)))
    print("##### {} EPISODES DONE #####".format(e + 1))
    print("Max rewards for all episodes: {}".format(np.max(all_rewards)))

    all_mean_rewards = np.array(all_mean_rewards)

    plt.plot(all_mean_rewards[:, -1], label="Total rewards")
    plt.ylabel("Mean rewards of last {} episodes".format(mean_episode))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("#### SIMULATION STARTED ####")
    print("  Max number of episodes: {}".format(
        MAIN_PARAMS["EPISODES"]))  # 表示训练的次数
    print("  Max time in each episode: {}".format(
        MAIN_PARAMS["MAX_TIME"]))  # 表示每次训练的最大时间
    print(
        "  {}Rendring simulation ".format(
            "" if MAIN_PARAMS["RENDER"] else "Not "
        )
    )
    main()
