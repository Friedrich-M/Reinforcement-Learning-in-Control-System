# Reinforcement-Learning-in-Tank-System-Control

<img decoding="async" width="90%" alt="1" src="https://user-images.githubusercontent.com/85838942/208973370-bebf9f9d-3f37-4282-8113-93e3c675a3f4.png">

Reinforcement learning is a type of machine learning that involves training agents to take actions in an environment in order to maximize some type of reward. In the context of process control, reinforcement learning can be used to develop control strategies that can optimize a process and improve its performance.

One of the key benefits of reinforcement learning in process control is its ability to learn from experience, which allows it to adapt to changes in the environment and improve over time. This can be particularly useful in complex processes that are difficult to model accurately using traditional methods.
	
Reinforcement learning algorithms can be applied to a wide range of process control applications, including the control of industrial processes, transportation systems, and energy systems. In these applications, the reinforcement learning agent can learn to take actions that improve the performance of the process, such as reducing energy consumption or increasing the speed of production.

The project was to see if the RL method from machine learning could be of use as control model for industrial systems. Replacing traditional controllers like P-controller and MPC. So this is more of a POC to see if its viable to throw a RL algorithm on a industrial system which needs to be controlled.

The motivation is that some large complex industrial systems have have model sequations which need to be solved in order to have a control-model. And the solution is based on the system equations. Sometime the solution is hard to converge and solve. So this project was to figure out if one could give the system to the Machine, without any knowledge about the system and see if it could learn to control the systemstate given an disturbance to the system.

<img decoding="async" width="50%" alt="1" src="https://user-images.githubusercontent.com/85838942/208973472-b6278488-378d-4df5-9238-1bb35282a6a1.png">


In conclusion, this study explored the use of reinforcement learning, specifically Q learning, as controllers in the process industries as an alternative to traditional controllers. The performance of the reinforcement learning controllers was evaluated and compared to a traditional P-controller in two cases of tank level regulation. The results showed that the reinforcement learning controllers were able to control the liquid level within the predetermined constraints and the P-controllers also performed well with smaller input changes. However, the potential benefits of using reinforcement learning in the process industries, including its ability to handle nonlinearity and long-term evaluation, make it a promising area for further research and standardization efforts.
