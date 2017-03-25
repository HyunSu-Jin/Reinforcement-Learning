# Reinforcement-Learning

## Lab2
Playing OPenAI GYM Games
1. Actor
2. Actor가 행하는 Action ex) LEFT,RIGHT,UP,DOWN
3. Environment
4. State, Observation, done, info

<pre><code>
import gym

env = gym.make("FrozenLake-v0")
observation = env.reset()
for _ in range(1000):
	env.render()
	action = env.action_space.sample()
	observation, reward, done, info = env.step(action)
</code></pre>

- Reinforcement Learning은 크게 Actor와 Env로 나눌 수 있으며 Actor가 행하는 각 ACTION에 따라서, Env는 그에따른 Observation정보를 리턴한다.
- Actor는 이러한 과정을 반복적으로 수행하여 좀 더 효율적으로 Goal을 달성하는 방식을 학습하며 Goal을 달성하는 최적의 path를 찾는다. 이러한 일련의 과정을 Reinforcement Learning 이라 한다.
