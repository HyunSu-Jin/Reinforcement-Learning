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

## Lab3
Dummy Q-learning, Q-table
1. 주어진 Environment,환경 내에서 Actor가 최고의 reward,optimal value를 얻기위한 알고리즘기법
2. 각 Actor는 현재 State에 해당하는 Q-table 값을 참조하여 다음으로 행해질 Action을 결정하게 된다.
<pre><code>
action = rargmax(Q[state,:])
</pre><code>
즉, 현 state에 존재하는 Q-table, ex) [0,0,1,0] 을 보고 maximum값인 1을 나타내는 index 2,action을 리턴한다.
3. 이러한 방법으로 Actor는 현 State에 해당하는 Q-table을 참고하여 Action을 결정하며, 이는 Actor로 하여금 goal을 달성하게 할 optimal solution을 제공한다.(단, dummy Q-learning에는 허점이 존재)
4. 그렇다면, 이러한 Q-table을 만드는 과정이 요구되는데, 이 과정을 다음과 같은 알고리즘을 바탕으로 하여 강화학습, Reinforcement Learning 과정을 통해 Q-talbe을 만들어낸다.
- Q(s,a) = reward + max Q(s',a')
- 현 State에서의 Q(s,a) 값은 다음 state에서의 reward + 해당 state의 Q-table값중 가장 큰 것의 합.
<pre><code>
Q[state,action] = reward + np.max(Q[new_state,:])
</code></pre>
5. 이러한 과정으로 다음과같이 Q-table을 만든다.
<pre><code>
for i in range(num_episodes):
	state = env.reset() # initial state
	done = False

	while not done:
		action = rargmax(Q[state,:])
		new_state, reward, done, _ = env.step(action) # conduct action
		Q[state,action] = reward + np.max(Q[new_state,:])
		state = new_state
# end of tranning
</code></pre>

6. 실행결과
![lab3-result](/lab3/result/figure.png)

7. 통계
- 초기의 훈련단계에서는 Q-table이 완성되어지지 않은 훈련단계로써 성공률이 0에 수렴하지만, 한번 Q-table이 완성되고 나서는 Actor는 정해진 길로만 action-path를 수행하여 goal을 달성하므로 성공률이 1로 수렴한다.
![lab3-figure](/lab3/result/lab3_result.png)

