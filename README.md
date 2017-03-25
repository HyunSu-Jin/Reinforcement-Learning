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
</code></pre>

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
![lab3-result](/lab3/result/lab3_result.png)

7. 통계
- 초기의 훈련단계에서는 Q-table이 완성되어지지 않은 훈련단계로써 성공률이 0에 수렴하지만, 한번 Q-table이 완성되고 나서는 Actor는 정해진 길로만 action-path를 수행하여 goal을 달성하므로 성공률이 1로 수렴한다.
![lab3-figure](/lab3/result/figure.png)

## Lab4
Q-learning(Table) ,Not dummy
- 기존의 dummy Q-leaning에서는 첫번째로 goal을 달성하게 되면 그때 정해진 path로 시종일관하여 Action path를 결정하게 되어 그 path외에는 다른 path는 탐색하지 않았다
- 이러한 특성때문에 결정된 Action path가 optimal하지 않은경우 algorithm의 효율성이 떨어지는 문제가 발생하였다.
- 이를 해결하기 위해서 Explicit vs. Exploration 을 E-greedy or random noise방법으로 결정하여 기존에 만든 path외에도 다른길을 탐색하여 최종적으로는 optimal poclicy(optimal path와 비슷)을 결정하도록 한다.
- 추가적으로, trainning과정 초기에는 자유도(e)가 상대적으로 커서 랜덤성이 강해 exploration 확률이 크지만, trainning이 일정부분 완성된 경우에는 자유도를 낮추어 exploration 확률을 적게한다. -- discounting 방법

### e-greedy 방법
<pre><code>
e = 1.0 / ((i//100) + 1)
if np.random.rand(1) < e:
			action = env.action_space.sample()
		else:
			action = np.argmax(Q[state,:])
</code></pre>
1. i값에따라서 discounting되는 e 를 결정한다.
2. random한 수를 뽑고 e와 비교해 Explicit vs. Exploration을 결정한다.
- 특징 : e - greedy는 random noise와는 다르게 랜덤성이 Q-table의 값과 전혀 무관하게 결정되어진다.
3. 실행결과
- 완전한 랜덤으로, random noise보다 성공률이 낮지만 optimal policy를 찾아내었다.
![lab4-e](/lab4/result/lab4_e.png)
### random noise 방법
<pre><code>
action = np.argmax(Q[state,:] + np.random.rand(1,env.action_space.n) / (i+1))
</code></pre>
1. 기존의 Q-table을 참조하여 action을 결정하는 algorithm에 일련의 noise값을 추가해 랜덤성을 부과한다.
2. 이 방법은 기존의 Q-table 값에 종속적이므로(dependent) 다음과 같은 실행결과를 갖는다.
3. 실행결과
- e-greedy보다 성공률이 높았다. 역시 마찬가지로 optimal policy를 보인다.
![lab4-n](/lab4/result/lab4_n.png)

## Lab5
Q-leaning at Non-deterministic(Stochastic)
- 기존의 실습환경은 Actor가 수행하는 Action에 따라서 그 다음의 State가 결정되어지는 환경이었다. 그러나, 이번에 확인하고자 하는 문제는 실세계와 같이 Actor가 Action을 취함에 따라서 결정되어지는 State가 일정하지않다.
이렇게 주어진 상황에서 기존의 deterministic 환경에서 정의했었던 Q-leaning 알고리즘을 사용한다면 어떠한 실행결과가 나올까?

### Non-deterministic 환경에 기존 알고리즘 적용
1. 실행결과
![lab5-1](/lab5/result/lab5_result1.png)
2. 통계

![lab5-2](/lab5/result/figure1.png)

실행결과가 상당히 성능이 떨어지는 것을 관찰할 수 있다.
1. 왜 그런것일까?
- 앞서 정의했었던 알고리즘은 다음과 같다.
<pre><code>
new_state, reward, done, _ = env.step(action)
Q[state,action] = reward + dis * np.max(Q[new_state,:])
</code></pre>
위 결과가 형편없는 이유는 Q-table이 말해주는 정보가 옳지 않은까닭이다. Q-table은 Actor가 '정한' action을 수행하고 이에 따른 new_state 의 정보를 Actor에게 전달하는 것이다. 그런데, new_state가 non-deterministic 함에 따라서 Q-table은 Actor가 RIGHT action을 수행하고 엉뚱한 state로 빠져 결과값을 얻은 것을 Actor에게 말해주는 꼴이 되버린것이다.
1. 이에 따라서 Q-table update 알고리즘을 다음과 같이 수정해야한다.
2. learing_rate 개념을 도입하여, next State에 대한 Q-table이 알려주는 정보를 완전히 신뢰하지 않고 weight를 주어 정보를 참고하고, 나머지 weight는 기존의 State에 대한 Q-table정보를 사용한다.
<pre><code>
Q[state,action] =  (1-learning_rate) * Q[state,action] + learning_rate *(reward + dis * np.max(Q[new_state,:]))
</code></pre>

3. 위와 같이 Q-update 알고리즘을 수정한뒤에 같은환경에서 실행.
4. 실행결과
![lab5-3](/lab5/result/lab5_result2.png)
5. 통계

![lab5-4](/lab5/result/figure2.png)

이전 알고리즘보다 성능이  개선된 것을 확인할 수 있다.

