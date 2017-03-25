import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')

# Initialize table with all zero
Q = np.zeros([env.observation_space.n,env.action_space.n])
dis = 0.99
num_episodes = 30000
rList = []
learning_rate = 0.2

for i in range(num_episodes):
	state = env.reset()
	rAll = 0
	done = False

	#e = 1.0 / ((i//100) + 1)
	while not done:
		action = np.argmax(Q[state,:] + np.random.rand(1,env.action_space.n) / (i+1))
		"""
		if np.random.rand(1) < e:
			action = env.action_space.sample()
		else:
			action = np.argmax(Q[state,:])
		"""
		new_state, reward, done, _ = env.step(action)
		#Q[state,action] = reward + dis * np.max(Q[new_state,:])
		Q[state,action] =  (1-learning_rate) * Q[state,action] + learning_rate *(reward + dis * np.max(Q[new_state,:]))
		rAll += reward
		state = new_state
	rList.append(rAll)

print("Success rate :" + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar( range(len(rList)), rList,color='blue')
plt.show()