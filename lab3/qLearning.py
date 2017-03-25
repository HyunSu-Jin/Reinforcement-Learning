import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import  register
import random as pr

def rargmax(vector):
	m = np.amax(vector)
	indices = np.nonzero(vector == m)[0]
	return pr.choice(indices)

register(
	id = 'FrozenLake-v3',
	entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
	kwargs={
		'map_name' : '4x4',
		'is_slippery' :False
	}
)

env = gym.make('FrozenLake-v3')
# initialize Q Table
Q = np.zeros([env.observation_space.n,env.action_space.n])
# set trainning count
num_episodes = 2000

rList =[]

for i in range(num_episodes):
	state = env.reset() # initial state
	rAll = 0 # the total reward sum until unit  episode
	done = False

	while not done:
		action = rargmax(Q[state,:]) # Q is 2-dimension array, choose state and select action that returns maxtimun value
		new_state, reward, done, _ = env.step(action) # conduct action
		# update Q table
		rAll += reward
		Q[state,action] = reward + np.max(Q[new_state,:])
		state = new_state
	# end of one episode, add the total result, suc: 1, fail : 0
	rList.append(rAll)

# end of tranning

print("Success rate : ",str(sum(rList)/num_episodes))
print("Final Q table values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color='blue')
plt.show()
