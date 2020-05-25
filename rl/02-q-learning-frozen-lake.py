import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    n = np.max(vector)
    indices = np.nonzero(vector == n)[0]
    return pr.choice(indices)

register(
    id = "FrozenLake-v3",
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make("FrozenLake-v3")
env.render()

Q = np.zeros([ env.observation_space.n, env.action_space.n ])
num_epoches = 2000

rList = []
for i in range(num_epoches):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = rargmax(Q[state, :])

        new_state, reward, done, _ = env.step(action)
        
        # Q-Learning 함수
        Q[state, action] = reward + np.max(Q[new_state, :])
        rAll += reward

        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_epoches))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rList)), rList, color="blue")
plt.show()