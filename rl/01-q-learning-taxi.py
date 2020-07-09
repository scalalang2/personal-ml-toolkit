import numpy as np
import random
import time
import gym

env = gym.make("Taxi-v3")
env.render()

action_size = env.action_space.n
state_size = env.observation_space.n
print("Action size {} , State size {}".format(action_size, state_size))
print("Action space: {}, State space: {}".format(env.action_space, env.observation_space))

qtable = np.zeros((state_size, action_size))
print("\nQ-Table: ")
print(qtable)

# 하이퍼 파라미터 정의
episodes = 30000
max_steps = 1000
lr = 0.3
decay_fac = 1E-5
gamma = 0.90

for episode in range(episodes):
    state = env.reset()
    done = False
    lr -= decay_fac
    step = 0

    if lr <= 0:
        break

    for step in range(max_steps):
        # 랜덤하게 액션을 선태한다.
        action = env.action_space.sample()

        # Action을 취한다.
        new_state, reward, done, info = env.step(action)

        if done == True:
            # Exploration
            # 너무 빨리 Qtable이 수렴하는 것을 방지한다.
            if (step < 199 | step > 201) :
                qtable[state, action] = qtable[state, action] + lr * (reward+gamma * 0 - qtable[state, action])
            break
        else:
            # 미래의 리워드 총합
            qtable[state, action] = qtable[state, action] + lr * (reward+gamma * np.max(qtable[new_state, :]) - qtable[state, action])
    
        if done == True:
            break
    
        state = new_state
    

    if (episode + 1) % 3000 == 0 :
        print("episode: [{}/{}] ".format(episode + 1, episodes))
    
print("Learning episode is done.\n")
state = env.reset()
env.render()
done = False
total_reward = 0

while done == False:
    # 리워드가 가장 큰 Action을 선택한다.
    # Policy Function
    action = np.argmax(qtable[state, :])
    state, reward, done, info = env.step(action)

    total_reward += reward
    time.sleep(0.5)
    env.render()
    print('Episode Reward: {}'.format(total_reward))
