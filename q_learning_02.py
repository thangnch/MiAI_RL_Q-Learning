import random

import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

c_learning_rate = 0.1
c_discount_value = 0.9
c_no_of_eps = 10000
c_show_each = 1000

v_epsilon = 0.9
c_start_ep_epsilon_decay = 1
c_end_ep_epsilon_decay = c_no_of_eps // 2
v_epsilon_decay = v_epsilon / (c_end_ep_epsilon_decay - c_start_ep_epsilon_decay)



q_table_size = [20, 20]
q_table_segment_size = (env.observation_space.high - env.observation_space.low) / q_table_size


# Hàm chuyển đổi từ real state về q_state
def convert_state(real_state):
    q_state = (real_state - env.observation_space.low) // q_table_segment_size
    return tuple(q_state.astype(np.int))


q_table = np.random.uniform(low=-2, high=0, size=(q_table_size + [env.action_space.n]))

max_ep_reward = -999
max_ep_action_list = []
max_start_state = None

for ep in range(c_no_of_eps):
    print("Eps = ", ep)
    done = False
    current_state = convert_state(env.reset())
    ep_reward = 0
    ep_start_state = current_state
    action_list = []

    if ep % c_show_each == 0:
        show_now = True
    else:
        show_now = False

    while not done:
        if np.random.random() > v_epsilon:
            # Lấy argmax Q value của current_state
            action = np.argmax(q_table[current_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        action_list.append(action)

        # Hành động theo action đã lấy
        next_real_state, reward, done, _  = env.step(action=action)
        ep_reward += reward

        if show_now:
            env.render()

        if done:
            # Kiểm tra xem vị trí x có lớn hơn lá cờ không
            if next_real_state[0] >= env.goal_position:
                print("Đã đến cờ tại ep = {}, reward = {}".format(ep, ep_reward))
                if ep_reward > max_ep_reward:
                    max_ep_reward = ep_reward
                    max_ep_action_list = action_list
                    max_start_state  = ep_start_state

        else:
            # Convert về q_state
            next_state = convert_state(next_real_state)

            # Update Q value cho (current_state, action)
            current_q_value = q_table[current_state + (action,)]

            new_q_value = (1 - c_learning_rate) * current_q_value + c_learning_rate * (reward + c_discount_value * np.max(q_table[next_state]))

            q_table[current_state + (action,)] = new_q_value

            current_state = next_state
    if c_end_ep_epsilon_decay >= ep > c_start_ep_epsilon_decay:
        v_epsilon = v_epsilon - v_epsilon_decay

print("Max reward = ", max_ep_reward)
print("Max action list = ", max_ep_action_list)

env.reset()
env.state = max_start_state
for action in max_ep_action_list:
    env.step(action)
    env.render()

done = False
while not done:
    _, _, done,_ = env.step(0)
    env.render()



