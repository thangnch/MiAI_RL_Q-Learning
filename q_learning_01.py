import gym

# Tạo biến môi trường
env = gym.make("MountainCar-v0")
env.reset()

# Lấy state hiện tại sau khởi tạo
print(env.state)

# Lấy số action mà xe có thể thực hiện
print(env.action_space.n)

# Lấy X tối thiểu, tối đa và vận tốc tối thiểu, tối đa
print(env.observation_space.high)
print(env.observation_space.low)
#
# # Render thử
# while True:
#     action = 2 # Thử luôn di về bên phải
#     new_state, reward, done, _ = env.step(action)
#     print("New state = {}, reward = {}".format(new_state, reward))
#     env.render()