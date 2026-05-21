from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
import numpy as np

env = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    ai2s=[microrts_ai.coacAI],
    max_steps=500,
)

obs = env.reset()

print("Obs shape:", obs.shape)

# sample random action from μRTS action space
action = np.array([env.action_space.sample()])

print("Sampled action:")
print(action)

next_obs, reward, done, info = env.step(action)

print("Step successful")
print("Reward:", reward)
print("Done:", done)

env.close()