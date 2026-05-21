import numpy as np
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai

env = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    ai2s=[microrts_ai.coacAI],
    max_steps=500,
)

obs = env.reset()

print("Obs shape:", obs.shape)

mask = env.get_action_mask()

print("Mask shape:")
print(mask.shape)

print("First values:")
print(mask[0][:50])

env.close()