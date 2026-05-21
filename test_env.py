from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai

print("Creating environment...")

env = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    ai2s=[microrts_ai.coacAI],
    max_steps=500,
)

print("Resetting environment...")

obs = env.reset()

print("Environment loaded successfully")
print("Observation type:", type(obs))
print("Observation shape:", obs.shape)

print("Action space:")
print(env.action_space)

print("Observation space:")
print(env.observation_space)

env.close()

print("Environment closed")