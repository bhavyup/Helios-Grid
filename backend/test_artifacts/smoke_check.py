import numpy as np

from app.simulations.grid_env import GridEnv

env = GridEnv(
    weather_file=r"D:\codes\Helios-Grid\Helios-Grid\backend\test_artifacts\smoke_weather.csv",
    num_households=1,
    max_episode_steps=2,
)
env.reset()
_, _, _, info = env.step(
    {"house_actions": np.zeros((1, 6), dtype=np.float32), "market_actions": 0}
)
print(info["weather"]["utc_timestamp"])
