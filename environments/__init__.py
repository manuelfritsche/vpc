from gym.envs.registration import register

register(
    id='mazeSmall-v0',
    entry_point='environments.env:MazeEnvSmall',
    max_episode_steps=150,
)


register(
    id='mazeLarge-v0',
    entry_point='environments.env:MazeEnvLarge',
    max_episode_steps=400,
)
