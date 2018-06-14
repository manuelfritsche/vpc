import numpy as np
import gym
from gym import spaces
import scr_maze

class MazeEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  
  def resize_state(self, observation):
      self.state = np.zeros([44, 44, 3], dtype=np.uint8)
      n, m = np.shape(observation)
      self.state[:n, :m, 0] = observation
      self.state[:n, :m, 1] = observation
      self.state[:n, :m, 2] = observation

  def __init__(self):
      self.game = scr_maze.make_game(self.game_id)
      self.game.its_showtime()
      observation, _, _ = self.game.play(4)
      self.resize_state(observation.board)
      n, m, _ = np.shape(self.state)
      self.action_space = spaces.Discrete(4)
      self.observation_space = spaces.Box(low=0, high=255, shape=(n, m, 3))
      
  def step(self, action):
      observation, reward, running = self.game.play(action)
      self.resize_state(observation.board)
      if not reward:
          reward = -0.001
      return self.state, reward, not running, {}

  def reset(self):
      self.game = scr_maze.make_game(self.game_id)
      self.game.its_showtime()
      observation, _, _ = self.game.play(4)
      self.resize_state(observation.board)
      #self.state = observation.board
      return self.state
    
class MazeEnvSmall(MazeEnv):
    def __init__(self):
        self.game_id = 0
        super(MazeEnvSmall, self).__init__()

class MazeEnvLarge(MazeEnv):
    def __init__(self):
        self.game_id = 1
        super(MazeEnvLarge, self).__init__()
