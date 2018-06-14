# Copyright 2017 the pycolab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A scrolling maze to explore. Collect all of the coins!

The scrolling mechanism used by this example is a bit old-fashioned. For a
recommended simpler, more modern approach to scrolling in games with finite
worlds, have a look at `better_scrolly_maze.py`. On the other hand, if you have
a game with an "infinite" map (for example, a maze that generates itself "on
the fly" as the agent walks through it), then a mechanism using the scrolling
protocol (as the game entities in this game do) is worth investigating.

Command-line usage: `scrolly_maze.py <level>`, where `<level>` is an optional
integer argument selecting Scrolly Maze levels 0, 1, or 2.

Keys: up, down, left, right - move. q - quit.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

from pycolab import ascii_art
import environments.drapes as prefab_drapes
from pycolab.prefab_parts import sprites as prefab_sprites


# pylint: disable=line-too-long
MAZES_ART = [
    # Each maze in MAZES_ART must have exactly one of the patroller sprites
    # 'a', 'b', and 'c'. I guess if you really don't want them in your maze, you
    # can always put them down in an unreachable part of the map or something.
    #
    # Make sure that the Player will have no way to "escape" the maze.
    #
    # Legend:
    #     '#': impassable walls.            'a': patroller A.
    #     '@': collectable coins.           'b': patroller B.
    #     'P': player starting location.    'c': patroller C.
    #     ' ': boring old maze floor.       '+': initial board top-left corner.
    #     't', 'z', 'u', 'i', 'o': little less boring maze floor
    #
    # Don't forget to specify the initial board scrolling position with '+', and
    # take care that it won't cause the board to extend beyond the maze.
    # Remember also to update the MAZES_WHAT_LIES_BENEATH array whenever you add
    # a new maze.
    
    # Maze #150 #30 steps optmimally
    [ '#########################################################################################',
      '#####################################################################################abc#',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '######################################zzzzzzzzzzzzz#####zzzzzzzzzzzzzzzzzzzzzzz##########',
      '######################################zzzzzzzzzzzzz# . #zzzzzzzzzzzzzzzzzzzzzzz##########',
      '######################################zzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzz##########',
      '######################################zzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzz##########',
      '######################################zzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzz##########',
      '######################################zzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzz##########',
      '######################################ttttttttttttt#   #ttttttttttttttttttttttt##########',
      '######################################ttttttttttttt#   #########ttttttttttttttt##########',
      '######################################ttttttttttttt#           #ttttttttttttttt##########',
      '######################################ttttttttttttt#           #ttttttttttttttt##########',
      '######################################ttttttttttttt#           #ttttttttttttttt##########',
      '######################################ttttttttttttt#########   #ttttttttttttttt##########',
      '######################################zzzzzzzzzzzzzzzzzzzzz#   #zzzzzzzzzzzzzzz##########',
      '######################################zzzzzzzzzzzzzzzzzzzzz#   #zzzzzzzzzzzzzzz##########',
      '######################################zzzzzzzzzzzzzzzzzzzzz#   #zzzzzzzzzzzzzzz##########',
      '######################################zzzzzzzzzzzzzzzzzzzzz#   #zzzzzzzzzzzzzzz##########',
      '######################################zzzzzzzzzzzzzzzzzzzzz#   #zzzzzzzzzzzzzzz##########',
      '######################################zzzzzzzzzzzzzzzzzzzzz#   #zzzzzzzzzzzzzzz##########',
      '######################################ttttttttttttttttttttt#   #ttttttttttttttt##########',
      '######################################tttttttttt+tttttttttt#   #ttttttttttttttt##########',
      '######################################ttttttttttttttttttttt#   #ttttttttttttttt##########',
      '######################################ttttttttttttttttttttt#   #ttttttttttttttt##########',
      '######################################ttttttttttttttttttttt#   #ttttttttttttttt##########',
      '######################################ttttttttttttttttttttt# P #ttttttttttttttt##########',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################'],
    
    
    
    # Maze #400 #50 steps optimally
    [ '#########################################################################################',
      '#######uuuuuuuuuuuuuuuuuuuuuuuuiiiiiiiiiiiiiiiiiiiiiiiiii############################abc#',
      '#######uuuuuuuuuuuuuuuuuuuuuuuuiiiiiiiiiiiiiiiiiiiiiiiiii################################',
      '#######uuuuuuuuuuuuuuuuuuuuuuuuiiiiiiiiiiiiiiiiiiiiiiiiii################################',
      '#######uuuuuuuuuuuuu#####################iiiiiiiiiiiiiiii################################',
      '#######uuuuuuuuuuuuu#                   #iiiiiiiiiiiiiiii################################',
      '#######uuuuuuuuuuuuu#                  .#iiiiiiiiiiiiiiii################################',
      '#######uuuuuuuuuuuuu#                   #iiiiiiiiiiiiiiii################################',
      '#######uuuuuuuuuuuuu#   #################iiiiiiiiiiiiiiii################################',
      '#######zzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz################################',
      '#######zzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz################################',
      '#######zzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz################################',
      '#######zzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz################################',
      '#######zzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz################################',
      '#######zzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz################################',
      '#######ttttttttttttt#   #tttttttttttttttttttttttttttttttt################################',
      '#######ttttttttttttt#   #tttttttttttttttttttttttttttttttt################################',
      '#######ttttttttttttt#   #########tttttttttttttttttttttttt################################',
      '#######ttttttttttttt#           #tttttttttttttttttttttttt################################',
      '#######ttttttttttttt#           #tttttttttttttttttttttttt################################',
      '#######ttttttttttttt#           #tttttttttttttttttttttttt################################',
      '#######ttttttttttttt#########   #tttttttttttttttttttttttt################################',
      '#######zzzzzzzzzzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzzz################################',
      '#######zzzzzzzzzzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzzz################################',
      '#######zzzzzzzzzzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzzz################################',
      '#######zzzzzzzzzzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzzz################################',
      '#######zzzzzzzzzzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzzz################################',
      '#######zzzzzzzzzzzzzzzzzzzzz#   #zzzzzzzzzzzzzzzzzzzzzzzz################################',
      '#######ttttttttttttttttttttt#   #tttttttttttttttttttttttt################################',
      '#######tttttttttt+tttttttttt#   #tttttttttttttttttttttttt################################',
      '#######ttttttttttttttttttttt#   #tttttttttttttttttttttttt################################',
      '#######ttttttttttttttttttttt#   #tttttttttttttttttttttttt################################',
      '#######ttttttttttttttttttttt#   #tttttttttttttttttttttttt################################',
      '#######ttttttttttttttttttttt# P #tttttttttttttttttttttttt################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################',
      '#########################################################################################'],

]
# pylint: enable=line-too-long

# -----------------------------------------------------------------------------
# insert random reward position to maze level   
def set_random_reward(level):
    # first build a matrix with 1 at every position with a '.'
    maze = np.array(MAZES_ART[level]).copy()
    n = np.size(maze)
    m = np.size(list(maze[0]))
    elements = np.zeros([n,m], dtype=np.uint8)
    for i in range(n):
        elements[i, :] = np.array(list(maze[i])).view(np.uint8)
    
    elements = (elements == ord('.')) * 1
    
    # next select a random position with a 1
    row, col = np.nonzero(elements)
    choice_row = np.random.choice(row)
    choice_col = np.random.choice(col)
    
    # finally set the corresponding point to a '@' and replace others with ' '
    row = list(maze[choice_row])
    row[choice_col] = '@'
    new_row = ''.join(row)
    maze[choice_row] = new_row
    replace_dots(maze)
    return maze

def replace_dots(maze):
    n = np.size(maze)
    m = np.size(list(maze[0]))
    for i in range(n):
        row = list(maze[i])
        for j in range(m):
            if row[j] == '.':
                row[j] = ' '
        maze[i] = ''.join(row)
    return maze

# -----------------------------------END---------------------------------------


MAZES_WHAT_LIES_BENEATH = 't'
    # What lies below '+' characters in MAZES_ART?
    # Unlike the what_lies_beneath argument to ascii_art_to_game, only single
    # characters are supported here for the time being.


STAR_ART = ['  .           .          .    ',
            '         .       .        .   ',
            '        .          .         .',
            '  .    .    .           .     ',
            '.           .          .   . .',
            '         .         .         .',
            '   .                 .        ',
            '           . .          .     ',
            '    .            .          . ',
            '  .      .              .  .  ']


# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {' ': (0, 0, 0),        # Inky blackness of SPAAAACE
             '.': (949, 929, 999),  # These stars are full of lithium
             '@': (999, 862, 110),  # Shimmering golden coins
             '#': (764, 0, 999),    # Walls of the SPACE MAZE
             't': (500, 500, 5),    # Walls of the SPACE MAZE
             'z': (400, 200, 200),    # Walls of the SPACE MAZE
             'u': (0, 500, 500),    # Walls of the SPACE MAZE
             'i': (600, 250, 250),    # Walls of the SPACE MAZE
             'o': (0, 50, 900),    # Walls of the SPACE MAZE
             'P': (0, 999, 999),    # This is you, the player
             'a': (999, 0, 780),    # Patroller A
             'b': (145, 987, 341),  # Patroller B
             'c': (987, 623, 145)}  # Patroller C

COLOUR_BG = {'.': (0, 0, 0),        # Around the stars, inky blackness etc.
             '@': (0, 0, 0)}


def make_game(level):
  """Builds and returns a Scrolly Maze game for the selected level."""
  # A helper object that helps us with Scrolly-related setup paperwork.
  maze = set_random_reward(level)
  scrolly_info = prefab_drapes.Scrolly.PatternInfo(
      maze, STAR_ART,
      board_northwest_corner_mark='+',
      what_lies_beneath=MAZES_WHAT_LIES_BENEATH)
  # Changed scroll_margins=(2, 3) to scroll_margins=None in Scrolly constructor

  walls_kwargs = scrolly_info.kwargs('#')
  t_kwargs = scrolly_info.kwargs('t')
  z_kwargs = scrolly_info.kwargs('z')
  u_kwargs = scrolly_info.kwargs('u')
  i_kwargs = scrolly_info.kwargs('i')
  o_kwargs = scrolly_info.kwargs('o')
  coins_kwargs = scrolly_info.kwargs('@')
  player_position = scrolly_info.virtual_position('P')
  patroller_a_position = scrolly_info.virtual_position('a')
  patroller_b_position = scrolly_info.virtual_position('b')
  patroller_c_position = scrolly_info.virtual_position('c')

  return ascii_art.ascii_art_to_game(
      STAR_ART, what_lies_beneath=' ',
      sprites={
          'P': ascii_art.Partial(PlayerSprite, player_position),
          'a': ascii_art.Partial(PatrollerSprite, patroller_a_position),
          'b': ascii_art.Partial(PatrollerSprite, patroller_b_position),
          'c': ascii_art.Partial(PatrollerSprite, patroller_c_position)},
      drapes={
          '#': ascii_art.Partial(MazeDrape, **walls_kwargs),
          't': ascii_art.Partial(MazeDrape, **t_kwargs),
          'z': ascii_art.Partial(MazeDrape, **z_kwargs),
          'u': ascii_art.Partial(MazeDrape, **u_kwargs),
          'i': ascii_art.Partial(MazeDrape, **i_kwargs),
          'o': ascii_art.Partial(MazeDrape, **o_kwargs),
          '@': ascii_art.Partial(CashDrape, **coins_kwargs)},
      # The base Backdrop class will do for a backdrop that just sits there.
      # In accordance with best practices, the one egocentric MazeWalker (the
      # player) is in a separate and later update group from all of the
      # pycolab entities that control non-traversable characters.
      update_schedule=[['#'], ['t', 'z', 'u', 'i', 'o', 'a', 'b', 'c', 'P'], ['@']],
      z_order='abc@#tzuioP')


class PlayerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player, the maze explorer.

  This egocentric `Sprite` requires no logic beyond tying actions to
  `MazeWalker` motion action helper methods, which keep the player from walking
  on top of obstacles.
  """

  def __init__(self, corner, position, character, virtual_position):
    """Constructor: player is egocentric and can't walk through walls."""
    super(PlayerSprite, self).__init__(
        corner, position, character, egocentric_scroller=True, impassable='#')
    self._teleport(virtual_position)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, things, layers  # Unused

    if actions == 0:    # go upward?
      self._north(board, the_plot)
    elif actions == 1:  # go downward?
      self._south(board, the_plot)
    elif actions == 2:  # go leftward?
      self._west(board, the_plot)
    elif actions == 3:  # go rightward?
      self._east(board, the_plot)
    elif actions == 4:  # do nothing?
      self._stay(board, the_plot)


class PatrollerSprite(prefab_sprites.MazeWalker):
  """Wanders back and forth horizontally, killing the player on contact."""

  def __init__(self, corner, position, character, virtual_position):
    """Constructor: changes virtual position to match the argument."""
    super(PatrollerSprite, self).__init__(corner, position, character, '#')
    self._teleport(virtual_position)
    # Choose our initial direction based on our character value.
    self._moving_east = bool(ord(character) % 2)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, layers, backdrop  # Unused.

    # We only move once every two game iterations.
    if the_plot.frame % 2:
      self._stay(board, the_plot)
      return

    # MazeWalker would make certain that we don't run into a wall, but only
    # if the sprite and the wall are visible on the game board. So, we have to
    # look after this ourselves in the general case.
    pattern_row, pattern_col = things['#'].pattern_position_prescroll(
        self.virtual_position, the_plot)
    next_to_wall = things['#'].whole_pattern[
        pattern_row, pattern_col+(1 if self._moving_east else -1)]
    if next_to_wall: self._moving_east = not self._moving_east

    # Make our move. If we're now in the same cell as the player, it's instant
    # game over!
    (self._east if self._moving_east else self._west)(board, the_plot)
    if self.virtual_position == things['P'].virtual_position:
      the_plot.terminate_episode()


class MazeDrape(prefab_drapes.Scrolly):
  """A scrolling `Drape` handling the maze scenery.

  This `Drape` requires no logic beyond tying actions to `Scrolly` motion
  action helper methods. Our job as programmers is to make certain that the
  actions we use have the same meaning between all `Sprite`s and `Drape`s in
  the same scrolling group (see `protocols/scrolling.py`).
  """

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, things, layers  # Unused

    if actions == 0:    # is the player going upward?
      self._north(the_plot)
    elif actions == 1:  # is the player going downward?
      self._south(the_plot)
    elif actions == 2:  # is the player going leftward?
      self._west(the_plot)
    elif actions == 3:  # is the player going rightward?
      self._east(the_plot)
    elif actions == 4:  # is the player doing nothing?
      self._stay(the_plot)


class CashDrape(prefab_drapes.Scrolly):
  """A scrolling `Drape` handling all of the coins.

  This `Drape` ties actions to `Scrolly` motion action helper methods, and once
  again we take care to map the same actions to the same methods. A little
  extra logic updates the scrolling pattern for when the player touches the
  coin, credits reward, and handles game termination.
  """

  def update(self, actions, board, layers, backdrop, things, the_plot):
    # If the player has reached a coin, credit one reward and remove the coin
    # from the scrolling pattern. If the player has obtained all coins, quit!
    player_pattern_position = self.pattern_position_prescroll(
        things['P'].position, the_plot)

    if self.whole_pattern[player_pattern_position]:
      the_plot.log('Coin collected at {}!'.format(player_pattern_position))
      the_plot.add_reward(1.0)
      self.whole_pattern[player_pattern_position] = False
      if not self.whole_pattern.any(): the_plot.terminate_episode()

    if actions == 0:    # is the player going upward?
      self._north(the_plot)
    elif actions == 1:  # is the player going downward?
      self._south(the_plot)
    elif actions == 2:  # is the player going leftward?
      self._west(the_plot)
    elif actions == 3:  # is the player going rightward?
      self._east(the_plot)
    elif actions == 4:  # is the player doing nothing?
      self._stay(the_plot)
    elif actions == 5:  # does the player want to quit?
      the_plot.terminate_episode()

