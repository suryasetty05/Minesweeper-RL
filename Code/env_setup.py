import gymnasium as gym
from gymnasium import spaces
import numpy as np
from random import randrange
import pygame
import pygame.freetype

# Minesweeper environment implemented in openai gym style

# State:
# Given the map of height x width, numbers in each block of map:
# -1 denotes unopened block
# 0-8 denotes the number of mines in the surrounding 8 blocks of the block

# Map:
# The map is unobervable to the agent where 0 is non-mine, 1 has a mine

# Actions:
# Action is Discrete(height * width) object
# Represents an attempt to open at the block's index when the map is flattened

# Step (Action): 
# Returns: next_state, reward, done, _

# Render: 
# Renders the current state using pygame

class MinesweeperEnv(gym.Env):
    def __init__(self, height = 16, width = 16, num_mines = 40):
        self.observation_space = spaces.Box(-1, 8, shape = (height, width), dtype = int)
        self.action_space = spaces.MultiDiscrete([height, width])

        self.height = height
        self.width = width
        self.num_mines = num_mines
        self.win_reward = 50
        self.fail_reward = -10
        self.map = np.array([[False] * width for _ in range(height)])
        self.state = np.zeros((height, width), dtype = int) - 1
        self.step_cntr = 0
        self.step_cntr_max = (height * width - num_mines)*2

        self.block_size = 25
        self.window_height = self.block_size * height
        self.window_width = self.block_size * width
        self.map = None        
        self.screen = None
        self.first_move = True

    def generate_mines(self, safe_x = None, safe_y = None):
        self.map = np.array([[False] * self.width for _ in range(self.height)])

        forbidden = set()
        if safe_x is not None:
            for i in range(max(0, safe_x - 1), min(self.height, safe_x + 2)):
                for j in range(max(0, safe_y - 1), min(self.width, safe_y + 2)):
                    forbidden.add((i, j))

        total_cells = self.height * self.width
        available_cells = total_cells - len(forbidden)

        if self.num_mines > available_cells:
            raise ValueError(f'Impossible mine placement: {self.num_mines} mines but only {available_cells} valid cells available.')
        
        placed = 0
        # Makes the first_move and its neighbors safe
        while placed < self.num_mines:
            x = randrange(self.height)
            y = randrange(self.width)
            if (x, y) in forbidden:
                continue

            if not self.map[x, y]:
                self.map[x, y] = True
                placed += 1

    def reset(self, seed = None, return_info = False, options = None):
        super().reset(seed = seed)

        info = {'steps': self.step_cntr, 'map': self.map}
        self.step_cntr = 0
        self.state = np.zeros((self.height, self.width),dtype=int)-1
        self.first_move = True
        self.map = None

        return self.state, info

    def get_num_opened(self):
        count = 0
        for i in self.state.flatten():
            if i >= 0: 
                count += 1
        return count

    def get_num_surr(self, x, y):
        count = 0
        for i in range(max(0, x - 1), min(self.height, x + 2)):
            for j in range(max(0, y - 1), min(self.width, y + 2)):
                if not (i == x and j == y):
                    if self.map[i,j]:
                        count += 1
        return count

    def update_state(self, x, y):
        num_surr = self.get_num_surr(x, y)
        self.state[x, y] = num_surr
        if num_surr == 0:
            for i in range(max(0, x - 1), min(self.height, x + 2)):
                for j in range(max(0, y - 1), min(self.width, y + 2)):
                    if (not (i == x and j == y)) and self.state[i, j] == -1:
                        self.update_state(i, j)

    def step(self, action):
        if len(action) != 2 or action[0] < 0 or action[1] >= self.height or action[1] < 0 or action[1] >= self.width:
            raise ValueError
        
        x, y = action[0], action[1]
        if self.first_move:
            self.generate_mines(safe_x = x, safe_y = y)
            self.first_move = False
            
        info = self._get_info()
        if self.step_cntr == self.step_cntr_max:
            return self.state, 0, False, True, info
        else:
            self.step_cntr += 1

        x,y = action[0],action[1]
        if self.map[x][y]:
            return self.state, self.fail_reward, True, False, info
        else:
            num_opened = self.get_num_opened()
            if self.state[x,y] != -1:
                return self.state, 0, False, False, info
            
            self.update_state(x, y)
            new_num_opened = self.get_num_opened()

            if new_num_opened == self.height * self.width - self.num_mines:
                return self.state, self.win_reward, True, False, info
            return self.state, new_num_opened - num_opened, False, False, info

    def drawGrid(self):
        for y in range(0, self.window_width, self.block_size):
            for x in range(0, self.window_height, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                num = int(self.state[int(x / self.block_size), int(y / self.block_size)])

                if num == -1:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)
                else:
                    color = (250, 250 - num * 30, 250 - num * 30)
                    pygame.draw.rect(self.screen, color, rect)
                    text = self.font.get_rect(str(num))
                    text.center = rect.center
                    self.font.render_to(self.screen, text.topleft, str(num), (0, 0, 0))

        pygame.display.update()

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            self.font = pygame.freetype.SysFont(pygame.font.get_default_font(), 13)

        self.screen.fill((0, 0, 0))
        self.drawGrid()

    def _get_info(self):
        return {'map': self.map}
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()