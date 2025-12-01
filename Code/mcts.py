from env_setup import create_minesweeper_env
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
import warnings

import copy
import numpy as np
import random

warnings.filterwarnings("ignore", category = UserWarning)

class Node:
    def __init__(self, env, board, parent = None):
        self.env = env
        self.board = board
        self.parent = parent
        
        self.children = {}
        self.rewards = 0
        self.visits = 0

    def __lt__(self, other):
        return self.rewards < other.rewards

    def __eq__(self, other):
        return self.rewards == other.rewards

    def __gt__(self, other):
        return self.rewards > other.rewards

def score(node, c):
    if node.visits == 0:
        return float('inf')

    return (node.rewards / node.visits) + c * np.sqrt(np.log(node.parent.visits) / node.visits)

def select(node):
    while node.children:
        children = node.children
        node = max(children.values(), key = lambda x: score(x, np.sqrt(2)))
    
    return node

def best_path(root):
    node = root
    while node.children:
       node = max(node.children.items(), key=lambda x: score(x[1], np.sqrt(2)))[1]

def expand(node):
    for action in get_actions(node.board):
        if action not in node.children:
            child_env = copy.deepcopy(node.env)
            board, reward, terminated, truncated, info = child_env.step(action)

            child = Node(env = child_env, board = board, parent = node)
            node.children[action] = child
            return child
    
    return random.choice(list(node.children.values()))

def simulate(node):
    env = copy.deepcopy(node.env)
    done = False
    total_reward = 0

    board = node.board
    while not done:
        random_action = random.choice(get_actions(board))
        board, reward, terminated, truncated, info = env.step(random_action)
        total_reward += reward
        done = terminated or truncated
    
    return total_reward

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.rewards += reward
        node = node.parent

def get_actions(board):
    def check_neighbors(board, i, j):
        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                if 0 <= x < board.shape[0] and 0 <= y < board.shape[1] and (x, y) != (i, j) and board[x][y] != -1:
                    return True
                    break
        return False

    possible_actions = []
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if board[i][j] == -1 and check_neighbors(board, i, j):
                possible_actions.append((i, j))
    
    return possible_actions

def run_mcts(env, n_iterations = 10):
    done = True
    while done:
        env.reset()
        initial_action = env.action_space.sample()
        board, reward, terminated, truncated, info = env.step(initial_action)
        done = terminated or truncated

    initial = Node(env, board)
    
    current_node = initial
    for i in range(n_iterations):
        expand(current_node)
        child = select(current_node)
        reward = simulate(child)
        backpropagate(child, reward)

    moves = [key for key in initial.children]
    for move in moves:
        env.step(move)
    env.render()
    # best_path(initial)


if __name__ == '__main__':
    env = create_minesweeper_env(8, 8, 10)

    run_mcts(env, 100)

    input("enter to exit")