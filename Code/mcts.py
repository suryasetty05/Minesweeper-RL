from env_setup import create_minesweeper_env
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
import warnings

import copy
import numpy as np
import random

warnings.filterwarnings("ignore", category = UserWarning)

class Node:
    def __init__(self, env, board, parent = None, action = None):
        self.env = env
        self.board = board
        self.parent = parent
        self.action = action
        
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
        not_visited = [c for c in node.children.values() if c.visits == 0]
        if not_visited:
            return node
        children = node.children
        node = max(children.values(), key = lambda x: score(x, np.sqrt(2)))
    
    return node

def expand(node):
    for action in get_actions(node.board):
        if action not in node.children:
            child_env = copy.deepcopy(node.env)
            board, reward, terminated, truncated, info = child_env.step(action)

            child = Node(env = child_env, board = board, parent = node, action = action)
            node.children[action] = child
    
    return random.choice(list(node.children.values()))

def simulate_random(node):
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

def simulate_logic(node):
    def get_score(board, i, j):
        m = 0
        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                if 0 <= x < board.shape[0] and 0 <= y < board.shape[1] and (x, y) != (i, j) and board[x][y] == -1:
                    m += 1
        return m

    def calculate_prob(cell_values, i, j):
        p = 1
        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                if (x, y) in cell_values:
                    p = p * (1 - cell_values[(x, y)])
        return 1 - p

    def single_step(board):
        cell_values = {}

        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if cell > 0:
                    cell_values[(i, j)] = cell / get_score(board, i, j)

        action_list = []
        for action in get_actions(board):
            action_list.append((action, calculate_prob(cell_values, action[0], action[1])))

        action_list = sorted(action_list, key = lambda x: x[1])
        return action_list[0]

    env = copy.deepcopy(node.env)
    done = False
    total_reward = 0

    board = node.board
    while not done:
        action = single_step(board)[0]
        board, reward, terminated, truncated, info = env.step(action)
        if reward == -10:
            total_reward += -40
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
    
    for i in range(n_iterations):
        path = select(initial)
        child = expand(path)
        reward = simulate_logic(child)
        backpropagate(child, reward)

    test_node = initial
    best_moves = [initial_action]

    while test_node.children:
        next_node = max(test_node.children.items(), key=lambda x: x[1].visits)
        best_moves.append(next_node[0])
        test_node = next_node[1]

    for move in best_moves:
        board, reward, terminated, truncated, info = env.step(move)
        if reward == -10:
            return (False, best_moves)
        
    return (True, best_moves)

if __name__ == '__main__':
    env = create_minesweeper_env(10, 10, 8)
    run = run_mcts(env, 10000)

    for move in run[1]:
        board, reward, terminated, truncated, info = env.step(move)
        print(f'{move}: {reward}')

    env.render()

    input("enter to exit")

# if __name__ == '__main__':
#     success = (False, [])
#     tries = 0

#     while not success[0]:
#         tries = tries + 1
#         env = create_minesweeper_env(5, 5, 5)
#         success = run_mcts(env, 1000)
        
#     print(f'Tries: {tries}')
#     for move in success[1]:
#         board, reward, terminated, truncated, info = env.step(move)
#         print(f'{move}: {reward}')
    
#     env.render()

#     input("enter to exit")