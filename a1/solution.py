import sys
import os

from game_env import GameEnv
from game_state import GameState
import heapq
import time
import math

"""
solution.py

Template file for you to implement your solution to Assignment 1.

This file should include a 'main' method, allowing this file to be executed as a program from the command line.

Your program should accept 3 command line arguments:
    1) input filename
    2) output filename
    3) mode (either 'ucs' or 'a_star')

COMP3702 2021 Assignment 1 Support Code

Last updated by njc 04/08/21
"""

class StateNode():
    def __init__(self, game_env, gamestate, actions, path_cost, data=None):
        """
        Create a new node state map
        :param gamestate:
        :param game_env:
        :param actions:
        """
        self.game_env = game_env
        self.gamestate = gamestate
        self.actions = actions
        self.path_cost = path_cost
        self.data = data
        if data is not None:
            self.ladder_tiles, self.exit_tile, self.gem_dist_min, self.ladder_dist_min = self.data

    def heuristic_precomputation(self):
        self.ladder_tiles = []
        self.exit_tile = (self.game_env.exit_row, self.game_env.exit_col)
        for r in range(self.game_env.n_rows):
            for c in range(self.game_env.n_cols):
                if self.game_env.grid_data[r][c] == GameEnv.LADDER_TILE:
                    self.ladder_tiles.append((r,c))

        gem_dist = max(self.game_env.n_rows, self.game_env.n_cols)
        for gem in self.game_env.gem_positions:
            dist = manhattan_dist_heuristic(gem, self.exit_tile)
            if dist < gem_dist:
                gem_dist = dist
        self.gem_dist_min = gem_dist

        ladder_dist = max(self.game_env.n_rows, self.game_env.n_cols)
        for ladder in self.ladder_tiles:
            dist = manhattan_dist_heuristic(ladder, self.exit_tile)
            if dist < ladder_dist:
                ladder_dist = dist

        self.ladder_dist_min = ladder_dist

        self.data = (self.ladder_tiles, self.exit_tile, self.gem_dist_min, self.ladder_dist_min)

    def get_successors(self):
        successors = []
        for move in self.game_env.ACTIONS:
            success, new_state = self.game_env.perform_action(self.gamestate, move)
            if success:
                new_state_cost = self.game_env.ACTION_COST[move]
                next_state = StateNode(self.game_env, new_state, self.actions + [move], self.path_cost + new_state_cost, self.data)
                successors.append(next_state)
        return successors

    def get_heuristic(self):
        player_pos = (self.game_env.init_row, self.game_env.init_col)
        dist1 = manhattan_dist_heuristic(player_pos, self.exit_tile)

        player_gem_min_dist = max(self.game_env.n_rows, self.game_env.n_cols)
        for tile in self.game_env.gem_positions:
            d = manhattan_dist_heuristic(player_pos, tile)
            if player_gem_min_dist < d:
                player_gem_min_dist = d
        dist2 = player_gem_min_dist + self.gem_dist_min

        player_ladder_min_dist = max(self.game_env.n_rows, self.game_env.n_cols)
        for tile in self.ladder_tiles:
            d = manhattan_dist_heuristic(player_pos, tile)
            if player_ladder_min_dist < d:
                player_ladder_min_dist = d
        dist3 = player_ladder_min_dist + self.ladder_dist_min

        return min((dist1, dist2, dist3))

    def __eq__(self, other):
        return self.gamestate.row == other.gamestate.row and self.gamestate.col == other.gamestate.col \
                and self.gamestate.gem_status == other.gamestate.gem_status

    def is_goal(self, state):
        return self.game_env.is_solved(state)

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def __le__(self, other):
        return self.path_cost <= other.path_cost

    def __ge__(self, other):
        return self.path_cost >= other.path_cost

    def __gt__(self, other):
        return self.path_cost > other.path_cost

def log(verbose, visited, container, end_node, n_expanded):
    if not verbose:
        return
    print(f'Visited Nodes: {len(visited)},\t\tExpanded Nodes: {n_expanded},\t\t'f'Nodes in Container: {len(container)}')
    print(f'Cost of Path (with Costly Moves): {end_node.path_cost}')

def ucs(stateNode, verbose=True):
    container = [(0, stateNode)]
    heapq.heapify(container)
    # dict: state --> path_cost
    visited = {stateNode.gamestate: 0}
    n_expanded = 0
    while len(container) > 0:
        n_expanded += 1
        _, node = heapq.heappop(container)
        # check if this state is the goal
        if node.is_goal(node.gamestate):
            log(verbose, visited, container, node, n_expanded)
            return node.actions

        # add unvisited (or visited at higher path cost) successors to container
        successors = node.get_successors()
        for s in successors:
            if s.gamestate not in visited.keys() or s.path_cost < visited[s.gamestate]:
                visited[s.gamestate] = s.path_cost
                heapq.heappush(container, (s.path_cost, s))
    return None

def a_star(stateNode, verbose=True):
    container = []
    heapq.heappush(container, (len(stateNode.actions) + stateNode.get_heuristic(), stateNode)) #fringe
    # dict: state --> path_cost
    visited = {stateNode.gamestate: 0}
    n_expanded = 0
    while len(container) > 0:
        n_expanded += 1
        node = heapq.heappop(container)[1]
        # check if this state is the goal
        if node.is_goal(node.gamestate):
            log(verbose, visited, container, node, n_expanded)
            return node.actions

        # add unvisited (or visited at higher path cost) successors to container
        successors = node.get_successors()
        for s in successors:
            if s.gamestate not in visited.keys() or s.path_cost < visited[s.gamestate]:
                visited[s.gamestate] = s.path_cost
                heapq.heappush(container,
                               (len(s.actions) + s.get_heuristic(), s))

    return None

# def manhattan_dist_heuristic(src, dest):
#     # Game_env only
#     return abs(dest[0] - src[0]) + abs(dest[1] - src[1])

def manhattan_dist_heuristic(src, dest):
    # Euclidean distance
    return math.sqrt(math.pow((dest[1] - src[1]), 2) + math.pow((dest[0]-src[0]), 2))

def write_output_file(filename, actions):
    """
    Write a list of actions to an output file.
    :param filename: name of output file
    :param actions: list of actions where is action an element of GameEnv.ACTIONS
    """
    f = open(filename, 'w')
    for i in range(len(actions)):
        f.write(str(actions[i]))
        if i < len(actions) - 1:
            f.write(',')
    f.write('\n')
    f.close()


def main(arglist):
    if len(arglist) != 3:
        print("Running this file launches your solver.")
        print("Usage: play_game.py [input_filename] [output_filename] [mode = 'ucs' or 'a*']")

    print(arglist[0])
    print(arglist[1])
    input_file = arglist[0]
    output_file = arglist[1]
    mode = arglist[2]
    n_trials = 1
    assert os.path.isfile(input_file), '/!\\ input file does not exist /!\\'
    assert mode == 'ucs' or mode == 'a_star', '/!\\ invalid mode argument /!\\'

    actions = []
    if mode == 'ucs':
        # Read the input testcase file
        game_env = GameEnv(input_file)
        initial = StateNode(game_env, game_env.get_init_state(), actions, 0)
        print('UCS:')
        t0 = time.time()
        for i in range(n_trials):
            actions = ucs(initial)
        t_ucs = (time.time() - t0) / n_trials
        print(f'Num Actions: {len(actions)},\nActions: {actions}')
        print(f'Time: {t_ucs}')
        print('\n')

    if mode == 'a_star':
        print('A*:')
        t0 = time.time()
        game_env = GameEnv(input_file)
        initial = StateNode(game_env, game_env.get_init_state(), actions, 0)
        initial.heuristic_precomputation()
        for i in range(n_trials):
            actions = a_star(initial)
        t_a_star = (time.time() - t0) / n_trials
        print(f'Num Actions: {len(actions)},\nActions: {actions}')
        print(f'Time: {t_a_star}')
        print('\n')

    # Write the solution to the output file
    write_output_file(output_file, actions)


if __name__ == '__main__':
    main(sys.argv[1:])

