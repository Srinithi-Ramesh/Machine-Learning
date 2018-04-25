from simulator import Simulator
import numpy as np
from nTupleNetwork import LookupTable

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class ReinforcementLearning:
    def __init__(self):
        self.lookup = LookupTable()
        self.move_score = 0
        self.shape = [3, 3]

    def get_action(self, dir):
        if dir == LEFT:
            return 'LEFT'
        if dir == RIGHT:
            return 'RIGHT'
        if dir == UP:
            return 'UP'
        return 'DOWN'


class TDAfterState(ReinforcementLearning):

    def __move_line(self, array):
        zeros_count = len(array[array == 0])
        new_array = np.flipud(array[array > 0])
        merge_array = []

        i = 0
        merge_zeros_count = 0

        while i != new_array.shape[0]:
            if i+1 == new_array.shape[0]:
                merge_array.append(new_array[i])
            elif new_array[i] == new_array[i+1]:
                merge_array.append(2 * new_array[i])
                i += 1
                self.move_score += 2 * new_array[i]
                merge_zeros_count += 1
            else:
                merge_array.append(new_array[i])
            i += 1

        merge_array = np.flipud(merge_array)
        zeros = (zeros_count + merge_zeros_count) * [0]
        zeros.extend(merge_array)
        return np.array(zeros)

    def after_state(self, original_matrix, direction):
        board = np.copy(original_matrix)
        self.move_score = 0
        if direction == UP:
            lines_cols = [np.flipud(board[:, i]) for i in range(self.shape[1])]
            for i, line in enumerate(lines_cols):
                board[:, i] = np.flipud(self.__move_line(line))
        elif direction == DOWN:
            lines_cols = [board[:, i] for i in range(self.shape[1])]
            for i, line in enumerate(lines_cols):
                board[:, i] = self.__move_line(line)
        elif direction == RIGHT:
            lines_rows = [board[i, :] for i in range(self.shape[0])]
            for i, line in enumerate(lines_rows):
                board[i, :] = self.__move_line(line)
        elif direction == LEFT:
            lines_rows = [np.flipud(board[i, :]) for i in range(self.shape[0])]
            for i, line in enumerate(lines_rows):
                board[i, :] = np.flipud(self.__move_line(line))
        else:
            raise ValueError("Unknown direction to move. Possible directions are 'up', 'down', 'left', 'right'")

        return board

    def run_episode(self, alpha):
        self.simulator = Simulator()
        s = self.simulator.board.matrix
        score = 0
        while not self.simulator.game_over():
            # Evaluate action
            action = self.get_next_action(s)
            # make move
            new_reward, s_prime, s_2_prime = self.simulator.play(self.get_action(action))
            # Learn
            self.learn(s_prime, s_2_prime, alpha)
            score = new_reward
            s = s_2_prime
        return score

    def learn(self, s_prime, s_2_prime, alpha):
        v_max_action = self.evaluate(s_2_prime)
        next_s = self.after_state(s_2_prime, v_max_action)
        next_r = self.move_score
        v_max = self.lookup.func_approx(next_s)
        diff = alpha * (next_r + v_max - self.lookup.func_approx(s_prime))
        self.lookup.update_mat_q(s_prime, diff)

    # evaluate for TD afterstate
    def evaluate(self, s):
        vals = np.zeros(4)
        for i in range(4):
            after_state_board = self.after_state(s, i)
            val = self.lookup.func_approx(after_state_board)
            vals[i] = val + self.move_score
        maxi = max(vals)
        all_max = [index for index, v in enumerate(vals) if v == maxi]
        return np.random.choice(all_max)

    def get_next_action(self, s):
        if np.random.choice(2, 1, p=[0.01, 1 - 0.01])[0]:
            self.evaluate(s)
        return np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
