from simulator import Simulator
import numpy as np
from nTupleNetwork import LookupTable

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class ReinforcementLearning:
    def __init__(self):
        self.lookups = {}
        for i in range(4):
            self.lookups[i] = LookupTable()

    def epsilon_greedy(self, s):

        # check if this returns index
        if np.random.choice(2, 1, p=[0.01, 0.99])[0]:
            q_vals = [self.lookups[UP].func_approx(s),
                      self.lookups[RIGHT].func_approx(s),
                      self.lookups[DOWN].func_approx(s),
                      self.lookups[LEFT].func_approx(s)]
            maxi = max(q_vals)
            all_max = [index for index, v in enumerate(q_vals) if v == maxi]
            return np.random.choice(all_max)
        else:
            return np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]

    def get_action(self, dir):
        if dir == LEFT:
            return 'LEFT'
        if dir == RIGHT:
            return 'RIGHT'
        if dir == UP:
            return 'UP'
        return 'DOWN'


class QLearning(ReinforcementLearning):

    def run_episode(self, alpha):
        self.simulator = Simulator()
        s = self.simulator.board.matrix
        score = self.simulator.board.score
        while not self.simulator.game_over():
            # Evaluate action
            action = self.epsilon_greedy(self.simulator.board.matrix)
            # make move
            new_reward, s_prime, s_2_prime = self.simulator.play(self.get_action(action.flatten()[0]))
            # calc new reward
            r = new_reward - score
            # Learn
            self.learn(s, action, r, s_2_prime, alpha)
            score = new_reward
            s = s_2_prime
        return score

    def learn(self, state_matrix, action, reward, s_2_prime, alpha):
        v_max_action = self.epsilon_greedy(s_2_prime)
        v_max = self.lookups[v_max_action].func_approx(s_2_prime)
        diff = alpha * (reward + v_max - self.lookups[action].func_approx(state_matrix))
        self.lookups[action].update_mat_q(state_matrix, diff)
