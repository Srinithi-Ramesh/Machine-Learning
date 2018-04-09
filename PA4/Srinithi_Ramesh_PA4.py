from multiprocessing.pool import Pool

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv():
    """
    You are an agent on an s x s grid and your goal is to reach the terminal
    state at the top right corner.
    For example, a 4x4 grid looks as follows:
    o  o  o  T
    o  o  o  o
    o  o  o  o
    x  o  o  o

    x is your position and T is the terminal state.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -0.1 at each step until you reach a terminal state.
    """

    def __init__(self, shape=[4, 4]):
        self.shape = shape

        nS = np.prod(shape)  # The area of the gridworld
        MAX_Y = shape[0]
        MAX_X = shape[1]
        nA = 4  # There are four possible actions
        self.P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex  # s is the current position id. s = y * 4 + x
            y, x = it.multi_index

            self.P[s] = {a: [] for a in range(nA)}

            is_done = lambda s: s == shape[1] - 1
            reward = 5.0 if is_done(s) else -0.1

            # We're stuck in a terminal state
            if is_done(s):
                self.P[s][UP] = [(s, reward, True)]
                self.P[s][RIGHT] = [(s, reward, True)]
                self.P[s][DOWN] = [(s, reward, True)]
                self.P[s][LEFT] = [(s, reward, True)]
            # Not a terminal state, and if the agent ’bump into the wall’, it will stay in the same state
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                if is_done(ns_up):
                    reward = 5.0
                else:
                    reward = -0.1
                self.P[s][UP] = [(ns_up, reward, is_done(ns_up))]
                if is_done(ns_right):
                    reward = 5.0
                else:
                    reward = -0.1
                self.P[s][RIGHT] = [(ns_right, reward, is_done(ns_right))]
                if is_done(ns_down):
                    reward = 5.0
                else:
                    reward = -0.1
                self.P[s][DOWN] = [(ns_down, reward, is_done(ns_down))]
                if is_done(ns_left):
                    reward = 5.0
                else:
                    reward = -0.1
                self.P[s][LEFT] = [(ns_left, reward, is_done(ns_left))]

            it.iternext()

    # The possible action has a 0.8 probability of succeeding
    def action_success(self, success_rate=0.8):
        return np.random.choice(2, 1, p=[1 - success_rate, success_rate])[0]

    # If the action fails, any action is chosen uniformly(including the succeeding action)
    def get_action(self, action):
        if self.action_success():
            return action
        else:
            random_action = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
            return random_action

    # Given the current position, this function outputs the position after the action.
    def move(self, s, action):
        return self.P[s][action]


class ReinforcementLearning:
    def __init__(self, grid_world_size):
        self.g = GridworldEnv([grid_world_size, grid_world_size])
        self.shape = grid_world_size
        self.reset_Q()

    def reset_Q(self):
        self.Q = {}
        nS = np.prod([self.shape, self.shape])  # The area of the gridworld
        grid = np.arange(nS).reshape([self.shape, self.shape])
        it = np.nditer(grid, flags=['multi_index'])
        nA = 4
        while not it.finished:
            s = it.iterindex  # s is the current position id. s = y * 4 + x
            self.Q[s] = {a: [] for a in range(nA)}
            self.Q[s][UP] = np.random.rand()
            self.Q[s][RIGHT] = np.random.rand()
            self.Q[s][DOWN] = np.random.rand()
            self.Q[s][LEFT] = np.random.rand()
            it.iternext()
        self.Q[self.shape - 1][UP] = 0
        self.Q[self.shape - 1][RIGHT] = 0
        self.Q[self.shape - 1][DOWN] = 0
        self.Q[self.shape - 1][LEFT] = 0

    def epsilon_greedy_action(self, state, epsilon):
        if np.random.choice(2, 1, p=[epsilon, 1 - epsilon])[0]:
            # check if this returns index
            maxi = max(self.Q[state].values())
            all_max = [k for k, v in self.Q[state].items() if v == maxi]
            return np.random.choice(all_max)
        return np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]


class QLearning(ReinforcementLearning):

    def run_episode(self, gamma, epsilon, alpha, l=None):
        state = self.shape * (self.shape - 1)
        is_done = False
        step = 0
        while not is_done:
            action = self.g.get_action(self.epsilon_greedy_action(state, epsilon))
            next_state, curr_rewards, is_done = self.g.move(state, action)[0]
            self.Q[state][action] = self.Q[state][action] + \
                                    alpha * (curr_rewards +
                                             gamma * self.Q[next_state][
                                                 max(self.Q[next_state], key=self.Q[next_state].get)] -
                                             self.Q[state][action])
            state = next_state
            step += 1
        return step


class Sarsa(ReinforcementLearning):

    def __init__(self,  grid_world_size):
        super().__init__(grid_world_size)
        self.reset_e()

    def reset_e(self):
        self.e = {}
        nS = np.prod([self.shape, self.shape])  # The area of the gridworld
        grid = np.arange(nS).reshape([self.shape, self.shape])
        it = np.nditer(grid, flags=['multi_index'])
        nA = 4
        while not it.finished:
            s = it.iterindex  # s is the current position id. s = y * 4 + x
            self.e[s] = {a: [] for a in range(nA)}
            self.e[s][UP] = 0
            self.e[s][RIGHT] = 0
            self.e[s][DOWN] = 0
            self.e[s][LEFT] = 0
            it.iternext()

    def run_episode(self, gamma, epsilon, alpha, l):
        state = self.shape * (self.shape - 1)
        is_done = False
        step = 0
        action = self.g.get_action(self.epsilon_greedy_action(state, epsilon))
        while not is_done:
            next_state, curr_rewards, is_done = self.g.move(state, action)[0]
            next_action = self.g.get_action(self.epsilon_greedy_action(next_state, epsilon))
            delta = curr_rewards + gamma * self.Q[next_state][next_action] - self.Q[state][action]
            self.e[state][action] += 1
            for s in range(self.shape * self.shape):
                for a in range(4):
                    self.Q[s][a] = self.Q[s][a] + \
                                   alpha * delta * self.e[s][a]
                    self.e[s][a] = gamma * l * self.e[s][a]
            state = next_state
            action = next_action
            step += 1
        return step


direction = {0: "up",
             1: "right",
             2: "down",
             3: "left"}


def run_experiment(eps, gamma, epsilon, alpha, q, l=0):
    steps = np.zeros((eps, 1))
    max_initial_q = np.zeros((eps, 1))
    for j in range(eps):
        print(j)
        step = q.run_episode(gamma, epsilon, alpha, l)
        steps[j] = step
        max_initial_q[j] = q.Q[q.shape * (q.shape - 1)][max(q.Q[q.shape * (q.shape - 1)])]
    return steps, max_initial_q


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--alg", "--algorithm", action="store", dest="alg")
    parser.add_option("--size", "--world_size", action="store", dest="size", default=3)
    parser.add_option("--gamma", "--discount_factor", action="store", dest="gamma", default=0.99)
    parser.add_option("--exps", "--experiments", action="store", dest="exps", default=3)
    parser.add_option("--eps", "--episodes", action="store", dest="eps", default=500)
    parser.add_option("--epsilon", "--greedy_epsilon", action="store", dest="epsilon", default=0.1)
    parser.add_option("--alpha", "--learning_rate", action="store", dest="alpha", default=0.1)
    parser.add_option("--lambda", "--sarsa_parameter", action="store", dest="l", default=0.9)

    (options, args) = parser.parse_args()

    size = int(options.size)
    alpha = float(options.alpha)
    epsilon = float(options.epsilon)
    gamma = float(options.gamma)
    expr = int(options.exps)
    eps = int(options.eps)
    l = float(options.l)
    pool = Pool()

    steps = np.zeros((eps, expr))
    max_initial_q = np.zeros((eps, expr))
    results = {}
    for i in range(expr):
        if options.alg == "q":
            results[i] = pool.apply_async(run_experiment, (eps, gamma, epsilon, alpha, QLearning(size)))
        else:
            results[i] = pool.apply_async(run_experiment, (eps, gamma, epsilon, alpha, Sarsa(size), l))
    pool.close()
    pool.join()
    for i in range(expr):
        for j in range(eps):
            steps[j][i] = results.get(i).get()[0][j][0]
            max_initial_q[j][i] = results.get(i).get()[1][j][0]
    avg_step = np.mean(steps, axis=1)
    avg_max_q = np.mean(max_initial_q, axis=1)
    episodes = np.arange(eps)
    plt.plot(episodes, avg_step)
    plt.xlabel("Episodes")
    plt.ylabel("Average steps")
    plt.show()
    plt.plot(episodes, avg_max_q)
    plt.xlabel("Episodes")
    plt.ylabel("Average initial q value")
    plt.show()
