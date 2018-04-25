import numpy as np
import math


class LookupTable:

    def __init__(self):
        self.GAME_DIM = 4
        self.TUPLE_SIZE = 17
        self.LUT = {}
        for i in range(self.TUPLE_SIZE):
            self.LUT[i] = np.zeros((17, 17, 17, 17))

    def normalize_matrix(self, mat):
        mat = [item for sublist in mat for item in sublist]
        norm_mat = [0] * self.GAME_DIM * self.GAME_DIM
        for i in range(self.GAME_DIM * self.GAME_DIM):
            if mat[i] == 0:
                continue
            else:
                norm_mat[i] = int(math.log(mat[i], 2))
        return norm_mat

    def load_from_file(self):
        for i in range(self.TUPLE_SIZE):
            self.LUT[i] = np.load("TDvalues[" + str(i) + "].npy")

    def make_tuples(self, mat):
        norm_mat = self.normalize_matrix(mat)
        tuples = {}
        tuples[0] = [norm_mat[0], norm_mat[1], norm_mat[2], norm_mat[3]]
        tuples[1] = [norm_mat[4], norm_mat[5], norm_mat[6], norm_mat[7]]
        tuples[2] = [norm_mat[8], norm_mat[9], norm_mat[10], norm_mat[11]]
        tuples[3] = [norm_mat[12], norm_mat[13], norm_mat[14], norm_mat[15]]

        tuples[4] = [norm_mat[0], norm_mat[4], norm_mat[8], norm_mat[12]]
        tuples[5] = [norm_mat[1], norm_mat[5], norm_mat[9], norm_mat[13]]
        tuples[6] = [norm_mat[2], norm_mat[6], norm_mat[10], norm_mat[14]]
        tuples[7] = [norm_mat[3], norm_mat[7], norm_mat[11], norm_mat[15]]

        tuples[8] = [norm_mat[0], norm_mat[1], norm_mat[5], norm_mat[4]]
        tuples[9] = [norm_mat[1], norm_mat[2], norm_mat[6], norm_mat[5]]
        tuples[10] = [norm_mat[2], norm_mat[3], norm_mat[7], norm_mat[6]]

        tuples[11] = [norm_mat[4], norm_mat[5], norm_mat[9], norm_mat[8]]
        tuples[12] = [norm_mat[5], norm_mat[6], norm_mat[10], norm_mat[9]]
        tuples[13] = [norm_mat[6], norm_mat[7], norm_mat[11], norm_mat[10]]

        tuples[14] = [norm_mat[8], norm_mat[9], norm_mat[13], norm_mat[12]]
        tuples[15] = [norm_mat[9], norm_mat[10], norm_mat[14], norm_mat[13]]
        tuples[16] = [norm_mat[10], norm_mat[11], norm_mat[15], norm_mat[14]]
        return tuples

    def look_up(self, tuple, n_tuple):
        return self.LUT[n_tuple][tuple[0]][tuple[1]][tuple[2]][tuple[3]]

    def get_q_val(self, tuples):
        q = 0
        for i in range(self.TUPLE_SIZE):
            q += self.look_up(tuples[i], i)
        return q

    def func_approx(self, mat):
        tuples = self.make_tuples(mat)
        q_val = self.get_q_val(tuples)
        return q_val

    def update_mat_q(self, s, diff):
        tuple_state = self.make_tuples(s)
        i = 0
        for i in range(self.TUPLE_SIZE):
            t = tuple_state[i]
            n_tuple_q_val = self.look_up(t, i) + diff
            self.update_tuple_q(t, i, n_tuple_q_val)

    def update_tuple_q(self, tuple, n_tuple, val):
        self.LUT[n_tuple][tuple[0]][tuple[1]][tuple[2]][tuple[3]] = val
