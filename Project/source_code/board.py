import numpy as np
from random import randint, choice as random_choice


class Board(object):
    # Represents the board of 2048 game. Internally it is a Numpy 2D array (matrix).
    POSSIBLE_MOVES = ["up", "down", "left", "right"]

    def __init__(self, width, height, max_random_value=4):

        # :param width: the board width
        # :param height: the board height
        # :param max_random_value: which maximal value can have new tile added after each round or at the game start
        self.__tile_values = np.array([2**x for x in range(1, 21)])

        if max_random_value not in self.__tile_values:
            raise ValueError("'max_random_value' must be from numbers of powering 2")
        self.__max_random_value = max_random_value

        self.__random_tile_values = self.__tile_values[self.__tile_values <= max_random_value]
        self.__matrix = self.__get_init_matrix(width, height)
        self.__last_random_tile_index = None
        self.__score = 0

    @property
    def matrix(self):
        return self.__matrix

    @matrix.setter
    def matrix(self, value):
        self.__matrix = value

    @property
    def shape(self):
        # return: Tuple of the gaming board shape, i.e. the Numpy 2D array dimensions -> (rows, columns)
        return self.matrix.shape

    @property
    def last_random_tile_index(self):
        # :return: last inserted random tile index tuple: (row, column)
        return self.__last_random_tile_index

    @property
    def score(self):
        return self.__score

    def __get_init_matrix(self, width, height):
        # Creates the gaming matrix with defined shape and two random initial tiles with given maximal value.
        # return: initialized gaming matrix

        matrix = np.zeros(shape=(height, width), dtype=np.int32)
        max_index_height = height - 1
        max_index_width = width - 1
        first_random_tile = (randint(0, max_index_height), randint(0, max_index_width))
        second_random_tile = (randint(0, max_index_height), randint(0, max_index_width))

        while second_random_tile == first_random_tile:
            second_random_tile = (randint(0, max_index_height), randint(0, max_index_width))

        matrix[first_random_tile] = np.random.choice(self.__random_tile_values)
        matrix[second_random_tile] = np.random.choice(self.__random_tile_values)

        return matrix

    def __move_line(self, array):

        # Moves and merges the tiles on one line of the gaming board.
        # 1) Count how many zero tiles are there.
        # 2) Extract the non-zero tiles in the reverse order.
        # 3) Merge them, count arising zeros.
        # 4) Reverse them to original order.
        # 5) Add zeros before them.
        #
        # example:
        # 1) split to the zero and non-zero tiles: [2 4 4 4 0 0 2 0 0 0 2] = [0 0 0 0 0] + [2 4 4 4 2 2]
        # 2) reverse the non-zero tiles array: [2 4 4 4 2 2] -> [2 2 4 4 4 2]
        # 3) merge the non-zero tiles and count the arising zeros: [2 2 4 4 4 2] -> [4 8 4 2] + [0 0]
        # 4) reverse to the original order: [4 8 4 2] -> [2 4 8 4]
        # 5) combine with the original and 'merge' zeros: [0 0 0 0 0] + [0 0] + [2 4 8 4] = [0 0 0 0 0 0 0 2 4 8 4]
        #
        # result: [2 4 4 4 0 0 2 0 0 0 2] -> [0 0 0 0 0 0 0 2 4 8 4]
        #
        # :param array: the numpy 1D array to be moved and merged
        #
        # >>> list(__move_line(np.array([2, 4, 4, 4, 0, 0, 2, 0, 0, 0, 2])))
        # [0, 0, 0, 0, 0, 0, 0, 2, 4, 8, 4]
        # >>> list(__move_line(np.array([2, 0, 0, 8, 0, 0, 4, 0, 0, 0, 2])))
        # [0, 0, 0, 0, 0, 0, 0, 2, 8, 4, 2]
        #

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
                self.__score += 2 * new_array[i]
                merge_zeros_count += 1
            else:
                merge_array.append(new_array[i])
            i += 1

        merge_array = np.flipud(merge_array)
        zeros = (zeros_count + merge_zeros_count) * [0]
        zeros.extend(merge_array)
        return np.array(zeros)

    def move(self, direction):

        # Moves the tiles to defined direction.
        # It slices the matrix to rows or lines and send them ordered in the movement
        # direction to the __move_line function.
        #
        # example:
        # matrix = [
        #     [2,  2,  4, 2, 8],
        #     [16, 32, 2, 8, 8],
        #     [32, 2,  2, 0, 0],
        #     [0,  0,  0, 0, 0],
        #     [0,  0,  2, 2, 0]
        # ]
        #
        # 'up' slices are columns from bottom to up: [0, 0, 32, 16, 2], [0, 0, 2, 32, 2] etc.
        # 'right': slices are rows from left to right: [2,  2,  4, 2, 8], [16, 32, 2, 8, 8] etc.
        #
        # :param direction: direction to move on: 'up', 'down', 'left', 'right'
        # :return: True if move is possible, False otherwise.
        #

        original_matrix = np.copy(self.matrix)
        if direction == "up":
            lines_cols = [np.flipud(self.matrix[:, i]) for i in range(self.shape[1])]
            for i, line in enumerate(lines_cols):
                self.matrix[:, i] = np.flipud(self.__move_line(line))
        elif direction == "down":
            lines_cols = [self.matrix[:, i] for i in range(self.shape[1])]
            for i, line in enumerate(lines_cols):
                self.matrix[:, i] = self.__move_line(line)
        elif direction == "right":
            lines_rows = [self.matrix[i, :] for i in range(self.shape[0])]
            for i, line in enumerate(lines_rows):
                self.matrix[i, :] = self.__move_line(line)
        elif direction == "left":
            lines_rows = [np.flipud(self.matrix[i, :]) for i in range(self.shape[0])]
            for i, line in enumerate(lines_rows):
                self.matrix[i, :] = np.flipud(self.__move_line(line))
        else:
            raise ValueError("Unknown direction to move. Possible directions are 'up', 'down', 'left', 'right'")

        if np.array_equal(original_matrix, self.matrix):
            return False
        else:
            return True

    def insert_random_tile(self):

        # Inserts the random tile.
        # :return: True if random tile was added, False otherwise (= the board is full).
        zero_indexes = np.where(self.matrix == 0)

        if len(zero_indexes[0]):
            random_zero_index = random_choice(list(zip(zero_indexes[0], zero_indexes[1])))
            self.matrix[random_zero_index] = np.random.choice(self.__random_tile_values)
            self.__last_random_tile_index = random_zero_index
            return True
        else:
            return False

    def move_insert(self, direction):

        # Combines the move() and insert_random_tile() functions.
        # :param direction: direction to move on: 'up', 'down', 'left', 'right'
        # :return: (True, True) if moved and inserted, (True, False) if moved and not inserted and (False, False)
        # when not moved.

        if self.move(direction):
            inserted = self.insert_random_tile()

            if inserted:
                return (True, True)
            else:
                return (True, False)
        else:
            return (False, False)

    def is_full(self):

        # Checks if the board contains zero tiles.
        # :return: True if board doesn't contain zero tiles, False otherwise.

        return not bool(len(np.where(self.matrix == 0)[0]))

    def check_gameover(self):
        # Checks if there are possible moves and if not the game is over.
        # :return: True if game is over, False otherwise.

        original_matrix = np.copy(self.matrix)

        if self.is_full():
            move_results = []
            for move in self.POSSIBLE_MOVES:
                move_results.append(self.move(move))
            self.matrix = original_matrix

            return not any(move_results)
        else:
            return False
