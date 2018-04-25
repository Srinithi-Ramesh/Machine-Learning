import pygame
import sys
from pygame.locals import *
from board import Board
from copy import copy


class Simulator:

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)

    WIDTH = 115
    HEIGHT = 115

    MARGIN = 8

    def __init__(self):
        pygame.init()
        self.WINDOW_SIZE = [500, 500]
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        pygame.display.set_caption("2048")
        self.board = Board(4, 4, max_random_value=4)
        self.FONT = pygame.font.SysFont("Arial", 24, bold=True)
        self.clock = pygame.time.Clock()
        self.display()

    # draw some text into an area of a surface
    # automatically wraps words
    # returns any text that didn't get blitted
    def drawText(self, surface, text, color, rect, font, aa=False, bkg=None):
        rect = Rect(rect)
        y = rect.top
        lineSpacing = -2

        # get the height of the font
        fontHeight = font.size("Tg")[1]

        while text:
            i = 1

            # determine if the row of text will be outside our area
            if y + fontHeight > rect.bottom:
                break

            # determine maximum width of line
            while font.size(text[:i])[0] < rect.width and i < len(text):
                i += 1

            # if we've wrapped the text, then adjust the wrap to the last word
            if i < len(text):
                i = text.rfind(" ", 0, i) + 1

            # render the line and build it to the surface
            if bkg:
                image = font.render(text[:i], 1, color, bkg)
                image.set_colorkey(bkg)
            else:
                image = font.render(text[:i], aa, color)

            surface.blit(image, (rect.left, y))
            y += fontHeight + lineSpacing

            # remove the text we just blitted
            text = text[i:]

        return text

    def display(self):

        self.screen.fill(self.BLACK)
        for row in range(self.board.shape[0]):
            for column in range(self.board.shape[1]):
                color = self.WHITE
                if self.board.matrix[row, column] == 0:
                    color = self.RED
                elif (row, column) == self.board.last_random_tile_index:
                    color = self.YELLOW
                pygame.draw.rect(self.screen,
                                 color,
                                 [(self.MARGIN + self.WIDTH) * column + self.MARGIN,
                                  (self.MARGIN + self.HEIGHT) * row + self.MARGIN,
                                  self.WIDTH,
                                  self.HEIGHT])
                self.drawText(self.screen,
                              str(self.board.matrix[row, column]),
                              self.BLACK,
                              [(self.MARGIN + self.WIDTH) * column + self.WIDTH/2,
                               (self.MARGIN + self.HEIGHT) * row + self.HEIGHT/2,
                               self.WIDTH,
                               self.HEIGHT],
                              self.FONT,
                              aa=True)

    def play(self, event):
        if event == 'QUIT':
            pygame.quit()
            sys.exit()

        if event in ('LEFT', 'RIGHT', 'UP', 'DOWN'):
            if event == 'LEFT':
                moved = self.board.move("left")
            elif event == 'RIGHT':
                moved = self.board.move("right")
            elif event == 'UP':
                moved = self.board.move("up")
            elif event == 'DOWN':
                moved = self.board.move("down")

            after_state = copy(self.board.matrix)
            if moved:
                self.add_random_tile()

                self.game_over()
        self.clock.tick(60)
        pygame.display.flip()
        return self.board.score, after_state, self.board.matrix

    def add_random_tile(self):
        self.board.insert_random_tile()
        self.display()

    def game_over(self):
        return self.board.check_gameover()
