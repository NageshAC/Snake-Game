import pygame
from enum import Enum
import numpy as np
import random
from collections import deque
from time import perf_counter_ns

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_w,
    K_s,
    K_a,
    K_d,
    K_x
)

# Game Window Configuration
grid_res = (80, 80)
# speed = 10  # speed hz per movement
grid_pixel = 10
SNAKE_SIZE = 5

MPS = 10 # speed => movement per second -> FPS
# SUB_MPS = 10
# MS_PER_FRAME = 1e-3 / MPS

top_offset      = 25
bottom_offset   = 0
left_offset     = 0
right_offset    = 0

# Colors used in the game window
class COLOR(Enum):
    WALL    = (200,   0,   0)  # RED   color
    BG      = (  0,   0,   0)  # BLACK color
    FOOD    = (  0, 200,   0)  # GREEN color
    SNAKE   = (255, 255, 255)  # WHITE color
    SCORE   = (255, 255, 255)  # WHITE color

class SYMBOL(Enum):
    BG    = 0
    WALL  = 1
    FOOD  = 2
    SNAKE = 3

# class DIR(Enum):
DIR = dict()
DIR['UP']    = np.array([-1,  0])
DIR['RIGHT'] = np.array([ 0,  1])
DIR['DOWN']  = np.array([ 1,  0])
DIR['LEFT']  = np.array([ 0, -1])


def get_pygame_loc (loc: tuple[int, int] | np.ndarray) -> tuple[int, int]:
    return (loc[1], loc[0])

class SnakeGame :

    def __init__(self, grid_res:tuple[int, int], grid_pixel:int, MPS:int, game_name = 'Snake Game') -> None:
        self.grid_pixel = grid_pixel
        self.grid_res = grid_res
        self.MPS = MPS

        self.win_res = get_pygame_loc((self.grid_res[0] * self.grid_pixel + top_offset + bottom_offset, 
                        self.grid_res[1] * self.grid_pixel + left_offset + right_offset))
        
        self.play_area = get_pygame_loc((self.grid_res[0] * self.grid_pixel, 
                          self.grid_res[1] * self.grid_pixel))
        
        # unit cell of snake in the game
        self._std_sqr = (self.grid_pixel, self.grid_pixel)


        #pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode(self.win_res)
        pygame.display.set_caption(game_name)
        self.clock = pygame.time.Clock()

        self.game_reset ()

    def run_game(self):
        time : int = perf_counter_ns()
        NS_PER_FRAME : float = 1e+9 / MPS # nano second per frame 

        while self.handle_keystroke() and not self._quit_event():
            if ((perf_counter_ns() - time) - NS_PER_FRAME) > -1e3:
                time = perf_counter_ns()    # Reset timer
                self.clock.tick()
                if not self.step_snake():
                    break

                print('time per frame: ', int((perf_counter_ns() - time) * 1e-3), ' micro s')
                print('Frame Rate: ', self.clock.get_fps())
        pygame.quit()

    def handle_keystroke (self) -> bool:
        # * Updates direction depending on the user input
         
        key_pressed = pygame.key.get_pressed() # user input dictionary

        if key_pressed [K_x]:
            # print("\n\n------------Quiting------------\n\n from 'X' key\n\n")
            return False
        
        if key_pressed [K_UP] or key_pressed [K_w]:
            # print("\n====== Pressed K_UP ======\n")
            # move UP if the direction in not DOWN
            if not np.array_equal(self.direction, DIR['DOWN']):
                self.direction = DIR['UP']
                # print(self.direction, "\n")

        if key_pressed [K_DOWN] or key_pressed [K_s]:
            # print("\n====== Pressed K_DOWN ======\n")
            # move DOWN if the direction in not UP
            if not np.array_equal(self.direction, DIR['UP']):
                self.direction = DIR['DOWN']
                # print(self.direction, "\n")

        if key_pressed [K_LEFT] or key_pressed [K_a]:
            # print("\n====== Pressed K_LEFT ======\n")
            # move LEFT if the direction in not RIGHT
            if not np.array_equal(self.direction, DIR['RIGHT']):
                self.direction = DIR['LEFT']
                # print(self.direction, "\n")

        if key_pressed [K_RIGHT] or key_pressed [K_d]:
            # print("\n====== Pressed K_RIGHT ======\n")
            # move RIGHT if the direction in not LEFT
            if not np.array_equal(self.direction, DIR['LEFT']):
                self.direction = DIR['RIGHT']
                # print(self.direction, "\n")

        return True

    def step_snake (self) -> bool:
        # print("\n\nStepping...")

        remove_tail = True
        # score_changed = False
        
        # Calculate new position of head
        new_pos = self.snake[0] + self.direction
        # print(new_pos)

        # if new position is wall or snake quit
        if self._is_snake(new_pos):
            # print("\n\nGame Over (Snake)\n\n------------Quiting------------\n\n")
            return False
        
        if self._is_wall(new_pos):
            # print("\n\nGame Over (Wall)\n\n------------Quiting------------\n\n")
            return False
            
        # if food at new position of head create new food
        if self._is_food(new_pos):
            self.score += 1
            # score_changed = True
            self._create_food()
            remove_tail = False
            

        # add new position as head
        self.snake.appendleft(new_pos)
        # if score_changed:
            # self._refresh_score()

        self._draw_square (self._std_sqr, COLOR.SNAKE.name, new_pos)
        if remove_tail:
            self._draw_square (self._std_sqr, COLOR.BG.name, self.snake.pop())
        else:
            self._refresh_score()
        
        pygame.display.flip()

        return True

    def game_reset (self) -> None:
        self.grid = np.zeros(self.grid_res, dtype=int)
        self.score: int = 0
        # self.font = pygame.font.Font('arial.ttf', top_offset)
        self.font = pygame.font.SysFont('arial', top_offset)

        # initialize wall
        self.grid[:,  0] = SYMBOL.WALL.value # left boundry
        self.grid[:, -1] = SYMBOL.WALL.value # right boundry
        self.grid[ 0, :] = SYMBOL.WALL.value # top boundry
        self.grid[-1, :] = SYMBOL.WALL.value # bottom boundry

        self._draw_square (self.play_area, COLOR.WALL.name, (0, 0), False)
        self._draw_square (
            (self.play_area[0]-2*self.grid_pixel, self.play_area[1]-2*self.grid_pixel), 
            COLOR.BG.name, 
            (1, 1),
            False)

        # initialise snake and initial direction
        posible_dir_keys = []
        done = False
        while not done:
            if not posible_dir_keys:
                posible_dir_keys = self._create_snake()

            for s in self.snake:
                self._draw_square (self._std_sqr, COLOR.SNAKE.name, s)

            # print("possible direction = ", posible_dir_keys)

            while posible_dir_keys:

                choice_key = random.choice(posible_dir_keys)
                self.direction : np.ndarray = DIR[choice_key]
                posible_dir_keys.remove(choice_key)

                new_pos = self.snake[0] + self.direction
                if not self._is_snake(new_pos):
                    if not self._is_wall(new_pos):
                        done = True
                        break

        # initialise food
        self._create_food()

        self._refresh_score()

        pygame.display.flip() # initial frame

    def _quit_event(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True 
        
        return False
    
    def _get_pix (self, coord : tuple[int, int]) -> tuple[int, int]:
        return get_pygame_loc((
            coord[0] * self.grid_pixel + top_offset, 
            coord[1] * self.grid_pixel + left_offset
        ))
    
    def _draw_square (self, size : tuple[int, int], typ : str, loc : tuple[int, int], change_grid : bool = True) -> None:
        # print("Drawing square:")
        # print("\tTYP: ", typ)
        # print("\tSIZE: ", size)
        # print("\tLOC: ", loc)
        fg_surf = pygame.Surface(size)
        fg_surf.fill(COLOR[typ].value)
        self.screen.blit(fg_surf, self._get_pix(loc))
        if change_grid:
            self.grid[tuple(loc)] = SYMBOL[typ].value
    
    def _is_snake (self, coord : tuple[int, int]) -> bool:
        if type(coord) == np.ndarray:
            coord = tuple(coord)
        return True if self.grid[coord] == SYMBOL.SNAKE.value else False
    
    def _is_wall (self, coord : tuple[int, int]) -> bool:
        if type(coord) == np.ndarray:
            coord = tuple(coord)
        return True if self.grid[coord] == SYMBOL.WALL.value else False
    
    def _is_food (self, coord : tuple[int, int]) -> bool:
        if type(coord) == np.ndarray:
            coord = tuple(coord)
        self.is_food = True if self.grid[coord] == SYMBOL.FOOD.value else False
        return self.is_food
    
    def _create_snake (self) -> None:
        snake_head = np.array([
            random.randint(1, self.grid_res[0]-2),
            random.randint(1, self.grid_res[1]-2)
        ])
        self.snake = deque()
        self.snake.append(snake_head)

        posible_dir_keys = list(DIR.keys())             # get list of directions  
        choice_key = random.choice(posible_dir_keys)    # choose a key at random
        dir_choice = DIR[choice_key]            # get its coordinates
        posible_dir_keys.remove(choice_key)             # remove that key to make it easier 
                                                # to choose movement direction or 
                                                # to choose an other direction later 

        # print("Snake Choice: ", choice_key)

        for _ in range(SNAKE_SIZE-1):
            while True:
                new_pos = self.snake[-1] + dir_choice
                if not self._is_snake(new_pos):
                    if not self._is_wall(new_pos):
                        break

                choice_key = random.choice(posible_dir_keys)
                dir_choice = DIR[choice_key]
                posible_dir_keys.remove(choice_key)
                # print("Snake Choice: (inner loop) ", choice_key)

            self.snake.append(new_pos)
            
        return posible_dir_keys

    def _create_food (self) -> None:
        while True: 
            self.food = (
                random.randint(1, self.grid_res[0]-2),
                random.randint(1, self.grid_res[1]-2)
            )
            if not self._is_snake(self.food):
                break
        
        self._draw_square (self._std_sqr, COLOR.FOOD.name, self.food)
    
    def _refresh_score (self) -> None:
        # self._draw_square((self.win_res[0], top_offset), COLOR.SNAKE.name, (0,0), False)
        fg_surf = pygame.Surface((self.win_res[1], top_offset))
        fg_surf.fill(COLOR['BG'].value)
        self.screen.blit(fg_surf, (0,0))
        text = self.font.render("Score: " + str(self.score), True, COLOR.SCORE.value)
        self.screen.blit(text, [0, 0])

if __name__ == "__main__":
    SnakeGame(grid_res, grid_pixel, MPS).run_game()