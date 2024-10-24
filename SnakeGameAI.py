import torch
import random
import matplotlib.pyplot as plt
from SnakeGame import SnakeGame, DIR
from collections import deque
import numpy as np
from model import Linear_QNet, QTrainer
from time import perf_counter_ns
from plot import plot

# Settings
GRID_RES = (24, 32)
GRID_PIXEL = 20
MPS = 100

# Agent settings
MAX_MEMORY : int = 1_000
BATCH_SIZE : int = 100
RATE_RANDOM_CHOICES : int = 100

LR = 0.01      # Learning Rate
GAMMA = 0.9     # discount rate for deep Q Learning



CHOICE = dict()
CHOICE[0] = [2, 3]
CHOICE[1] = [2, 3]
CHOICE[2] = [0, 1]
CHOICE[3] = [0, 1]

class SnakeGameAI (SnakeGame):

    def __init__(self, model, trainer):
        super(SnakeGameAI, self).__init__(GRID_RES, GRID_PIXEL, MPS)

        self.n_games : int   = 0
        self.eps     : int   = 0    # ramdomness parameter

        self.memory = deque(maxlen = MAX_MEMORY)

        self.model = model
        self.trainer = trainer
        
        for key, val in DIR.items():
            if np.array_equal(self.direction, val):
                act = key
        self.action : int = list(DIR.keys()).index(act)

    def get_state(self, game_over):

        # ! State of the model (8 values)
        # * Danger around the head   (4 values) [2 verticle and 2 Horizontal components]
        # * Movement Direction       (2 values) [verticle and Horizontal components]
        # * Food Direction           (2 values) [verticle and Horizontal components]


        danger      = np.zeros((4,), dtype=int)
        food        = np.zeros((4,), dtype=int)
        direction   = np.zeros((4,), dtype=int)
        game_over   = np.array([game_over], dtype=int)

        # Calculate food state and distance
        distance = self.food - self.snake[0]
        food[0] = distance[0] < 0 # UP
        food[1] = distance[0] > 0 # DOWN
        food[2] = distance[1] < 0 # LEFT
        food[3] = distance[1] > 0 # RIGHT
        distance = (np.sqrt(np.sum(distance**2)))
        

        # Calculate danger direction
        for idx, key in enumerate(DIR.keys()):
            new_pos = self.snake[0] + DIR[key]
            danger[idx] = 1 if self._is_snake(new_pos) or self._is_wall(new_pos) \
                            else 0
            
        # Calculate movement direction
        direction[0] = np.array_equal(self.direction, DIR['UP'])
        direction[1] = np.array_equal(self.direction, DIR['DOWN'])
        direction[2] = np.array_equal(self.direction, DIR['LEFT'])
        direction[3] = np.array_equal(self.direction, DIR['RIGHT'])

        # print(self.direction)
        # print(self.food)
        # print(np.vstack((danger, direction, food)))
        
        return distance, np.hstack((danger, direction, food, game_over))

    def get_action(self, state) -> None:
        # Eploration using randon movement direction
        self.eps = max(RATE_RANDOM_CHOICES - self.n_games, 5)
        if random.randint(0, RATE_RANDOM_CHOICES) < self.eps:
            # print('Action: ', self.action)
            # print('Action Choice: ', CHOICE[self.action])
            act = random.choice(CHOICE[self.action])
            # print('Chosen Action: ', act)
            # print("Random Move")

        else:
            act = torch.argmax(self.model(torch.tensor(state, dtype=torch.float))).item()

        if act in CHOICE[self.action]:
            self.direction = DIR[list(DIR.keys())[act]]
            self.action = act

        return self.action

    def remember_state(self, state_old, action, reward, state_new, game_over) -> None:
        self.memory.append((state_old, action, reward, state_new, game_over))

    def train_long_memory(self) -> None:
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        state_old, action, reward, state_new, game_over = zip(*mini_sample)
        self.trainer.train_step(state_old, action, reward, state_new, game_over)

    def train_short_memory(self, state_old, action, reward, state_new, game_over) -> None:
        self.trainer.train_step(state_old, action, reward, state_new, game_over)

    def get_score(self) -> int:
        return self.score


def train() -> None:
    # # Run on GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    model = Linear_QNet(13, 128, 4)
    ai_game  = SnakeGameAI(model, QTrainer(model, lr=LR, gamma=GAMMA))
    
    plot_scores, plot_mean_scores = ai_game.model.load()

    time : int = perf_counter_ns()
    NS_PER_FRAME : float = 1e+9 / MPS # nano second per frame

    # total_reward = 0
    total_score = sum(plot_scores)
    ai_game.n_games = len(plot_scores)
    game_over = False

    # get old state
    old_distance, state_old = ai_game.get_state(game_over)

    while not ai_game._quit_event():

        if ((perf_counter_ns() - time) - NS_PER_FRAME) > -1e3:
            time = perf_counter_ns()    # Reset timer

            # get new move 
            action = ai_game.get_action(state_old)

            # calculate next frame
            game_over = not ai_game.step_snake()

            # get new state
            new_distance, state_new = ai_game.get_state(game_over)


            # ! Reward setting 
            # * move towards food   +1
            # * move away from food -1
            # * Eat food            +10
            # * Game over           -10
            if game_over:
                reward = -10
            elif ai_game.is_food:
                reward = 10
                old_distance = GRID_RES[0]
            else:
                reward = 1 if (old_distance - new_distance) >= 0 \
                         else -1
                old_distance = new_distance
                # reward = 0
            reward *= -1
            # train short memory
            ai_game.train_short_memory(state_old, action, reward, state_new, game_over)

            # total_reward += reward
            # print(reward)

            # remember
            ai_game.remember_state(state_old, action, reward, state_new, game_over)

            state_old = state_new

            if game_over:
                # train long memory
                ai_game.train_long_memory()

                # update the plot
                plot_scores.append(ai_game.get_score())
                total_score += ai_game.get_score()
                ai_game.n_games += 1
                plot_mean_scores.append(total_score / ai_game.n_games)
                total_reward = 0
                plot(plot_scores, plot_mean_scores)
                # print('game_over')

                # # save the model
                # ai_game.model.save(plot_scores, plot_mean_scores)

                # reset the game
                ai_game.game_reset()


    # save the model
    ai_game.model.save(plot_scores, plot_mean_scores)
            
            

if __name__ == '__main__':
    train()