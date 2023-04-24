import random
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
#For some reason, I can't get this to work perfectly. It works fine but still has flaws

class QLearningPlayer():

    #Epsilon, gamma, alpha are for the q algorithm
    def __init__(self, symbol):
        self.symbol = symbol
        self.q_table = {}
        self.epsilon = 0.1
        self.gamma = 0.3
        self.alpha = 0.9


    # model = keras.models.load_model("saved_model")

    def convert_board(self, board):
        symbol_map = {-1: 'O', 1: 'X', 0: ' '}
        tmpboard = ''
        for row in board:
            for i in row:
                tmpboard+=symbol_map[i]
        return tmpboard

    #This is a decent pre-trained q table. It works fine. 
    def load_q_table(self):
        with open("tmpqdata2.json") as file:
            self.q_table = json.load(file)

    #Turns board into string to store it in the q_table dictionary
    def get_board(self, board):
        if not type(board) is np.ndarray:
            return ''.join(board)
        else:
            return self.convert_board(board)
                

    # def print_board(board):
    #     print(f' {board[0]} | {board[1]} | {board[2]} ')
    #     print('-----------')
    #     print(f' {board[3]} | {board[4]} | {board[5]} ')
    #     print('-----------')
    #     print(f' {board[6]} | {board[7]} | {board[8]} ')

    def check_draw(self, board):
        return ' ' not in board
    # epsilon-greedy learning
    def select_action(self, board):
        if type(board) is np.ndarray:
            board = self.convert_board(board)
        if random.uniform(0, 1) < self.epsilon:
            # random action
            return random.choice([i for i in range(len(board)) if board[i] == ' '])
        else:
            state = self.get_board(board)
            if not state in self.q_table:#Not explored
                self.q_table[state] = [0] * 9
            max_q_value = max(self.q_table[state])
            best_actions = [i for i in range(len(self.q_table[state])) if self.q_table[state][i] == max_q_value and board[i] == ' ']
            if len(best_actions) == 0:
                return random.choice([i for i in range(len(board)) if board[i] == ' '])#No best action, random exploration
            elif len(best_actions) == 1:
                return best_actions[0]
            else:
                return random.choice(best_actions)

    def check_winner(self, board):
        # Check rows
        for i in range(0, 9, 3):
            if board[i] == board[i+1] == board[i+2] and board[i] != ' ':
                return True
        # Check columns
        for i in range(3):
            if board[i] == board[i+3] == board[i+6] and board[i] != ' ':
                return True
        # Check diagonals
        if board[0] == board[4] == board[8] and board[0] != ' ':
            return True
        if board[2] == board[4] == board[6] and board[2] != ' ':
            return True
        return False


    def explore_next(self, board, player, action):
        next_board = board.copy()

        next_board[action] = player

        if self.check_winner(next_board):#Reward for winning and losing moves
            if player == 'X':
                return 1, next_board
            else:
                return -1, next_board
        elif self.check_draw(next_board):#I think drawing is good, positive reward
            return 0.5, next_board
        else:
            return 0.5, next_board


    def update_q_table(self, board, action, reward, next_board):
        alpha = self.alpha
        gamma = self.gamma

        state = self.get_board(board)
        next_state = self.get_board(next_board)
        #Initialize Q value for unexplored states
        if state not in self.q_table:
            self.q_table[state] = [0] * 9
        if next_state not in self.q_table:
            self.q_table[next_state] = [0] *9
        self.q_table[state][action] += alpha * (reward + gamma * max(self.q_table[next_state]) - self.q_table[state][action])#Q value equation

    # def play_game(q_table, epsilon):
    #     board = [' '] * 9
    #     dict = {'X': 1, 'O': -1, ' ': 0}
    #     player = 'X'
    #     while True:

    #         if player == 'X':
    #             action = select_action(board, q_table, epsilon)
    #         else:
    #             tmpboard = np.zeros((3,3), dtype=int)
    #             for i in range(len(board)):
    #                 tmpboard[i//3, i %3] = dict[board[i]]
    #             model_input = np.expand_dims(tmpboard, axis=0)
    #             model_output = model(model_input)
    #             available_moves = np.argwhere(board == 0)
    #             model_output_reshaped = np.reshape(model_output, (3, 3))
    #             available_moves = np.argwhere(tmpboard == 0)
    #             probabilities = model_output_reshaped[available_moves[:, 0], available_moves[:, 1]]
    #             probabilities /= np.sum(probabilities)
    #             model_move_index = np.random.choice(range(len(probabilities)), p=probabilities)
    #             model_move = available_moves[model_move_index]
    #             model_move = (model_move[0] * 3 + model_move[1])
    #             action = model_move
    #         reward, next_board = explore_next(board, player, action)
    #         next_state = get_board(next_board)
    #         update_q_table(board, action, reward, next_board, q_table, alpha, gamma)
    #         if check_winner(next_board):
    #             if player == 'X':
    #                 return 1
    #             else:
    #                 return -1
    #         elif check_draw(next_board):
    #             return 0
    #         board = next_board
    #         player = 'O' if player == 'X' else 'X'

    # try:
    #     # while True:
    #         # q_table = {}
    #         cnt = 0
    #         wins = 0
    #         losses = 0
    #         draws = 0
    #         max_win_rate = 0
    #         while True:
    #             cnt+=1
    #             result = play_game(q_table, epsilon)
    #             if result == 1:
    #                 wins += 1
    #             elif result == -1:
    #                 losses += 1
    #             else:
    #                 draws += 1
    #             if(cnt %1000 == 0):
    #                 # win_rate = wins / (wins + losses)
    #                 # print(win_rate)
    #                 # if (win_rate > max_win_rate and win_rate > 0.7):
    #                 #     max_win_rate = win_rate
    #                 #     with open("Qdata.json", "w+") as file:
    #                 #         json.dump(q_table, file)
    #                 #         print("updated")
    #                 print(f"Wins: {wins}/ Losses: {losses}/ Draws: {draws}")
    #                 cnt = 0
    #                 wins = 0
    #                 losses = 0
    #                 draws = 0
    #                 max_win_rate = 0
    #         # print(f"Reward for losing is: {lossreward}: Wins:{wins}, Losses:{losses}, Draws:{draws}")
    #         # lossreward -= 0.1
    # except KeyboardInterrupt:
    #     pass
    # print(f'Wins: {wins}, Losses: {losses}, Draws: {draws}')

    #Finds next move
    def next_move(self, board):
        self.epsilon = 0 #Zero because we dont want to explore
        # while True:
        #     board = [' '] * 9
        #     player = 'X'
        #     while True:
        #         print_board(board)
        #         if player == 'X':
        # symbol_map = {-1: 'O', 1: 'X', 0: ' '}
        # tmpboard = ''
        # for row in board:
        #     for i in row:
        #         tmpboard+=symbol_map[i]
        ret = self.select_action(board)
        return [(ret)//3, ret%3]
            #         if check_winner(board):
            #             print_board(board)
            #             if player == 'X':
            #                 print('AI wins!')
            #             else:
            #                 print('You win!')
            #             break
            #         elif check_draw(board):
            #             print_board(board)
            #             print('Draw!')
            #             break
            #     else:
            #         while True:
            #             cell = int(input('Enter cell (1-9): ')) - 1
            #             if board[cell] == ' ':
            #                 board[cell] = 'O'
            #                 break
            #             else:
            #                 print('Cell is not empty!')
            #         if check_winner(board):
            #             print_board(board)
            #             if player == 'X':
            #                 print('AI wins!')
            #             else:
            #                 print('You win!')
            #             break
            #         elif check_draw(board):
            #             print_board(board)
            #             print('Draw!')
            #             break
            #     player = 'O' if player == 'X' else 'X'
            # again = input("Play again(y/n)?: ")
            # if(again == 'n'):
            #     break
            # else:
            #     continue

