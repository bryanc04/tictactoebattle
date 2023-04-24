from TicTacToe_Qlearning import QLearningPlayer
from TicTacToe_NeuralNetwork import NNPlayer
import numpy as np

def isHumanFirst():
    inp = input("Do you want to go first(y/n). **If playing against Q-learning, you will be second by default.")
    while not isValid(inp, 'y', 'n'):
        inp = input("Do you want to go first(y/n)")
    if inp == 'y':
        return True
    return False


def isValid(input,option1, option2):
    if input != option1 and input != option2:
        return False
    else:
        return True

def TrainQlearningPlayer(model):
    model2 = NNPlayer('O')
    model2.load_model()
    alpha = model.alpha
    gamma = model.gamma
    epsilon = model.epsilon
    q_table = model.q_table
    board = [' '] * 9
    dict = {'X': 1, 'O': -1, ' ': 0}
    player = 'X'
    while True:
        if player == 'X':
            action = model.next_move(board)
        else:
            tmpboard = np.zeros((3,3), dtype=int)

            for i in range(len(board)):
                tmpboard[i//3, i %3] = dict[board[i]]

            action = model2.next_move(tmpboard)
        action = (action[0] * 3 + action[1])

        reward, next_board = model.explore_next(board, player, action)
        next_state = model.get_board(next_board)
        model.update_q_table(board, action, reward, next_board)
        if model.check_winner(next_board):
            if player == 'X':
                return 1
            else:
                return -1
        elif model.check_draw(next_board):
            return 0
        board = next_board
        player = 'O' if player == 'X' else 'X'


def isTrainQ():
    inp = input("Do you want to train the Q learning agent(y/n) If not the agent will be pre-trained with pretty good accuracy but not perfect: ")
    while not isValid(inp, 'y', 'n'):
        inp = input("Do you want to train the Q learning agent(y/n): ")
    if inp == 'y':
        num_iter = input("How many times do you want to train the agent?(Recommmended at least 100,000 for decent precison.)")
        while not num_iter.isnumeric():
            num_iter = input("How many times do you want to train the agent?(Recommmended at least 100,000 for decent precision.)")
        return num_iter
    return False
    
def isTrainNN():
    inp = input("Do you want to train the neural network(y/n) If not the model will be pre-trained with pretty good precision but not perfect: ")
    while not isValid(inp, 'y', 'n'):
        inp = input("Do you want to train the neural network(y/n): ")
    if inp == 'y':
        num_iter = input("How many times do you want to train the neural network with a epoch of 10?(Recommmended between 10-100)")
        while not num_iter.isnumeric():
            num_iter = input("How many times do you want to train the neural network with a epoch of 10?(Recommmended between 10-100)")
        return num_iter
    return False

def isAgainstQ():
    inp = input("What do you want to play against? Enter 1 to play against the Q learning agent, 2 to play against the Neural Network: ")
    while not isValid(inp, '1', '2'):
        inp = input("What do you want to play against? Enter 1 to play against the Q learning agent, 2 to play against the Neural Network: ")
    if inp == '1':
        return True
    return False

def print_board(board):
    symbol_map = {-1: 'O', 0: '-', 1: 'X'}
    print("-----")
    for i in range(len(board)):
        row = [symbol_map[board[i][j]] for j in range(len(board[i]))]
        row_str = '|'.join(row)
        print(row_str)
        print("-----")

    # Checks winner, returning the winner (1 or -1).
def check_winner(board):
    # Check Rows
    for i in range(3):
        if board[i, 0] == board[i, 1] == board[i, 2] != 0:
            return board[i, 0]
    # Check Columns
    for i in range(3):
        if board[0, i] == board[1, i] == board[2, i] != 0:
            return board[0, i]
    # Check Diagonals
    if board[0, 0] == board[1, 1] == board[2, 2] != 0:
        return board[1, 1]
    if board[0, 2] == board[1, 1] == board[2, 0] != 0:
        return board[1, 1]
    # No winner
    return 0

def check_draw(board):
    return np.count_nonzero(board == 0) == 0

def main():
    if isHumanFirst():
        pSymbol = 'X'
        cSymbol = 'O'
    else:
        pSymbol = 'O'
        cSymbol = 'X'

    againstQ = isAgainstQ()
    if againstQ:
        cSymbol = 'X'
        pSymbol = 'O'
        model = QLearningPlayer(cSymbol)
        num_train = int(isTrainQ())
        if num_train:
            print("Training....")
            wins = 0
            losses = 0
            draws = 0
            for i in range(num_train):
                print(f"{i}/{num_train}")
                result = TrainQlearningPlayer(model)
                if result == 1:
                    wins += 1
                elif result == -1:
                    losses += 1
                else:
                    draws += 1
            print(f"Training result: Wins: {wins}, Losses: {losses}, Draws: {draws}")
        else:
            model.load_q_table()
    else:
        model = NNPlayer(cSymbol)
        num_train = int(isTrainNN())
        if num_train:
            model.train_model(num_train)
        else:
            model.load_model()
    while True:
        player = 'X'
        board = np.zeros((3,3), dtype=int)
        symbol_map = {'O': -1, 'X': 1}
        while True:
            print_board(board)
            if player == pSymbol:
                pos = input("Enter square(1-9): ")
                while not pos.isnumeric():
                    if not 0 < pos < 10 and not board[(int(pos)-1)//3, (int(pos)-1)%3] == 0:
                        print("Invalid!")
                        pos = input("Enter square(1-9): ")
                pos = int(pos)
                row = (pos-1) // 3
                col = (pos-1) % 3
            else:
                pos = model.next_move(board)
                row = pos[0]
                col = pos[1]
            board[row,col] = symbol_map[player]
            winner = check_winner(board)
            if winner:
                print_board(board)
                if player == pSymbol:
                    print("You won!")
                    break
                else:
                    print("You lost!")
                    break
            else:
                if check_draw(board):
                    print("Draw!")
                    break
            player = 'O' if player == 'X' else 'X'
        again = input("Play again(y/n)?: ")
        if(again == 'n'):
            break
        else:
            continue 


            
main()


