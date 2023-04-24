import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
#Note: I understand that using neural network for this is not the best way (minimax and simple logic will be much better), but others were generic and tried so many times by others. Minimax is especially boring with alpha beta prunings

class NNPlayer():
    def __init__(self, symbol):
        self.symbol = symbol
        self.model = None


    #Trains the model with 10 epoches each for user inputted iterations. 10 epochs worked fine
    def train_model(self, num_iter):
        #dataframe for a dataset showing board state and according best move. I tried to not use datasets and just try a bunch of random moves--the model turned out terrible, so I found a dataset that made it much easier (but supervised learning is less fun)
        df2 = pd.read_csv('data2.csv', sep='\s+', header=None)

        #Creates X and Y variable from the dataframe, where X contians the board state (input for the model) and Y contiains the board state + the best move(output for the model)
        X = []
        Y = []

        for rownum, row in df2.iterrows():
            board = np.zeros((3,3))
            player = 1
            for i in range(len(row)):
                if i != 9:
                    board[i // 3, i % 3] = row[i]
                else:
                    newboard = [0]*9
                    newboard[row[i]] = 1
            X.append(board)
            Y.append(newboard)

        # Convert X and Y to numpy arrays
        X = np.array(X)
        Y = np.array(Y)

        # Define the model using dense layers (The dimensions and total number of layers were determined by simple trial and error). I also used basic activation functions which I thought made sense
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(3, 3)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(9, activation='softmax')
        ])

        # Compile the model (adam was the only gradient descent model I knew of..., loss function didn't really matter but I tried several and just stuck with the last one I tried)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        for i in range(num_iter):
            self.model.fit(X, Y, epochs=10, batch_size=32, validation_split=0.2)  

    #Checks winner, returning the winner (1 or -1).
    # def check_winner(board):
    #     # Check Rows
    #     for i in range(3):
    #         if board[i, 0] == board[i, 1] == board[i, 2] != 0:
    #             return board[i, 0]
    #     # Check Columns
    #     for i in range(3):
    #         if board[0, i] == board[1, i] == board[2, i] != 0:
    #             return board[0, i]
    #     # Check Diagonals
    #     if board[0, 0] == board[1, 1] == board[2, 2] != 0:
    #         return board[1, 1]
    #     if board[0, 2] == board[1, 1] == board[2, 0] != 0:
    #         return board[1, 1]
    #     # No winner
    #     return 0

    #Loads pre-trained model. This model has good accuracy and low loss, so if too lazy one could just load a pre-trained model.
    def load_model(self):
        self.model = keras.models.load_model("saved_model")


    # Model calculates the next move, and returns the array of [x,y] pair of move
    def next_move(self, board):
        # while True:
        #     model = keras.models.load_model("saved_model")
        #     board = np.zeros((3, 3), dtype=int)
        #     player = 1
        #     train = input("train model again(y/n)?: ")
        #     if not (train == 'n'):
        #         model.fit(X, Y, epochs=10, batch_size=32, validation_split=0.2)
        #     print(board)
        #     while True:
        #         model.save("saved_model_1")



        model_input = np.expand_dims(board, axis=0)#adds another dimension to array boards to match input dimensions
        model_output = self.model(model_input)#predicts best move using the model
        available_moves = np.argwhere(board == 0)#ensures model does not pick an unavailable spot for next move
        model_output_reshaped = np.reshape(model_output, (3, 3))#reshapes into 3x3 array
        probabilities = model_output_reshaped[available_moves[:, 0], available_moves[:, 1]]#extract the first and second column of available moves array, into a proabilities vector in the same shape as available_moves
        probabilities /= np.sum(probabilities)  #normalize probability
        model_move_index = np.random.choice(range(len(probabilities)), p=probabilities)#random choice weighed on probability, not simply highest for strategic purposes, according to what I learned a long time ago(I don't know if I know exactly why, but I remember seeing someone do it like this)
        model_move = available_moves[model_move_index]#Finds the move
        return model_move
            #     board[model_move[0], model_move[1]] = -player
                
            #     print(board)
                
            #     # Check for end of game
            #     winner = check_winner(board)
            #     if winner != 0:
            #         if winner == 1:
            #             print("You won!")
            #         elif winner == -1:
            #             print("You lost!")
            #         else:
            #             print("Draw!")
            #         break
            #     if np.count_nonzero(board) == 9:
            #         print("Draw!")
            #         break


            #     # Player's move
            #     row = int(input("Enter row: "))
            #     col = int(input("Enter column: "))
            #     while board[row, col] != 0:
            #         row = int(input("Enter row: "))
            #         col = int(input("Enter column: "))
            #     board[row, col] = player
            #     print(board)
            #     # Check for end of game
            #     winner = check_winner(board)
            #     if winner != 0:
            #         if winner == 1:
            #             print("You won!")
            #         elif winner == -1:
            #             print("You lost!")
            #         else:
            #             print("Draw!")
            #         break
            #     if np.count_nonzero(board) == 9:
            #         print("Draw!")
            #         break
            # again = input("Play again(y/n)?: ")
            # if(again == 'n'):
            #     break
            # else:
            #     continue

