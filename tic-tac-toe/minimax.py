from tictactoe import *

# This is a minimax implementation for tic tac toe

def valueOfPosition(board, maxplayerLetter, minplayerLetter):
    if isWinner(board, maxplayerLetter):
        return 1
    if isWinner(board, minplayerLetter):
        return -1
    else:
        return 0

def minimax(board, steps, totalNumSteps, maximizingPlayer, playerLetter, adversaryLetter, savedMove=0):
    if steps == 0 or gameEnded(board, playerLetter, adversaryLetter):
        #print('Game has ended')
        if maximizingPlayer:
            return (valueOfPosition(board, playerLetter, adversaryLetter), -1)
        else:
            return (valueOfPosition(board, adversaryLetter, playerLetter), -1)          
    
    if maximizingPlayer:
        maxVal = -100 # should be inf
        listOfMoves = getListOfMoves(board)
        for move in listOfMoves:
            #print('Trying move {0}'.format(move))
            copy = list(board)
            makeMove(copy, playerLetter, move)
            val = minimax(copy, steps-1, totalNumSteps, False, adversaryLetter, playerLetter, savedMove)[0]
            #print('Value is {0}'.format(val))
            if val > maxVal:
                maxVal = val
                if steps == totalNumSteps:
                    savedMove = move    
        #print('Optimal Choice is move {0} with value {1}'.format(savedMove, maxVal))
        return (maxVal, savedMove)
    else:
        minVal = 100
        listOfMoves = getListOfMoves(board)
        for move in listOfMoves:
            copy = list(board)
            makeMove(copy, playerLetter, move)
            val = minimax(copy, steps-1, totalNumSteps, True, adversaryLetter, playerLetter, savedMove)[0]
            minVal = min(val, minVal)
        return (minVal, savedMove)

def test():
    exBoard  = [' ', 'o', 'o', ' ', ' ', 'x', ' ', 'x', 'o', ' ']
    exBoard2 = [' ', 'o', 'x', ' ', ' ', ' ', ' ', 'x', ' ', ' '] 
    exBoard3 = [' ', 'o', 'x', ' ', ' ', 'x', ' ', 'x', ' ', ' '] 
    exBoard4 = [' ', 'x', ' ', 'o', ' ', ' ', ' ', ' ', ' ', 'x']
    minimax(exBoard3, 1, 1, True, 'x', 'o')
