import random
import game
import sys
import numpy as np

# Author:				chrn (original by nneonneo)
# Date:				11.11.2016
# Description:			The logic of the AI to beat the game.

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

LINEAR_WEIGHT_MATRIX = np.array([
    [16, 15, 14, 13],
    [9, 10, 11, 12],
    [8, 7, 6, 5],
    [1, 2, 3, 4]
])

QUADRATIC_WEIGHT_MATRIX = np.array([
    [16 ** 2, 15 ** 2, 14 ** 2, 13 ** 2],
    [9 ** 2, 10 ** 2, 11 ** 2, 12 ** 2],
    [8 ** 2, 7 ** 2, 6 ** 2, 5 ** 2],
    [1 ** 2, 2 ** 2, 3 ** 2, 4 ** 2]
])

POWER_2_WEIGHT_MATRIX = np.array([
    [2 ** 15, 2 ** 14, 2 ** 13, 2 ** 12],
    [2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11],
    [2 ** 7, 2 ** 6, 2 ** 5, 2 ** 4],
    [2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3]
])

POWER_4_WEIGHT_MATRIX = np.array([
    [4 ** 15, 4 ** 14, 4 ** 13, 4 ** 12],
    [4 ** 8, 4 ** 9, 4 ** 10, 4 ** 11],
    [4 ** 7, 4 ** 6, 4 ** 5, 4 ** 4],
    [4 ** 0, 4 ** 1, 4 ** 2, 4 ** 3]
])

POWER_8_WEIGHT_MATRIX = np.array([
    [8 ** 15, 8 ** 14, 8 ** 13, 8 ** 12],
    [8 ** 8, 8 ** 9, 8 ** 10, 8 ** 11],
    [8 ** 7, 8 ** 6, 8 ** 5, 8 ** 4],
    [8 ** 0, 8 ** 1, 8 ** 2, 8 ** 3]
])

POWER_10_WEIGHT_MATRIX = np.array([
    [10 ** 15, 10 ** 14, 10 ** 13, 10 ** 12],
    [10 ** 8, 10 ** 9, 10 ** 10, 10 ** 11],
    [10 ** 7, 10 ** 6, 10 ** 5, 10 ** 4],
    [10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
])


def get_monotonicity(log_board):
    score = 0
    # Check rows and columns
    for i in range(4):
        row = log_board[i, :]
        col = log_board[:, i]
        for j in range(3):
            # If current tile is larger than next, it's good (decreasing order)
            if row[j] >= row[j+1]: score += 1
            if col[j] >= col[j+1]: score += 1
    return score

def get_smoothness(log_board):
    penalty = 0
    # Measure the cliff between neighbors
    for i in range(4):
        for j in range(3):
            # Horizontal cliffs
            penalty -= abs(log_board[i, j] - log_board[i, j+1])
            # Vertical cliffs
            penalty -= abs(log_board[j, i] - log_board[j+1, i])
    return penalty

def score_board(board: np.ndarray, weight_matrix: np.ndarray) -> int:
    """Calculate the score of the board with a given weight matrix."""

    # Convert tile values to their logarithmic form (base 2) for better scaling
    log_board = np.where(board > 0, np.log2(board), 0)

    # Weights: Foundations (e.g., Exponential Snake)
    w_score = np.sum(log_board * weight_matrix)

    # Bonuses/Penalties: The "Wow" logic
    m_score = get_monotonicity(log_board) * 100
    s_score = get_smoothness(log_board) * 50

    return w_score + m_score + s_score


def find_best_move(board):
    bestmove = -1
    best_score = -1
    weight_matrix = POWER_4_WEIGHT_MATRIX

    for move in [UP, DOWN, LEFT, RIGHT]:
        newboard = execute_move(move, board)
        if not board_equals(board, newboard):
            score = score_board(newboard, weight_matrix)
            if score > best_score:
                best_score = score
                bestmove = move

    return bestmove

def find_best_move_random_agent():
    return random.choice([UP, DOWN, LEFT, RIGHT])


def execute_move(move, board):
    """
    move and return the grid without a new random tile 
	It won't affect the state of the game in the browser.
    """

    if move == UP:
        return game.merge_up(board)
    elif move == DOWN:
        return game.merge_down(board)
    elif move == LEFT:
        return game.merge_left(board)
    elif move == RIGHT:
        return game.merge_right(board)
    else:
        sys.exit("No valid move")


def board_equals(board, newboard):
    """
    Check if two boards are equal
    """
    return (newboard == board).all()
