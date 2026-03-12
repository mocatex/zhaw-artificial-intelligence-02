import random
import game
import sys
import numpy as np

# Author:				chrn (original by nneonneo)
# Date:				11.11.2016
# Description:			The logic of the AI to beat the game.

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


# --- Heuristic helpers -------------------------------------------------


def max_in_corner(board: np.ndarray) -> float:
    max_tile = np.max(board)
    corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
    return 1 if max_tile in corners else 0


def best_score(board: np.ndarray) -> float:

    # First we gain corner bonus, we want to keep highest tile in corner
    corner_bonus = max_in_corner(board) * 50

    # Check for empty tiles, we want to have more empty tiles
    empty_tiles = np.sum(board == 0) * 100

    # We want to have monotonic rows and columns, so we check for monotonicity
    monotonicity = 0
    for i in range(4):
        row = board[i, :]
        col = board[:, i]
        if np.all(np.diff(row) <= 0) or np.all(np.diff(row) >= 0):
            monotonicity += 1
        if np.all(np.diff(col) <= 0) or np.all(np.diff(col) >= 0):
            monotonicity += 1
    monotonicity_score = monotonicity * 10

    return corner_bonus + empty_tiles + monotonicity_score




def find_best_move(board):
    bestmove = -1
    best_value = -float("inf")


    for m in [UP, DOWN, LEFT, RIGHT]:
        newboard = execute_move(m, board)
        if not board_equals(board, newboard):
            score = best_score(newboard)
            if score > best_value:
                best_value = score
                bestmove = m

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
    """Check if two boards are equal."""
    return bool(np.array_equal(np.asarray(board), np.asarray(newboard)))
