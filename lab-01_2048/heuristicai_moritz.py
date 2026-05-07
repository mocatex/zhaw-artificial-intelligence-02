import random
import game
import sys
import math

# Author:				chrn (original by nneonneo)
# Date:				11.11.2016
# Description:			The logic of the AI to beat the game.

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

LINEAR_WEIGHT_MATRIX = [
    [16, 15, 14, 13],
    [9, 10, 11, 12],
    [8, 7, 6, 5],
    [1, 2, 3, 4]
]

QUADRATIC_WEIGHT_MATRIX = [
    [16 ** 2, 15 ** 2, 14 ** 2, 13 ** 2],
    [9 ** 2, 10 ** 2, 11 ** 2, 12 ** 2],
    [8 ** 2, 7 ** 2, 6 ** 2, 5 ** 2],
    [1 ** 2, 2 ** 2, 3 ** 2, 4 ** 2]
]

QUARTIC_WEIGHT_MATRIX = [
    [16 ** 4, 15 ** 4, 14 ** 4, 13 ** 4],
    [9 ** 4, 10 ** 4, 11 ** 4, 12 ** 4],
    [8 ** 4, 7 ** 4, 6 ** 4, 5 ** 4],
    [1 ** 4, 2 ** 4, 3 ** 4, 4 ** 4]
]

POWER_2_WEIGHT_MATRIX = [
    [2 ** 15, 2 ** 14, 2 ** 13, 2 ** 12],
    [2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11],
    [2 ** 7, 2 ** 6, 2 ** 5, 2 ** 4],
    [2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3]
]

POWER_4_WEIGHT_MATRIX = [
    [4 ** 15, 4 ** 14, 4 ** 13, 4 ** 12],
    [4 ** 8, 4 ** 9, 4 ** 10, 4 ** 11],
    [4 ** 7, 4 ** 6, 4 ** 5, 4 ** 4],
    [4 ** 0, 4 ** 1, 4 ** 2, 4 ** 3]
]


def get_monotonicity(log_board):
    """
    Calculate monotonicity score for rows and columns.
    -> each tile should be larger than the next one in the direction of the snake.
    """
    score = 0
    for i in range(4):
        for j in range(3):
            # Horizontal check
            if log_board[i][j] >= log_board[i][j + 1]:
                score += 1
            # Vertical check
            if log_board[j][i] >= log_board[j + 1][i]:
                score += 1
    return score

def get_smoothness(log_board):
    """
    Calculate smoothness penalty for rows and columns.
    -> penalize large differences between adjacent tiles (cliffs).
    """
    penalty = 0
    for i in range(4):
        for j in range(3):
            # Horizontal difference
            penalty -= (abs(log_board[i][j] - log_board[i][j + 1]) ** 2)
            # Vertical difference
            penalty -= (abs(log_board[j][i] - log_board[j + 1][i]) ** 2)
    return penalty


def get_isolation_penalty(board):
    """
    Calculate isolation penalty for tiles that have no compatible neighbors.
     - A tile is "lonely" if it has no adjacent tile of the same value or a value that can merge with it (half/double).
     - Penalize lonely big tiles more, as they are harder to merge and can block the board.
     - This encourages the AI to keep tiles clustered together for better merging opportunities.
    """
    penalty = 0
    # Assuming board is a list of lists (4x4)
    for r in range(4):
        for c in range(4):
            val = board[r][c]
            if val == 0: continue

            has_neighbor = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 4 and 0 <= nc < 4:
                    neighbor = board[nr][nc]
                    if neighbor == val or neighbor == val * 2 or neighbor == val / 2:
                        has_neighbor = True
                        break
            if not has_neighbor:
                penalty -= math.log2(val) * 10
    return penalty

def get_merge_potential(board):
    """
    Calculate merge potential bonus for tiles that have 'virtual' neighbors.
    -> A tile has 'virtual' neighbors if there are non-zero tiles in the same row or column that could merge with it if they were moved together.
    """
    bonus = 0
    for i in range(4):
        row = board[i]
        col = [board[r][i] for r in range(4)]

        for line in [row, col]:
            filtered = [x for x in line if x != 0]
            for j in range(len(filtered) - 1):
                if filtered[j] == filtered[j + 1]:
                    bonus += math.log2(filtered[j]) * 20
    return bonus

def score_board(board, weight_matrix) -> int:
    """Calculate the score of the board with a given weight matrix."""

    # Convert tile values to their logarithmic form (base 2) for better scaling
    # Pre-calculate log_board and w_score in one pass to save loops
    log_board = [[0.0] * 4 for _ in range(4)]
    w_score = 0
    empty_cells = 0

    for r in range(4):
        for c in range(4):
            val = board[r][c]
            if val > 0:
                l_val = math.log2(val)
                log_board[r][c] = l_val
                w_score += l_val * weight_matrix[r][c]
            else:
                empty_cells += 1

    # Coefficients
    C_WEIGHTS, C_SMOOTH, C_LONELY, C_MERGE, C_MONO, C_EMPTY = 1, 16.0, 256.0, 4096.0, 65536.0, 1048576.0

    w_score *= C_WEIGHTS
    m_score = get_monotonicity(log_board) * C_MONO
    s_score = get_smoothness(log_board) * C_SMOOTH
    l_score = get_isolation_penalty(board) * C_LONELY
    b_score = get_merge_potential(board) * C_MERGE
    e_score = empty_cells * C_EMPTY

    return w_score + m_score + s_score + l_score + b_score + e_score


def find_best_move(board):
    bestmove = -1
    best_score = -1
    weight_matrix = QUARTIC_WEIGHT_MATRIX

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
    return newboard == board
