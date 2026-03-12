import random
import game
import sys
import numpy as np

TRANSPOSITION_TABLE = {} # cache for already evaluated board states

# Author:      chrn (original by nneonneo)
# Date:        11.11.2016
# Copyright:   Algorithm from https://github.com/nneonneo/2048-ai
# Description: The logic to beat the game. Based on expectimax algorithm.

QUARTIC_WEIGHT_MATRIX = np.array([
    [16 ** 4, 15 ** 4, 14 ** 4, 13 ** 4],
    [9 ** 4, 10 ** 4, 11 ** 4, 12 ** 4],
    [8 ** 4, 7 ** 4, 6 ** 4, 5 ** 4],
    [1 ** 4, 2 ** 4, 3 ** 4, 4 ** 4]
])

POWER_4_WEIGHT_MATRIX = np.array([
    [4 ** 15, 4 ** 14, 4 ** 13, 4 ** 12],
    [4 ** 8, 4 ** 9, 4 ** 10, 4 ** 11],
    [4 ** 7, 4 ** 6, 4 ** 5, 4 ** 4],
    [4 ** 0, 4 ** 1, 4 ** 2, 4 ** 3]
])

def get_monotonicity(log_board):
    """
    Calculate monotonicity score for rows and columns.
    -> each tile should be larger than the next one in the direction of the snake.
    """
    score = 0
    for i in range(4):
        row = log_board[i, :]
        col = log_board[:, i]
        for j in range(3):
            # If current tile is larger than next -> good
            if row[j] >= row[j+1]: score += 1
            if col[j] >= col[j+1]: score += 1
    return score

def get_smoothness(log_board):
    """
    Calculate smoothness penalty for rows and columns.
    -> penalize large differences between adjacent tiles (cliffs).
    """
    penalty = 0
    for i in range(4):
        for j in range(3):
            # Horizontal cliffs
            penalty -= (abs(log_board[i, j] - log_board[i, j+1]) ** 2)
            # Vertical cliffs
            penalty -= (abs(log_board[j, i] - log_board[j+1, i]) ** 2)
    return penalty


def get_isolation_penalty(board):
    """
    Calculate isolation penalty for tiles that have no compatible neighbors.
     - A tile is "lonely" if it has no adjacent tile of the same value or a value that can merge with it (half/double).
     - Penalize lonely big tiles more, as they are harder to merge and can block the board.
     - This encourages the AI to keep tiles clustered together for better merging opportunities.
    """
    penalty = 0
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == 0: continue

            val = board[r, c]
            has_neighbor = False
            # Check all 4 directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor = board[nr, nc]
                    # If neighbor is same or compatible (half/double)
                    if neighbor == val or neighbor == val * 2 or neighbor == val / 2:
                        has_neighbor = True
                        break
            if not has_neighbor:
                penalty -= np.log2(val) * 10  # Penalize lonely big tiles more
    return penalty

def get_merge_potential(board):
    """
    Calculate merge potential bonus for tiles that have 'virtual' neighbors.
    -> A tile has 'virtual' neighbors if there are non-zero tiles in the same row or column that could merge with it if they were moved together.
    """
    bonus = 0
    # Check rows and columns for 'virtual' neighbors
    for i in range(4):
        for line in [board[i, :], board[:, i]]: # Check row then column
            # Filter out the zeros to see what tiles would touch if moved
            filtered = line[line != 0]
            for j in range(len(filtered) - 1):
                if filtered[j] == filtered[j+1]:
                    # Reward potential merges based on their log value
                    bonus += np.log2(filtered[j]) * 20
    return bonus

def score_board(board: np.ndarray, weight_matrix: np.ndarray) -> int:
    """Calculate the score of the board with a given weight matrix."""

    # Convert tile values to their logarithmic form (base 2) for better scaling
    clean_board = np.where(board > 0, board, 1)
    log_board = np.log2(clean_board)
    log_board[board == 0] = 0

    # Weights in snake pattern
    C_WEIGHTS = 1.0
    C_SMOOTH = 16.0
    C_LONELY = 256.0
    C_MERGE = 4096.0
    C_MONO = 65536.0
    C_EMPTY = 1048576.0

    # 3. Calculate Components
    w_score = np.sum(log_board * weight_matrix) * C_WEIGHTS
    m_score = get_monotonicity(log_board) * C_MONO
    s_score = get_smoothness(log_board) * C_SMOOTH
    l_score = get_isolation_penalty(board) * C_LONELY
    b_score = get_merge_potential(board) * C_MERGE
    e_score = np.sum(board == 0) * C_EMPTY

    return w_score + m_score + s_score + l_score + b_score + e_score


import numpy as np


# (Assume your score_board and helper functions are pasted here)

def expectimax(board, depth, is_player_turn):
    """
    The recursive Expectimax algorithm.
    """

    state_key = (board.tobytes(), depth, is_player_turn)

    if state_key in TRANSPOSITION_TABLE:
        return TRANSPOSITION_TABLE[state_key]

    # ---------------------------------------------------------
    # BASE CASE (Leaf Node) -> evaluate the board with the heuristic
    # ---------------------------------------------------------
    if depth == 0 or not game.move_exists(board):
        # We use the heuristic we built in the previous task!
        return score_board(board, POWER_4_WEIGHT_MATRIX)

    # ---------------------------------------------------------
    # MAX NODE (Player's Turn) -> maximize the score
    # ---------------------------------------------------------
    if is_player_turn:
        best_score = -float('inf')

        for move in [0, 1, 2, 3]:  # UP, DOWN, LEFT, RIGHT
            new_board = execute_move(move, board)

            # filter vaild moves
            if not board_equals(board, new_board):
                # recursive call one level down -> game's turn (chance node)
                score = expectimax(new_board, depth, is_player_turn=False)
                best_score = max(best_score, score)

        # no valid moves -> dead end
        if best_score == -float('inf'):
            return score_board(board, POWER_4_WEIGHT_MATRIX)

        TRANSPOSITION_TABLE[state_key] = best_score  # Cache the result before returning

        return best_score

    # ---------------------------------------------------------
    # CHANCE NODE (Game's Turn - Tile Spawning)
    # ---------------------------------------------------------
    else:
        # Find the coordinates of all empty cells
        empty_cells = list(zip(*np.where(board == 0)))

        if not empty_cells:
            return score_board(board, POWER_4_WEIGHT_MATRIX)

        expected_score = 0
        num_empty = len(empty_cells)

        # test every possible spawn of 2 and 4 in each empty cell
        for r, c in empty_cells:
            # --- Simulate 2 (90% chance) ---
            board[r, c] = 2
            # Pass turn back to Player (Max Node), and NOW we decrease the depth!
            score_2 = expectimax(board, depth - 1, is_player_turn=True)

            if depth == 4:
                expected_score += (0.9 / num_empty) * score_2

            else:
                # Normal logic for higher depths
                expected_score += (0.9 / num_empty) * score_2

                board[r, c] = 4
                score_4 = expectimax(board, depth - 1, is_player_turn=True)
                expected_score += (0.1 / num_empty) * score_4

            # --- FIELD CORRECTION ---
            # Since we just put 2s and 4s on the board -> clean up
            board[r, c] = 0

        TRANSPOSITION_TABLE[state_key] = expected_score  # Cache the result before returning

        return expected_score

def find_best_move(board):
    """
    find the best move for the next turn.
    """
    bestmove = -1
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    move_args = [UP,DOWN,LEFT,RIGHT]
    
    result = [score_toplevel_move(i, board) for i in range(len(move_args))]
    bestmove = result.index(max(result))

    # for m in move_args:
    #     print("move: %d score: %.4f" % (m, result[m]))

    return bestmove
    
def score_toplevel_move(move, board):
    """
    Entry Point to score the first move.
    """
    newboard = execute_move(move, board)

    if board_equals(board,newboard):
        return -1e10  # Invalid move, return a very low score

    SEARCH_DEPTH = 3
    return expectimax(newboard, SEARCH_DEPTH, is_player_turn=False)

def execute_move(move, board):
    """
    move and return the grid without a new random tile 
	It won't affect the state of the game in the browser.
    """

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

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
    return  (newboard == board).all()  
