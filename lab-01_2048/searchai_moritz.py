import random
import game
import sys

TRANSPOSITION_TABLE = {} # cache for already evaluated board states
# Fast log2 lookup for tiles up to 2^16
LOG_MAP = {2**i: i for i in range(1, 17)}
LOG_MAP[0] = 0

# Author:      chrn (original by nneonneo)
# Date:        11.11.2016
# Copyright:   Algorithm from https://github.com/nneonneo/2048-ai
# Description: The logic to beat the game. Based on expectimax algorithm.

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
            # Row check
            if log_board[i][j] >= log_board[i][j + 1]: score += 1
            # Column check (log_board[row][col])
            if log_board[j][i] >= log_board[j + 1][i]: score += 1
    return score

def get_smoothness(log_board):
    """
    Calculate smoothness penalty for rows and columns.
    -> penalize large differences between adjacent tiles (cliffs).
    """
    penalty = 0
    for i in range(4):
        for j in range(3):
            # Horizontal
            penalty -= (abs(log_board[i][j] - log_board[i][j + 1]) ** 2)
            # Vertical
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
    for r in range(4):
        for c in range(4):
            val = board[r][c]
            if val == 0: continue

            has_neighbor = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 4 and 0 <= nc < 4:
                    neighbor = board[nr][nc]
                    if neighbor == val or neighbor == val * 2 or neighbor == val // 2:
                        has_neighbor = True
                        break
            if not has_neighbor:
                penalty -= LOG_MAP.get(val, 0) * 10
    return penalty

def get_merge_potential(board):
    """
    Calculate merge potential bonus for tiles that have 'virtual' neighbors.
    -> A tile has 'virtual' neighbors if there are non-zero tiles in the same row or column that could merge with it if they were moved together.
    """
    bonus = 0
    for i in range(4):
        # 1. Extract the row and column natively
        row = board[i]
        col = [board[x][i] for x in range(4)]

        for line in [row, col]:
            # 2. Filter out the zeros natively
            filtered = [val for val in line if val != 0]

            # 3. Check for virtual neighbors
            for j in range(len(filtered) - 1):
                if filtered[j] == filtered[j + 1]:
                    # 4. Use the fast lookup map instead of np.log2
                    # Make sure LOG_MAP is defined at the top of your file!
                    bonus += LOG_MAP.get(filtered[j], 0) * 20

    return bonus

def score_board(board, weight_matrix) -> int:
    """Calculate the score of the board with a given weight matrix."""

    # Convert tile values to their logarithmic form (base 2) for better scaling
    log_board = [[LOG_MAP.get(cell, 0) for cell in row] for row in board]

    w_score = 0
    empty_count = 0
    for r in range(4):
        for c in range(4):
            w_score += log_board[r][c] * weight_matrix[r][c]
            if board[r][c] == 0:
                empty_count += 1

    # Weights in snake pattern
    # C_WEIGHTS = 1.0
    C_SMOOTH = 16.0
    C_LONELY = 256.0
    C_MERGE = 4096.0
    C_MONO = 65536.0
    C_EMPTY = 1048576.0

    # 3. Calculate Components
    # w_score = np.sum(log_board * weight_matrix) * C_WEIGHTS
    m_score = get_monotonicity(log_board) * C_MONO
    s_score = get_smoothness(log_board) * C_SMOOTH
    l_score = get_isolation_penalty(board) * C_LONELY
    b_score = get_merge_potential(board) * C_MERGE
    e_score = empty_count * C_EMPTY

    return w_score + m_score + s_score + l_score + b_score + e_score

# (Assume your score_board and helper functions are pasted here)

def expectimax(board, depth, is_player_turn):
    """
    The recursive Expectimax algorithm.
    """

    board_tuple = tuple(tuple(row) for row in board)
    state_key = (board_tuple, depth, is_player_turn)

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
        empty_cells = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]

        if not empty_cells:
            return score_board(board, POWER_4_WEIGHT_MATRIX)

        expected_score = 0
        num_empty = len(empty_cells)

        # test every possible spawn of 2 and 4 in each empty cell
        for r, c in empty_cells:
            # --- Simulate 2 (90% chance) ---
            board[r][c] = 2
            # Pass turn back to Player (Max Node), and NOW we decrease the depth!
            score_2 = expectimax(board, depth - 1, is_player_turn=True)
            expected_score += (0.9 / num_empty) * score_2

            # --- Simulate 4 (10% chance) ---
            board[r][c] = 4
            score_4 = expectimax(board, depth - 1, is_player_turn=True)
            expected_score += (0.1 / num_empty) * score_4

            # --- FIELD CORRECTION ---
            # Since we just put 2s and 4s on the board -> clean up
            board[r][c] = 0

        TRANSPOSITION_TABLE[state_key] = expected_score  # Cache the result before returning

        return expected_score

def find_best_move(board):
    """
    find the best move for the next turn.
    """

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    move_args = [UP,DOWN,LEFT,RIGHT]
    base_score = score_board(board, POWER_4_WEIGHT_MATRIX)

    valid_moves = []
    for move in [UP, DOWN, LEFT, RIGHT]:
        newboard = execute_move(move, board)
        if not board_equals(board, newboard):
            # Do a quick immediate check
            immediate_score = score_board(newboard, POWER_4_WEIGHT_MATRIX)

            # If the move drops our score by a massive amount (e.g., 20%), skip the deep search!
            if immediate_score < (base_score * 0.8):
                continue

            valid_moves.append(move)

    bestmove = -1

    best_score = -float('inf')
    for move in valid_moves:
        score = score_toplevel_move(move, board)
        if score > best_score:
            best_score = score
            bestmove = move

    # Fallback if everything was pruned
    return bestmove if bestmove != -1 else random.choice([UP, DOWN, LEFT, RIGHT])
    
def score_toplevel_move(move, board):
    """
    Entry Point to score the first move.
    """
    newboard = execute_move(move, board)

    if board_equals(board,newboard):
        return -1e10  # Invalid move, return a very low score

    empty_count = sum(row.count(0) for row in newboard)

    # Dynamically adjust search depth based on the number of empty cells
    if empty_count >= 10:
        SEARCH_DEPTH = 2  # Open board, shallow search
    elif empty_count >= 6:
        SEARCH_DEPTH = 3  # Standard
    elif empty_count >= 3:
        SEARCH_DEPTH = 4  # Deep search
    elif empty_count >= 1:
        SEARCH_DEPTH = 5  # Survival mode!
    else:
        SEARCH_DEPTH = 6  # Either die or win.

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
    return newboard == board
