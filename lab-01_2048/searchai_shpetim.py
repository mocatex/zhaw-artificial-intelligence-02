import random
import game
import sys
import numpy as np

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

PROB_2 = 0.9
PROB_4 = 0.1
DEPTH = 2
MAX_CHANCE_CELLS = 3


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def score_board(board: np.ndarray) -> float:
    max_tile = int(np.max(board))

    # 1. Corner bonus
    corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
    corner_bonus = max_tile * 3 if max_tile in corners else 0

    # 2. Empty cells — heavily weighted to avoid getting stuck
    empty = int(np.sum(board == 0))
    empty_bonus = empty * 150

    # 3. Monotonicity — best of all 4 rotations
    def mono_score(b):
        score = 0
        for row in b:
            for i in range(3):
                if row[i] >= row[i + 1]:
                    score += row[i] - row[i + 1]
        return score

    mono = max(
        mono_score(board),
        mono_score(np.rot90(board, 1)),
        mono_score(np.rot90(board, 2)),
        mono_score(np.rot90(board, 3)),
    )

    # 4. Merge bonus — reward adjacent equal tiles (easy merges available)
    merge = 0
    for i in range(4):
        for j in range(3):
            if board[i, j] == board[i, j+1] and board[i, j] != 0:
                merge += board[i, j]
            if board[j, i] == board[j+1, i] and board[j, i] != 0:
                merge += board[j, i]

    return corner_bonus + empty_bonus + mono + merge * 2


# ---------------------------------------------------------------------------
# Expectimax
# ---------------------------------------------------------------------------

def expectimax(board: np.ndarray, depth: int, is_player_turn: bool) -> float:

    if depth == 0:
        return score_board(board)

    if is_player_turn:
        best = -1
        for move in [UP, DOWN, LEFT, RIGHT]:
            new_board = execute_move(move, board)
            if not board_equals(board, new_board):
                score = expectimax(new_board, depth - 1, is_player_turn=False)
                if score > best:
                    best = score
        return best if best != -1 else score_board(board)

    else:
        empty_cells = list(zip(*np.where(board == 0)))

        if not empty_cells:
            return score_board(board)

        # Sample a fixed small number of cells — keeps branching factor constant
        sampled = random.sample(empty_cells, min(MAX_CHANCE_CELLS, len(empty_cells)))

        total = 0.0
        for (r, c) in sampled:
            for tile_value, prob in [(2, PROB_2), (4, PROB_4)]:
                new_board = board.copy()
                new_board[r, c] = tile_value
                total += (prob / len(sampled)) * expectimax(new_board, depth, is_player_turn=True)

        return total


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def find_best_move(board):
    best_move = -1
    best_score = -1

    for move in [UP, DOWN, LEFT, RIGHT]:
        score = score_toplevel_move(move, board)
        print("move: %d  score: %.4f" % (move, score))
        if score > best_score:
            best_score = score
            best_move = move

    return best_move


def score_toplevel_move(move, board):
    new_board = execute_move(move, board)
    if board_equals(board, new_board):
        return 0
    return expectimax(new_board, depth=DEPTH, is_player_turn=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def execute_move(move, board):
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
    return (newboard == board).all()