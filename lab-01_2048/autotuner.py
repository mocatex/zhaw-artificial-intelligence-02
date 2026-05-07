import random
import json
import copy
import time
import searchai_moritz as ai  # Make sure this matches your filename!
import game
import logging

# --- Setup Logging ---
# This creates a file called 'autotuner.log' and writes everything to it.
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("autotuner.log"),
        logging.StreamHandler() # Also prints to the console
    ]
)

def log(message):
    logging.info(message)

# Native Tile Spawner
def add_random_tile(board):
    empty_cells = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
    if not empty_cells:
        return
    r, c = random.choice(empty_cells)
    board[r][c] = 2 if random.random() < 0.9 else 4


# Headless Game Loop
def play_headless_game(weights):
    # Inject the mutated weights into the AI
    ai.HEURISTIC_WEIGHTS = weights

    # Clear the cache so PyPy doesn't run out of memory
    ai.TRANSPOSITION_TABLE.clear()

    board = [[0] * 4 for _ in range(4)]
    add_random_tile(board)
    add_random_tile(board)

    moves_made = 0
    while game.move_exists(board):
        move = ai.find_best_move(board)
        if move == -1: break

        if move == 0:
            new_board = game.merge_up(board)
        elif move == 1:
            new_board = game.merge_down(board)
        elif move == 2:
            new_board = game.merge_left(board)
        elif move == 3:
            new_board = game.merge_right(board)

        if new_board == board:
            break  # AI got stuck

        board = new_board
        add_random_tile(board)
        moves_made += 1

    # We use the total sum of tiles as a fast proxy for the game score
    total_sum = sum(sum(row) for row in board)
    max_tile = max(max(row) for row in board)
    return total_sum, max_tile


# The Auto-Tuner (Stochastic Hill Climbing)
def run_autotuner():
    log("🚀 Starting 2048 Expectimax Auto-Tuner...")

    baseline_weights = ai.HEURISTIC_WEIGHTS.copy()
    best_avg_score = 0

    # We play 3 games per generation to average out RNG luck
    GAMES_PER_GEN = 4

    for generation in range(1, 1000):
        log(f"\n--- Generation {generation} ---")

        # Mutate the weights
        test_weights = copy.deepcopy(baseline_weights)
        for key in test_weights:
            # Mutate between -20% and +20%
            mutation_factor = random.uniform(0.8, 1.2)
            test_weights[key] *= mutation_factor

        log(f"Testing new mutation...")

        scores = []
        max_tiles = []

        for i in range(GAMES_PER_GEN):
            start_time = time.time()
            score, max_t = play_headless_game(test_weights)
            duration = time.time() - start_time

            scores.append(score)
            max_tiles.append(max_t)
            log(f"  Game {i + 1}: Sum = {score}, Max Tile = {max_t} (Took {duration:.1f}s)")

        avg_score = sum(scores) / len(scores)

        if avg_score > best_avg_score:
            log(f"🏆 NEW HIGH SCORE! Avg Sum: {avg_score}")
            best_avg_score = avg_score
            baseline_weights = test_weights

            # Save the winning weights to a file so you don't lose them!
            with open("best_weights.json", "w") as f:
                json.dump(baseline_weights, f, indent=4)
            log("Saved to best_weights.json")
        else:
            log(f"X Mutation failed (Avg: {avg_score} < Best: {best_avg_score}). Discarding.")


if __name__ == "__main__":
    run_autotuner()