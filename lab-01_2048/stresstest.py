# import numpy as np
import random
import game
import heuristicai_moritz as ai # custom heuristic AI for Task 3
# import searchai_moritz as ai # expectimax search AI for Task 4
import time


def add_random_tile(board):
    """Simulates the game spawning a new tile (90% chance of 2, 10% of 4)"""
    empty_cells = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
    if not empty_cells:
        return board
    r, c = random.choice(empty_cells)
    board[r][c] = 2 if random.random() < 0.9 else 4
    return board


def simulate_game():
    # Initialize an empty board and add two tiles
    board = [[0] * 4 for _ in range(4)]
    board = add_random_tile(add_random_tile(board))

    score = 0
    while True:
        # 1. Ask the AI for the best move
        # move = ai.find_best_move_random_agent() # random agent test
        move = ai.find_best_move(board)  # Use this for your heuristic or search agent
        if move < 0: break

        # 2. Execute move on a temporary board to see if it's valid
        new_board = ai.execute_move(move, board)

        # 3. If the board didn't change, the move was invalid
        if ai.board_equals(board, new_board):
            # Try other moves if the "best" one is blocked (simple fallback)
            valid_move_found = False
            for fallback in [0, 1, 2, 3]:
                test_board = ai.execute_move(fallback, board)
                if not ai.board_equals(board, test_board):
                    new_board = test_board
                    valid_move_found = True
                    break
            if not valid_move_found: break  # No moves left

        # 4. Update board and add a random tile
        board = add_random_tile(new_board)

    return max(max(row) for row in board), score

def run_stress_test(games=200):
    results = []
    print(f"Starting {games} headless simulations...")
    start_time = time.time()

    for i in range(games):
        max_tile, _ = simulate_game()
        results.append(max_tile)

        if (i + 1) % 25 == 0:
            print(f"Completed {i + 1} games...")

    end_time = time.time()
    print("\n--- RESULTS ---")
    print(f"Time taken: {end_time - start_time:.2f}s")
    print(f"Average Max Tile: {sum(results) / len(results)}")
    print(f"Best Tile Reached: {max(results)}")

    # Optional: Count how many times you hit 512, 1024, etc.
    for threshold in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        count = sum(1 for r in results if r >= threshold)
        print(f"Reached {threshold}+: {count}/{games} ({count / games * 100}%)")


if __name__ == "__main__":
    run_stress_test(500)