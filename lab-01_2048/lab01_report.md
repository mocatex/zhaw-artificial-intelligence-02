## 🏗️ 1. Core Architecture: The Expectimax Algorithm

Unlike Minimax (used for Chess), 2048 is a game of chance. Our agent uses **Expectimax** to handle the non-deterministic nature of the game (the random tile spawns).

* **Max Nodes (Agent's Turn):** The AI evaluates the 4 possible moves (UP, DOWN, LEFT, RIGHT) and chooses the one that maximizes the expected future score.
* **Chance Nodes (Game's Turn):** The AI simulates every possible spawn of a `2` (90% probability) or a `4` (10% probability) in every empty cell. It calculates the **weighted average** of these outcomes.

---

## 🚀 2. Performance Engineering: The PyPy Rewrite

Initially, our search was limited to **Depth 2** because Python's standard interpreter was too slow. To achieve **Depth 4-6**, we implemented a two-pronged optimization strategy:

1. **Eliminating NumPy Overhead:** For tiny $4 \times 4$ matrices, NumPy's C-extension overhead is actually slower than native Python. We rewrote the entire game engine (`game.py`) and the heuristic engine using **pure Python lists and list comprehensions**.
2. **PyPy JIT Compilation:** By running our "De-NumPy-ed" code with the **PyPy interpreter**, we leveraged its Just-In-Time (JIT) compiler. This translated our Python loops into optimized machine code, resulting in a **~15x speedup**.

---

## 🧠 3. Search Optimizations

To reach deeper into the future without a computational explosion, we added several clever search techniques:

* **Transposition Table:** We cache already-evaluated board states in a dictionary (`TRANSPOSITION_TABLE`). If the AI sees the same board twice via different move orders, it returns the score instantly.
* **Dynamic Search Depth:** The branching factor of 2048 changes based on how full the board is. Our AI uses "geared" depth:
* **Open Board (10+ empty):** Depth 2 (Cruising)
* **Danger Zone (<3 empty):** Depth 5-6 (Survival)


* **The "Bad Move" Realization:** We initially tried to prune moves that dropped the heuristic score significantly. However, we discovered this was **self-sabotage**. In the late game, the AI *must* be allowed to temporarily break its "Snake" pattern to execute massive merges (e.g., combining two 4096 tiles). We removed this pruning to allow for high-level tactical sacrifices.

---

## 🎨 4. The Heuristic Scoring System

The "Brain" of the AI at the leaf nodes uses a multi-component heuristic:

1. **Monotonicity:** Ensures tiles decrease in value along a "snake" path.
2. **Smoothness:** Penalizes large value jumps between adjacent tiles.
3. **Isolation Penalty:** Discourages "lonely" high-value tiles that are hard to merge.
4. **Merge Potential:** Rewards tiles that *could* merge if moved together.
5. **Weight Matrix:** A geometric $2^X$ power matrix that provides "gravity" to keep the largest tile in the corner.

---

## 🧬 5. Self-Optimization: Stochastic Hill Climbing

Tuning 6 coefficients by hand is impossible. We built an **Auto-Tuner** script that uses a **Stochastic Hill Climbing** algorithm:

* It takes a "baseline" set of weights.
* It **mutates** those weights by $\pm 20\%$.
* It plays 3 headless games (fast, no browser) to average out luck.
* If the new weights perform better, they become the new baseline and are saved to `best_weights.json`.

---

## Expectimax vs our own heuristic

Our own heuristic in Task 3 could only look 1 step ahead. So it becomes a greedy algorithm that picks the move with the best immediate score.
We then learned about the phenomenon called **"Horizon Effect"** where in the short term, a move might look good, yet since our horizon is capped at 1, we can't see the consequences of that move.

The same was with our first implementation of Expectimax, where we had a depth of 2. It was better than the heuristic, but it still suffered from the Horizon Effect.

---

## 📊 Summary for the Professor

By combining a **probabilistic search algorithm** with **JIT-compiled native Python** and **automated parameter tuning**, we created an agent that doesn't just play 2048—it optimizes for it. The result is an agent capable of looking **6 steps ahead** in critical situations, easily surpassing the 100,000-point threshold.

**Would you like me to generate a specific set of PowerPoint-style bullet points based on this report for your colleague's presentation?**