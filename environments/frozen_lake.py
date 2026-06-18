"""
FrozenLake environment for Experiential Reinforcement Learning.

A deterministic n×n grid world with n sampled from [2, 5] per episode
(paper arXiv 2602.13949, Appendix B.1).  The agent sees the grid and
the list of valid actions; no rules or symbol meanings are provided.

Grid symbols (abstract, matches paper Appendix B.1)
---------------------------------------------------
  A   player (agent)
  B   goal
  C   hole
  D   frozen tile (safe floor)
"""

import random
from collections import deque


class FrozenLake:
    ACTIONS   = ["Up", "Down", "Left", "Right"]
    MAX_STEPS = 8                       # Paper Appendix B.1: fixed step budget
    MIN_N     = 2
    MAX_N     = 5
    HOLE_PROB = 0.2                     # Density of C cells in interior

    _DELTAS = {
        "Up":    (-1,  0),
        "Down":  ( 1,  0),
        "Left":  ( 0, -1),
        "Right": ( 0,  1),
    }

    def __init__(self, seed: int = 0):
        """
        Parameters
        ----------
        seed : int
            Master seed.  The k-th call to ``reset()`` with no explicit
            seed uses ``seed + k`` so episode maps are reproducible and
            distinct.
        """
        self._master_seed  = seed
        self._episode_idx  = 0
        self._current_seed = None
        self._generate(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, seed: int = None):
        """
        Return to the start state.

        When ``seed`` is given, regenerate the grid with that seed.
        When ``seed`` is None, reset the agent position on the *current*
        grid (used when a caller wants to replay the current map).
        """
        if seed is not None:
            self._generate(seed)
        else:
            self.row, self.col = self._start
            self.steps = 0
            self.done  = False
        return self.get_observation()

    def get_observation(self):
        """
        Return a clean string that the LM agent sees.

        Contains: current grid (agent marked as 'A'), valid actions,
        and the current step count.  No rules or symbol explanations.
        """
        return (
            f"Grid:\n{self._render_grid()}\n\n"
            f"Valid actions: {self.ACTIONS}\n"
            f"Step: {self.steps}/{self.MAX_STEPS}"
        )

    def render(self):
        """Print the current grid to stdout."""
        print(self._render_grid())

    def step(self, actions):
        """
        Execute a sequence of action strings one step at a time.

        Returns (final_grid_str, feedback_str, reward, done).
        """
        if self.done:
            return self._render_grid(), "Episode already ended.", 0, True

        parts  = []
        reward = 0

        for action in actions:
            if self.done:
                break

            step_num = self.steps + 1

            if action not in self._DELTAS:
                parts.append(f"Step {step_num}: invalid action '{action}', skipped.")
                self.steps += 1
                if self.steps >= self.MAX_STEPS:
                    self.done = True
                    parts.append("Maximum steps reached.")
                continue

            dr, dc   = self._DELTAS[action]
            new_row  = self.row + dr
            new_col  = self.col + dc

            if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
                parts.append(
                    f"Step {step_num}: moved {action}, hit the boundary, stayed in place."
                )
                self.steps += 1
                if self.steps >= self.MAX_STEPS:
                    self.done = True
                    parts.append("Maximum steps reached.")
                continue

            self.row, self.col = new_row, new_col
            self.steps += 1
            cell = self._floor[self.row][self.col]

            if cell == 'C':
                parts.append(
                    f"Step {step_num}: moved {action}, stepped on C (hole), episode over."
                )
                self.done = True
                reward = 0
            elif cell == 'B':
                parts.append(f"Step {step_num}: moved {action}, reached B (goal)!")
                self.done = True
                reward = 1
            else:
                parts.append(
                    f"Step {step_num}: moved {action}, now on D (frozen tile)."
                )

            if self.steps >= self.MAX_STEPS and not self.done:
                self.done = True
                parts.append("Maximum steps reached.")

        feedback = " ".join(parts) if parts else "No actions executed."
        return self._render_grid(), feedback, reward, self.done

    # Backwards-compatible accessors (some callers still use ROWS/COLS)
    @property
    def ROWS(self): return self.rows
    @property
    def COLS(self): return self.cols

    # ------------------------------------------------------------------
    # Map generation
    # ------------------------------------------------------------------

    def _generate(self, seed):
        """Generate a solvable n×n grid using ``seed``."""
        self._current_seed = seed
        rng = random.Random(seed)

        # Sample grid size n ∈ [MIN_N, MAX_N]
        n = rng.randint(self.MIN_N, self.MAX_N)
        start = (0, 0)
        goal  = (n - 1, n - 1)

        # Retry until we get a solvable layout.  Fall back to hole-free.
        floor = None
        for _ in range(50):
            candidate = [['D'] * n for _ in range(n)]
            candidate[goal[0]][goal[1]] = 'B'
            for r in range(n):
                for c in range(n):
                    if (r, c) == start or (r, c) == goal:
                        continue
                    if rng.random() < self.HOLE_PROB:
                        candidate[r][c] = 'C'
            if self._is_solvable(candidate, start, goal, n):
                floor = candidate
                break

        if floor is None:
            floor = [['D'] * n for _ in range(n)]
            floor[goal[0]][goal[1]] = 'B'

        self._floor = floor
        self._start = start
        self._goal  = goal
        self.rows   = n
        self.cols   = n
        self.row, self.col = start
        self.steps  = 0
        self.done   = False
        self._episode_idx += 1

    @staticmethod
    def _is_solvable(floor, start, goal, n):
        """BFS: is goal reachable from start avoiding C cells?"""
        if start == goal:
            return True
        q = deque([start])
        seen = {start}
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                if (nr, nc) in seen:
                    continue
                if floor[nr][nc] == 'C':
                    continue
                if (nr, nc) == goal:
                    return True
                seen.add((nr, nc))
                q.append((nr, nc))
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _render_grid(self):
        lines = []
        for r in range(self.rows):
            cells = []
            for c in range(self.cols):
                if r == self.row and c == self.col:
                    cells.append('A')
                else:
                    cells.append(self._floor[r][c])
            lines.append(' '.join(cells))
        return '\n'.join(lines)


# ======================================================================
# Quick smoke-test
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("FrozenLake Environment — Self-Test")
    print("=" * 50)

    env = FrozenLake(seed=0)
    print(f"\nInitial observation (n={env.rows}, seed={env._current_seed}):")
    print(env.get_observation())

    # --- Verify solvability: every generated map must have a path ---
    for s in range(20):
        env2 = FrozenLake(seed=s)
        assert FrozenLake._is_solvable(env2._floor, env2._start, env2._goal, env2.rows), \
            f"Seed {s} produced an unsolvable map"
    print("[PASS] 20 random seeds all produce solvable maps.")

    # --- Verify abstract symbol set ---
    env.reset(seed=42)
    grid = env._render_grid()
    for sym in ('S', 'F', 'H', 'G', '*'):
        assert sym not in grid, f"Legacy semantic symbol '{sym}' leaked"
    for sym in ('A',):
        assert sym in grid, f"Agent symbol '{sym}' missing from grid"
    print("[PASS] Abstract symbol encoding correct.")

    # --- Verify grid size ∈ [2, 5] ---
    for s in range(30):
        env3 = FrozenLake(seed=s)
        assert FrozenLake.MIN_N <= env3.rows <= FrozenLake.MAX_N, \
            f"Seed {s}: n={env3.rows} out of [{FrozenLake.MIN_N},{FrozenLake.MAX_N}]"
    print("[PASS] All grid sizes in [2, 5].")

    # --- Verify seed=None reuses the same grid ---
    env4 = FrozenLake(seed=7)
    snapshot = env4._floor
    env4.step(["Right", "Right"])
    env4.reset()                                     # same map, reset agent
    assert env4._floor is snapshot, "reset() without seed must keep same grid"
    assert (env4.row, env4.col) == env4._start, "Agent not at start after reset()"
    print("[PASS] reset() without seed reuses current grid.")

    # --- Verify boundary clamping doesn't crash ---
    env5 = FrozenLake(seed=0)
    _, fb, _, _ = env5.step(["Up", "Left"])
    assert "boundary" in fb, f"Expected boundary message, got: {fb}"
    print("[PASS] Boundary clamping works.")

    # --- Verify MAX_STEPS == 8 (paper Appendix B.1) ---
    assert FrozenLake.MAX_STEPS == 8, f"Expected MAX_STEPS=8, got {FrozenLake.MAX_STEPS}"
    print("[PASS] MAX_STEPS=8 confirmed.")

    print("\n[ALL PASSED]")
