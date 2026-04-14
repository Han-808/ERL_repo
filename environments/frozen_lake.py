"""
FrozenLake environment for Experiential Reinforcement Learning.

A deterministic 4x4 grid world.  The agent sees the grid and the list of
valid actions; no rules or symbol meanings are provided to it.
"""


class FrozenLake:
    # fmt: off
    DEFAULT_MAP = [
        ['S', 'F', 'F', 'F'],
        ['F', 'H', 'F', 'H'],
        ['F', 'F', 'F', 'H'],
        ['H', 'F', 'F', 'G'],
    ]
    # fmt: on

    ACTIONS   = ["up", "down", "left", "right"]
    MAX_STEPS = 8
    ROWS      = 4
    COLS      = 4

    # Abstract encoding used in get_observation() — matches paper Appendix B.1:
    # A = agent, B = goal, C = hole, D = safe frozen tile
    _ABSTRACT = {'S': 'D', 'F': 'D', 'H': 'C', 'G': 'B'}

    _DELTAS = {
        "up":    (-1,  0),
        "down":  ( 1,  0),
        "left":  ( 0, -1),
        "right": ( 0,  1),
    }

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        """Return to the start state and return the initial observation."""
        self.row   = 0
        self.col   = 0
        self.steps = 0
        self.done  = False
        return self.get_observation()

    def get_observation(self):
        """
        Return a clean string that the LM agent sees.

        Contains: current grid (agent marked as '*'), valid actions,
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
        Execute a full trajectory (list of action strings) one step at a time.

        Parameters
        ----------
        actions : list[str]
            Sequence of actions to attempt, e.g. ["down", "right", ...].

        Returns
        -------
        final_state : str
            The grid with the agent's current position marked as '*'.
        feedback : str
            Natural-language description of every step that was executed.
        reward : int
            1 if the agent reached the goal, 0 otherwise.
        done : bool
            True when the episode has ended (goal, hole, or max steps).
        """
        if self.done:
            return self._render_grid(), "Episode already ended.", 0, True

        parts  = []
        reward = 0

        for action in actions:
            if self.done:
                break

            step_num = self.steps + 1

            # ---- invalid action ----------------------------------------
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

            # ---- out of bounds -----------------------------------------
            if not (0 <= new_row < self.ROWS and 0 <= new_col < self.COLS):
                parts.append(
                    f"Step {step_num}: moved {action}, hit the boundary, stayed in place."
                )
                self.steps += 1
                if self.steps >= self.MAX_STEPS:
                    self.done = True
                    parts.append("Maximum steps reached.")
                continue

            # ---- valid move --------------------------------------------
            self.row, self.col = new_row, new_col
            self.steps += 1
            cell = self.DEFAULT_MAP[self.row][self.col]

            if cell == 'H':
                parts.append(f"Step {step_num}: moved {action}, fell into a hole.")
                self.done = True
                reward = 0
            elif cell == 'G':
                parts.append(f"Step {step_num}: moved {action}, reached the goal!")
                self.done = True
                reward = 1
            else:
                parts.append(f"Step {step_num}: moved {action}.")

            if self.steps >= self.MAX_STEPS and not self.done:
                self.done = True
                parts.append("Maximum steps reached.")

        feedback = " ".join(parts) if parts else "No actions executed."
        return self._render_grid(), feedback, reward, self.done

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _render_grid(self):
        lines = []
        for r in range(self.ROWS):
            cells = []
            for c in range(self.COLS):
                if r == self.row and c == self.col:
                    cells.append('A')                        # agent
                else:
                    cells.append(self._ABSTRACT[self.DEFAULT_MAP[r][c]])
            lines.append(' '.join(cells))
        return '\n'.join(lines)


# ======================================================================
# Quick smoke-test
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("FrozenLake Environment — Self-Test")
    print("=" * 50)

    env = FrozenLake()
    print("\nInitial observation:")
    print(env.get_observation())

    # Verified optimal path (avoids all holes):
    #   (0,0) -d-> (1,0) -d-> (2,0) -r-> (2,1) -r-> (2,2) -d-> (3,2) -r-> (3,3)=G
    solution = ["down", "down", "right", "right", "down", "right"]
    print(f"\nExecuting solution: {solution}")

    final_state, feedback, reward, done = env.step(solution)

    print(f"\nStep-by-step feedback:\n  {feedback}")
    print(f"\nFinal grid:\n{final_state}")
    print(f"\nReward: {reward}  |  Done: {done}")

    assert reward == 1,    f"Expected reward=1, got {reward}"
    assert done is True,   f"Expected done=True, got {done}"
    # Verify abstract encoding: agent at goal renders as 'A', no semantic symbols visible
    assert 'A' in final_state, "Expected agent symbol 'A' in final grid"
    assert 'S' not in final_state and 'G' not in final_state, \
        "Semantic symbols must not appear in rendered output"

    print("\n[PASS] Agent reached the goal — reward=1 confirmed.")

    # --- Verify abstract symbols in the grid portion only ---
    env.reset()
    grid_lines = env._render_grid()
    for sym in ('S', 'F', 'H', 'G', '*'):
        assert sym not in grid_lines, \
            f"Semantic symbol '{sym}' leaked into rendered grid"
    for sym in ('A', 'B', 'C', 'D'):
        assert sym in grid_lines, \
            f"Abstract symbol '{sym}' missing from rendered grid"
    print("[PASS] Abstract symbol encoding correct.")

    # --- Verify that falling in a hole gives reward=0 ---
    env.reset()
    _, _, r_hole, d_hole = env.step(["right", "down"])   # (0,0)->(0,1)->(1,1)=H
    assert r_hole == 0 and d_hole is True, "Expected hole → reward=0, done=True"
    print("[PASS] Falling into a hole gives reward=0 confirmed.")

    # --- Verify boundary clamping does not crash ---
    env.reset()
    _, fb_wall, _, _ = env.step(["up", "left"])           # both should be blocked
    assert "boundary" in fb_wall, "Expected boundary message"
    print("[PASS] Boundary clamping works correctly.")

    # --- Verify MAX_STEPS == 8 ---
    assert FrozenLake.MAX_STEPS == 8, f"Expected MAX_STEPS=8, got {FrozenLake.MAX_STEPS}"
    print("[PASS] MAX_STEPS=8 confirmed.")
