"""
Sokoban environment for Experiential Reinforcement Learning.

A 5x5 grid puzzle with standard push mechanics.  The agent sees the grid
and the list of valid actions; no rules or symbol meanings are provided.

Grid symbols
------------
  #   wall
  ' ' empty floor
  @   player
  $   box
  .   target
  *   box on target
  +   player on target
"""


class Sokoban:
    # fmt: off
    DEFAULT_LAYOUT = [
        ['#', '#', '#', '#', '#'],
        ['#', '.', '#', '#', '#'],
        ['#', '@', '$', '.', '#'],
        ['#', '#', '#', '#', '#'],
        ['#', '#', '#', '#', '#'],
    ]
    # fmt: on

    ACTIONS   = ["up", "down", "left", "right"]
    MAX_STEPS = 8
    ROWS      = 5
    COLS      = 5

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
        """
        Parse DEFAULT_LAYOUT into mutable state and return the initial
        observation string.
        """
        self.steps  = 0
        self.done   = False
        self.player = None          # (row, col)
        self.boxes  = set()         # set of (row, col)
        self.targets = set()        # set of (row, col)  — never changes
        self.floor  = {}            # (row, col) -> '#' | ' ' | '.'

        for r, row in enumerate(self.DEFAULT_LAYOUT):
            for c, cell in enumerate(row):
                if cell == '#':
                    self.floor[(r, c)] = '#'
                elif cell in ('.', '*', '+'):   # target square
                    self.floor[(r, c)] = '.'
                    self.targets.add((r, c))
                else:                           # empty floor
                    self.floor[(r, c)] = ' '

                if cell in ('@', '+'):
                    self.player = (r, c)
                if cell in ('$', '*'):
                    self.boxes.add((r, c))

        return self.get_observation()

    def get_observation(self):
        """
        Return a clean string that the LM agent sees.

        Contains: current grid, valid actions, and step count.
        No rules or symbol explanations.
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

        Push mechanics:  when the player walks into a box, the box is pushed
        one cell in the same direction.  A push fails if the destination cell
        is a wall, out of bounds, or occupied by another box (standard Sokoban).

        Parameters
        ----------
        actions : list[str]
            Sequence of actions to attempt.

        Returns
        -------
        final_state : str
            String representation of the current grid.
        feedback : str
            Natural-language description of every step that was executed.
        reward : int
            1 if all boxes are on targets, 0 otherwise.
        done : bool
            True when the episode has ended (win or max steps).
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

            dr, dc = self._DELTAS[action]
            pr, pc = self.player
            nr, nc = pr + dr, pc + dc       # candidate player position

            # ---- out of bounds -----------------------------------------
            if not self._in_bounds(nr, nc):
                parts.append(
                    f"Step {step_num}: moved {action}, hit the boundary, stayed in place."
                )
                self.steps += 1
                if self.steps >= self.MAX_STEPS:
                    self.done = True
                    parts.append("Maximum steps reached.")
                continue

            # ---- wall --------------------------------------------------
            if self._is_wall(nr, nc):
                parts.append(
                    f"Step {step_num}: moved {action}, hit a wall, stayed in place."
                )
                self.steps += 1
                if self.steps >= self.MAX_STEPS:
                    self.done = True
                    parts.append("Maximum steps reached.")
                continue

            # ---- box push ----------------------------------------------
            if (nr, nc) in self.boxes:
                bnr, bnc = nr + dr, nc + dc     # candidate box destination

                if (
                    not self._in_bounds(bnr, bnc)
                    or self._is_wall(bnr, bnc)
                    or (bnr, bnc) in self.boxes
                ):
                    parts.append(
                        f"Step {step_num}: moved {action}, "
                        "box cannot be pushed, move failed."
                    )
                    self.steps += 1
                    if self.steps >= self.MAX_STEPS:
                        self.done = True
                        parts.append("Maximum steps reached.")
                    continue

                # Execute push
                self.boxes.discard((nr, nc))
                self.boxes.add((bnr, bnc))
                self.player = (nr, nc)

                if (bnr, bnc) in self.targets:
                    parts.append(
                        f"Step {step_num}: moved {action}, pushed box onto a target."
                    )
                else:
                    parts.append(f"Step {step_num}: moved {action}, pushed box.")

            # ---- normal move -------------------------------------------
            else:
                self.player = (nr, nc)
                parts.append(f"Step {step_num}: moved {action}.")

            self.steps += 1

            # ---- win check ---------------------------------------------
            if self._all_boxes_on_targets():
                self.done = True
                reward = 1
                parts.append("All boxes are on targets. Puzzle solved!")
                break

            if self.steps >= self.MAX_STEPS and not self.done:
                self.done = True
                parts.append("Maximum steps reached.")

        feedback = " ".join(parts) if parts else "No actions executed."
        return self._render_grid(), feedback, reward, self.done

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _in_bounds(self, r, c):
        return 0 <= r < self.ROWS and 0 <= c < self.COLS

    def _is_wall(self, r, c):
        return self.floor.get((r, c), '#') == '#'

    def _all_boxes_on_targets(self):
        return all(b in self.targets for b in self.boxes)

    def _render_grid(self):
        # Abstract encoding — matches paper Appendix B.2:
        # A = agent on floor, a = agent on goal, B = box on floor,
        # b = box on goal,    C = goal tile,     D = floor, E = wall
        lines = []
        for r in range(self.ROWS):
            cells = []
            for c in range(self.COLS):
                pos  = (r, c)
                base = self.floor.get(pos, '#')

                if base == '#':
                    cells.append('E')
                elif pos == self.player:
                    cells.append('a' if base == '.' else 'A')
                elif pos in self.boxes:
                    cells.append('b' if base == '.' else 'B')
                else:
                    cells.append('C' if base == '.' else 'D')
            lines.append(' '.join(cells))
        return '\n'.join(lines)


# ======================================================================
# Quick smoke-test
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Sokoban Environment — Self-Test")
    print("=" * 50)

    env = Sokoban()
    print("\nInitial observation:")
    print(env.get_observation())

    # Verified solution:
    #   Player (2,1) moves right → pushes box from (2,2) to (2,3)=target → win
    solution = ["right"]
    print(f"\nExecuting solution: {solution}")

    final_state, feedback, reward, done = env.step(solution)

    print(f"\nStep-by-step feedback:\n  {feedback}")
    print(f"\nFinal grid:\n{final_state}")
    print(f"\nReward: {reward}  |  Done: {done}")

    assert reward == 1,  f"Expected reward=1, got {reward}"
    assert done is True, f"Expected done=True, got {done}"
    # Verify box is now on target (2,3): abstract symbol 'b' (box on goal)
    rows = final_state.split('\n')
    assert 'b' in rows[2], "Expected 'b' (box on goal) in row 2 of final grid"
    print("\n[PASS] All boxes on targets — reward=1 confirmed.")

    # --- Verify abstract symbols in the grid portion only ---
    # This layout has no empty floor (D); every non-wall cell is a
    # goal (C), player (A), or box (B).
    env.reset()
    grid_lines = env._render_grid()
    for sym in ('@', '$', '.', '*', '+', '#'):
        assert sym not in grid_lines, \
            f"Legacy symbol '{sym}' leaked into rendered grid"
    for sym in ('A', 'B', 'C', 'E'):  # D absent: no empty floor in this layout
        assert sym in grid_lines, \
            f"Abstract symbol '{sym}' missing from rendered grid"
    print("[PASS] Abstract symbol encoding correct.")

    # --- Verify player walking into wall is reported ---
    env.reset()
    _, fb_wall, _, _ = env.step(["left"])   # (2,1) left → (2,0) is wall
    assert "wall" in fb_wall, f"Expected wall message, got: {fb_wall}"
    print("[PASS] Movement into wall correctly reported.")

    # --- Verify loop terminates immediately after solve ---
    env2 = Sokoban()
    _, fb_early, r2, d2 = env2.step(["right", "up"])   # "up" should not run
    assert r2 == 1 and d2 is True, "Expected solved after first 'right'"
    assert fb_early.count("Step") == 1, "Expected only one step before done"
    print("[PASS] Loop terminates immediately after solve; extra actions ignored.")

    # --- Verify box cannot be pushed into a wall ---
    env3 = Sokoban()
    env3.player = (2, 2)
    env3.boxes = {(2, 3)}
    _, fb_blocked, r3, _ = env3.step(["right"])
    assert "cannot be pushed" in fb_blocked, \
        f"Expected blocked message, got: {fb_blocked}"
    print("[PASS] Box blocked by wall correctly reported.")

    # --- Max steps = 8 ---
    assert Sokoban.MAX_STEPS == 8, f"Expected MAX_STEPS=8, got {Sokoban.MAX_STEPS}"
    env4 = Sokoban()
    oscillate = ["up", "down"] * 4   # 8 steps total, no solve
    _, fb_max, r4, d4 = env4.step(oscillate)
    assert d4 is True, "Expected done=True after max steps"
    assert r4 == 0,    "Expected reward=0 without solving"
    assert "Maximum steps" in fb_max, "Expected max-steps message"
    print("[PASS] MAX_STEPS=8 and termination work correctly.")
